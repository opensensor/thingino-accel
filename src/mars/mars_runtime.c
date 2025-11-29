/*
 * Mars Runtime - Execute .mars models on Ingenic T41 NNA
 *
 * Copyright (c) 2024 OpenSensor Project
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>

#include "mars.h"
#include "mars_runtime.h"
#include "nna.h"
#include "device_internal.h"

/* Align value up to alignment */
#define ALIGN_UP(x, align) (((x) + (align) - 1) & ~((align) - 1))

/* Error strings */
static const char* error_strings[] = {
    "OK",
    "Invalid magic number",
    "Version mismatch",
    "Memory allocation failed",
    "Invalid file format",
    "NNA initialization failed",
    "Layer execution failed",
    "Invalid tensor",
    "Invalid layer",
};

const char* mars_get_error_string(mars_error_t err) {
    int idx = -err;
    if (idx >= 0 && idx < (int)(sizeof(error_strings)/sizeof(error_strings[0]))) {
        return error_strings[idx];
    }
    return "Unknown error";
}

/* Calculate tensor size in bytes */
static size_t tensor_byte_size(const mars_tensor_t *t) {
    size_t elem_size;
    switch (t->dtype) {
        case MARS_DTYPE_FLOAT32: elem_size = 4; break;
        case MARS_DTYPE_INT32:   elem_size = 4; break;
        case MARS_DTYPE_INT16:   elem_size = 2; break;
        case MARS_DTYPE_INT8:    elem_size = 1; break;
        case MARS_DTYPE_UINT8:   elem_size = 1; break;
        case MARS_DTYPE_UINT4:   elem_size = 1; break;  /* 2 elements per byte */
        default: elem_size = 1;
    }

    size_t numel = 1;
    for (uint32_t i = 0; i < t->ndims; i++) {
        numel *= t->shape[i];
    }

    if (t->dtype == MARS_DTYPE_UINT4) {
        return (numel + 1) / 2;  /* 4-bit packed */
    }
    return numel * elem_size;
}

mars_error_t mars_load_memory(const void *data, size_t size, mars_model_t **out_model) {
    if (!data || !out_model || size < sizeof(mars_header_t)) {
        return MARS_ERR_INVALID_FILE;
    }

    const uint8_t *ptr = (const uint8_t *)data;

    /* Parse header */
    mars_header_t header;
    memcpy(&header, ptr, sizeof(header));
    ptr += sizeof(header);

    /* Validate magic */
    if (header.magic != MARS_MAGIC) {
        fprintf(stderr, "Mars: Invalid magic 0x%08x (expected 0x%08x)\n",
                header.magic, MARS_MAGIC);
        return MARS_ERR_INVALID_MAGIC;
    }

    /* Check version */
    if (header.version_major != MARS_VERSION_MAJOR) {
        fprintf(stderr, "Mars: Version mismatch %d.%d (expected %d.x)\n",
                header.version_major, header.version_minor,
                MARS_VERSION_MAJOR);
        return MARS_ERR_VERSION_MISMATCH;
    }

    /* Allocate model context */
    mars_model_t *model = (mars_model_t *)calloc(1, sizeof(mars_model_t));
    if (!model) {
        return MARS_ERR_ALLOC_FAILED;
    }

    memcpy(&model->header, &header, sizeof(header));

    /* Allocate tensors array */
    model->tensors = (mars_runtime_tensor_t *)calloc(header.num_tensors,
                                                       sizeof(mars_runtime_tensor_t));
    if (!model->tensors) {
        free(model);
        return MARS_ERR_ALLOC_FAILED;
    }

    /* Read tensor descriptors */
    for (uint32_t i = 0; i < header.num_tensors; i++) {
        memcpy(&model->tensors[i].desc, ptr, sizeof(mars_tensor_t));
        ptr += sizeof(mars_tensor_t);
    }

    /* Allocate layers array */
    model->layers = (mars_runtime_layer_t *)calloc(header.num_layers,
                                                     sizeof(mars_runtime_layer_t));
    if (!model->layers) {
        free(model->tensors);
        free(model);
        return MARS_ERR_ALLOC_FAILED;
    }

    /* Read layer descriptors */
    for (uint32_t i = 0; i < header.num_layers; i++) {
        memcpy(&model->layers[i].desc, ptr, sizeof(mars_layer_t));
        ptr += sizeof(mars_layer_t);
    }

    /* Get NNA resources */
    model->ddr_base = nna_device_get_ddr();
    model->ddr_paddr = (void *)(uintptr_t)nna_device_get_ddr_pbase();
    model->ddr_size = 8 * 1024 * 1024;  /* 8MB */
    model->oram_base = nna_device_get_oram();
    model->oram_paddr = model->oram_base;  /* TODO: get actual paddr */
    model->oram_size = 384 * 1024;  /* 384KB */

    /* Load weights into DDR */
    if (header.weights_size > 0) {
        const uint8_t *weights_src = (const uint8_t *)data + header.weights_offset;
        model->weights_size = header.weights_size;

        if (model->weights_size > model->ddr_size) {
            fprintf(stderr, "Mars: Weights too large (%zu > %zu)\n",
                    model->weights_size, model->ddr_size);
            free(model->layers);
            free(model->tensors);
            free(model);
            return MARS_ERR_ALLOC_FAILED;
        }

        /* Copy weights to DDR */
        memcpy(model->ddr_base, weights_src, model->weights_size);
        model->weights = model->ddr_base;
    }

    /* Set up tensor data pointers */
    uint8_t *ddr_ptr = (uint8_t *)model->ddr_base + model->weights_size;
    size_t ddr_remaining = model->ddr_size - model->weights_size;

    for (uint32_t i = 0; i < header.num_tensors; i++) {
        mars_runtime_tensor_t *rt = &model->tensors[i];

        /* Check if tensor has weight data (data_size > 0 means it's stored in weights section) */
        if (rt->desc.data_size > 0) {
            rt->vaddr = (uint8_t *)model->ddr_base + rt->desc.data_offset;
            rt->paddr = (uint8_t *)model->ddr_paddr + rt->desc.data_offset;
            rt->alloc_size = rt->desc.data_size;
            continue;
        }

        /* Allocate runtime tensor in DDR */
        size_t needed = ALIGN_UP(tensor_byte_size(&rt->desc), 64);
        if (needed > ddr_remaining) {
            fprintf(stderr, "Mars: Out of DDR memory for tensor %u\n", i);
            free(model->layers);
            free(model->tensors);
            free(model);
            return MARS_ERR_ALLOC_FAILED;
        }

        rt->vaddr = ddr_ptr;
        rt->paddr = (uint8_t *)model->ddr_paddr + (ddr_ptr - (uint8_t *)model->ddr_base);
        rt->alloc_size = needed;
        ddr_ptr += needed;
        ddr_remaining -= needed;
    }

    *out_model = model;
    return MARS_OK;
}

mars_error_t mars_load_file(const char *path, mars_model_t **model) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Mars: Cannot open %s\n", path);
        return MARS_ERR_INVALID_FILE;
    }

    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    void *data = malloc(size);
    if (!data) {
        fclose(fp);
        return MARS_ERR_ALLOC_FAILED;
    }

    if (fread(data, 1, size, fp) != (size_t)size) {
        free(data);
        fclose(fp);
        return MARS_ERR_INVALID_FILE;
    }
    fclose(fp);

    mars_error_t err = mars_load_memory(data, size, model);
    free(data);  /* Model copies what it needs */
    return err;
}

void mars_free(mars_model_t *model) {
    if (!model) return;
    free(model->layers);
    free(model->tensors);
    free(model);
}

mars_runtime_tensor_t* mars_get_input(mars_model_t *model, int index) {
    if (!model || index < 0 || (uint32_t)index >= model->header.num_inputs) {
        return NULL;
    }
    uint32_t tid = model->header.input_tensor_ids[index];
    if (tid >= model->header.num_tensors) return NULL;
    return &model->tensors[tid];
}

mars_runtime_tensor_t* mars_get_output(mars_model_t *model, int index) {
    if (!model || index < 0 || (uint32_t)index >= model->header.num_outputs) {
        return NULL;
    }
    uint32_t tid = model->header.output_tensor_ids[index];
    if (tid >= model->header.num_tensors) return NULL;
    return &model->tensors[tid];
}

int mars_get_num_inputs(mars_model_t *model) {
    return model ? model->header.num_inputs : 0;
}

int mars_get_num_outputs(mars_model_t *model) {
    return model ? model->header.num_outputs : 0;
}

void mars_print_summary(mars_model_t *model) {
    if (!model) return;

    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║  Mars Model Summary                                      ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    printf("Layers: %u\n", model->header.num_layers);
    printf("Tensors: %u\n", model->header.num_tensors);
    printf("Inputs: %u\n", model->header.num_inputs);
    printf("Outputs: %u\n", model->header.num_outputs);
    printf("Weights: %zu bytes\n", model->weights_size);
    printf("\n");
}

/* Forward declaration for layer execution */
static mars_error_t execute_layer(mars_model_t *model, mars_runtime_layer_t *layer);

mars_error_t mars_run(mars_model_t *model) {
    if (!model) return MARS_ERR_INVALID_FILE;

    /* Reset execution flags */
    for (uint32_t i = 0; i < model->header.num_layers; i++) {
        model->layers[i].is_executed = false;
    }

    /* Execute layers in order */
    for (uint32_t i = 0; i < model->header.num_layers; i++) {
        mars_error_t err = execute_layer(model, &model->layers[i]);
        if (err != MARS_OK) {
            fprintf(stderr, "Mars: Layer %u execution failed\n", i);
            return err;
        }
        model->layers[i].is_executed = true;
    }

    model->inference_count++;
    return MARS_OK;
}

/* Software INT8 convolution - NHWC format */
static void conv2d_int8_sw(
    const int8_t *input, int in_h, int in_w, int in_c,
    const int8_t *weight, int out_c, int kh, int kw,
    const int32_t *bias,
    int8_t *output, int out_h, int out_w,
    int stride_h, int stride_w,
    int pad_top, int pad_left,
    float in_scale, float w_scale, float out_scale)
{
    /* Quantization: out = clamp((sum * in_scale * w_scale) / out_scale) */
    float combined_scale = (in_scale * w_scale) / out_scale;

    for (int oc = 0; oc < out_c; oc++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                int32_t sum = bias ? bias[oc] : 0;

                for (int khi = 0; khi < kh; khi++) {
                    for (int kwi = 0; kwi < kw; kwi++) {
                        int ih = oh * stride_h - pad_top + khi;
                        int iw = ow * stride_w - pad_left + kwi;

                        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            for (int ic = 0; ic < in_c; ic++) {
                                /* NHWC input: [batch, h, w, c] */
                                int in_idx = ih * in_w * in_c + iw * in_c + ic;
                                /* OHWI weight: [out_c, kh, kw, in_c] */
                                int w_idx = oc * kh * kw * in_c + khi * kw * in_c + kwi * in_c + ic;
                                sum += (int32_t)input[in_idx] * (int32_t)weight[w_idx];
                            }
                        }
                    }
                }

                /* Apply scale and clamp to int8 range */
                float scaled = sum * combined_scale;
                int32_t result = (int32_t)(scaled + 0.5f);
                if (result > 127) result = 127;
                if (result < -128) result = -128;

                /* NHWC output */
                int out_idx = oh * out_w * out_c + ow * out_c + oc;
                output[out_idx] = (int8_t)result;
            }
        }
    }
}

/* Execute Conv2D layer */
static mars_error_t execute_conv2d(mars_model_t *model, mars_runtime_layer_t *layer) {
    const mars_layer_t *desc = &layer->desc;
    const mars_conv_params_t *params = &desc->params.conv;

    /* Get input tensor */
    uint32_t in_id = desc->input_tensor_ids[0];
    mars_runtime_tensor_t *input = NULL;
    for (uint32_t i = 0; i < model->header.num_tensors; i++) {
        if (model->tensors[i].desc.id == in_id) {
            input = &model->tensors[i];
            break;
        }
    }
    if (!input || !input->vaddr) return MARS_ERR_INVALID_TENSOR;

    /* Get output tensor */
    uint32_t out_id = desc->output_tensor_ids[0];
    mars_runtime_tensor_t *output = NULL;
    for (uint32_t i = 0; i < model->header.num_tensors; i++) {
        if (model->tensors[i].desc.id == out_id) {
            output = &model->tensors[i];
            break;
        }
    }
    if (!output || !output->vaddr) return MARS_ERR_INVALID_TENSOR;

    /* Get weight tensor */
    uint32_t w_id = params->weight_tensor_id;
    mars_runtime_tensor_t *weight = NULL;
    for (uint32_t i = 0; i < model->header.num_tensors; i++) {
        if (model->tensors[i].desc.id == w_id) {
            weight = &model->tensors[i];
            break;
        }
    }
    if (!weight || !weight->vaddr) return MARS_ERR_INVALID_TENSOR;

    /* Get bias tensor (optional) */
    uint32_t b_id = params->bias_tensor_id;
    mars_runtime_tensor_t *bias = NULL;
    if (b_id != 0xFFFFFFFF) {
        for (uint32_t i = 0; i < model->header.num_tensors; i++) {
            if (model->tensors[i].desc.id == b_id) {
                bias = &model->tensors[i];
                break;
            }
        }
    }

    /* Extract dimensions - NHWC format */
    int in_h = input->desc.shape[1];
    int in_w = input->desc.shape[2];
    int in_c = input->desc.shape[3];
    int out_h = output->desc.shape[1];
    int out_w = output->desc.shape[2];
    int out_c = output->desc.shape[3];

    /* Calculate padding for SAME mode */
    int pad_top = 0, pad_left = 0;
    if (params->padding == MARS_PAD_SAME) {
        int pad_h = (out_h - 1) * params->stride_h + params->kernel_h - in_h;
        int pad_w = (out_w - 1) * params->stride_w + params->kernel_w - in_w;
        pad_top = pad_h / 2;
        pad_left = pad_w / 2;
    }

    printf("  Conv2D: %dx%dx%d -> %dx%dx%d (k=%dx%d, s=%d)\n",
           in_h, in_w, in_c, out_h, out_w, out_c,
           params->kernel_h, params->kernel_w, params->stride_h);

    /* Run software convolution */
    conv2d_int8_sw(
        (int8_t *)input->vaddr, in_h, in_w, in_c,
        (int8_t *)weight->vaddr, out_c, params->kernel_h, params->kernel_w,
        bias ? (int32_t *)bias->vaddr : NULL,
        (int8_t *)output->vaddr, out_h, out_w,
        params->stride_h, params->stride_w,
        pad_top, pad_left,
        input->desc.scale, weight->desc.scale, output->desc.scale
    );

    /* Apply activation if specified */
    if (params->activation == MARS_ACT_RELU) {
        int8_t *out = (int8_t *)output->vaddr;
        int total = out_h * out_w * out_c;
        for (int i = 0; i < total; i++) {
            if (out[i] < 0) out[i] = 0;
        }
    }

    return MARS_OK;
}

/* Layer execution dispatcher */
static mars_error_t execute_layer(mars_model_t *model, mars_runtime_layer_t *layer) {
    const mars_layer_t *desc = &layer->desc;

    switch (desc->type) {
        case MARS_LAYER_CONV2D:
            printf("  Executing Conv2D layer %u\n", desc->id);
            return execute_conv2d(model, layer);

        case MARS_LAYER_DEPTHWISE_CONV2D:
            printf("  Executing DepthwiseConv2D layer %u\n", desc->id);
            break;

        case MARS_LAYER_MAXPOOL:
            printf("  Executing MaxPool layer %u\n", desc->id);
            break;

        case MARS_LAYER_AVGPOOL:
            printf("  Executing AvgPool layer %u\n", desc->id);
            break;

        case MARS_LAYER_RELU:
        case MARS_LAYER_RELU6:
        case MARS_LAYER_SILU:
        case MARS_LAYER_SIGMOID:
        case MARS_LAYER_LEAKY_RELU:
            printf("  Executing Activation layer %u (type=%d)\n", desc->id, desc->type);
            break;

        case MARS_LAYER_CONCAT:
            printf("  Executing Concat layer %u\n", desc->id);
            break;

        case MARS_LAYER_ADD:
            printf("  Executing Add layer %u\n", desc->id);
            break;

        case MARS_LAYER_UPSAMPLE:
            printf("  Executing Upsample layer %u\n", desc->id);
            break;

        case MARS_LAYER_RESHAPE:
            printf("  Executing Reshape layer %u\n", desc->id);
            break;

        default:
            fprintf(stderr, "Mars: Unknown layer type %d\n", desc->type);
            return MARS_ERR_INVALID_LAYER;
    }

    return MARS_OK;
}

