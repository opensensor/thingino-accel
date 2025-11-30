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
#include <math.h>

#include "mars.h"
#include "mars_runtime.h"
#include "nna.h"
#include "device_internal.h"
#include "mxu_ops.h"

/* External MXU-accelerated convolutions */
extern void conv2d_int8_mxu(
    const int8_t *input, int in_h, int in_w, int in_c,
    const int8_t *weight, int out_c, int kh, int kw,
    const int32_t *bias,
    int8_t *output, int out_h, int out_w,
    int stride_h, int stride_w,
    int pad_top, int pad_left,
    float in_scale, float w_scale, float out_scale);

extern void conv2d_float32_mxu(
    const float *input, int in_h, int in_w, int in_c,
    const float *weight, int out_c, int kh, int kw,
    const float *bias,
    float *output, int out_h, int out_w,
    int stride_h, int stride_w,
    int pad_top, int pad_left,
    float *scratch);

/* Set to 1 to use MXU acceleration, 0 for software fallback */
#ifndef USE_MXU
#define USE_MXU 1
#endif

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
/* Calculate tensor size based on format */
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

    /* Handle NNA native formats */
    if (t->format == MARS_FORMAT_NDHWC32 && t->ndims >= 4) {
        /* NDHWC32: [N, D_C32, H, W, 32] where D_C32 = ceil(C/32) */
        int n = t->shape[0];
        int c = t->shape[1];  /* original channels */
        int h = t->shape[2];
        int w = t->shape[3];
        int d_c32 = (c + 31) / 32;
        return n * d_c32 * h * w * 32 * elem_size;
    }

    if (t->format == MARS_FORMAT_NMHWSOIB2 && t->ndims >= 4) {
        /* NMHWSOIB2: packed 1024-byte blocks */
        int out_ch = t->shape[0];
        int in_ch = t->shape[1];
        int kh = t->ndims > 2 ? t->shape[2] : 1;
        int kw = t->ndims > 3 ? t->shape[3] : 1;
        int n_ofp = (out_ch + 31) / 32;
        int m_ifp = (in_ch + 31) / 32;
        return n_ofp * m_ifp * kh * kw * 1024;  /* 32*32 = 1024 */
    }

    /* Standard formats */
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

    /*
     * Memory allocation strategy: Double-buffer scheme
     * Instead of allocating all tensors, we use 2 working buffers and ping-pong.
     * This works because most ops are: input -> output, no persistent tensors needed.
     *
     * Exception: Skip connections (concat, add) need inputs kept around.
     * For now, use simple double-buffer; will need enhancement for skip connections.
     */
    size_t ddr_remaining = model->ddr_size - model->weights_size;
    uint8_t *ddr_ptr = (uint8_t *)model->ddr_base + model->weights_size;

    /* Find max tensor size needed for working buffers */
    size_t max_tensor_size = 0;
    for (uint32_t i = 0; i < header.num_tensors; i++) {
        mars_runtime_tensor_t *rt = &model->tensors[i];
        if (rt->desc.data_size == 0) {  /* Runtime tensor, not weight */
            size_t sz = ALIGN_UP(tensor_byte_size(&rt->desc), 64);
            if (sz > max_tensor_size) max_tensor_size = sz;
        }
    }

    /* Allocate two working buffers (ping-pong) plus one extra for skip connections */
    size_t num_buffers = 3;
    size_t buffer_size = max_tensor_size;
    size_t total_buffer = buffer_size * num_buffers;

    fprintf(stderr, "Mars: Max tensor size: %zu, buffers: %zu x %zu = %zu bytes\n",
            max_tensor_size, num_buffers, buffer_size, total_buffer);

    if (total_buffer > ddr_remaining) {
        /* Fall back to smaller buffers or fail gracefully */
        fprintf(stderr, "Mars: Need %zu bytes for buffers, have %zu\n",
                total_buffer, ddr_remaining);
        /* Try with just 2 buffers */
        num_buffers = 2;
        total_buffer = buffer_size * num_buffers;
        if (total_buffer > ddr_remaining) {
            fprintf(stderr, "Mars: Out of DDR memory (need %zu, have %zu)\n",
                    total_buffer, ddr_remaining);
            free(model->layers);
            free(model->tensors);
            free(model);
            return MARS_ERR_ALLOC_FAILED;
        }
    }

    /* Working buffer pointers - store in model for later use */
    uint8_t *work_buffers[3];
    for (size_t b = 0; b < num_buffers; b++) {
        work_buffers[b] = ddr_ptr + b * buffer_size;
    }

    /* Assign tensors to working buffers in round-robin */
    uint32_t buf_idx = 0;
    for (uint32_t i = 0; i < header.num_tensors; i++) {
        mars_runtime_tensor_t *rt = &model->tensors[i];

        /* Check if tensor has weight data (data_size > 0 means it's stored in weights section) */
        if (rt->desc.data_size > 0) {
            rt->vaddr = (uint8_t *)model->ddr_base + rt->desc.data_offset;
            rt->paddr = (uint8_t *)model->ddr_paddr + rt->desc.data_offset;
            rt->alloc_size = rt->desc.data_size;
            continue;
        }

        /* Assign to working buffer (round-robin) */
        rt->vaddr = (void *)work_buffers[buf_idx % num_buffers];
        rt->paddr = (void *)((uint8_t *)model->ddr_paddr +
                             ((uint8_t *)rt->vaddr - (uint8_t *)model->ddr_base));
        rt->alloc_size = buffer_size;
        buf_idx++;
    }

    fprintf(stderr, "Mars: Allocated %u tensors using %zu working buffers\n",
            header.num_tensors, num_buffers);

#if USE_MXU
    /* Initialize MXU for compute operations */
    if (!mxu_is_initialized()) {
        mxu_init(model->ddr_base);
        fprintf(stderr, "Mars: MXU initialized for SIMD acceleration\n");
    }
#endif

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

    /* Check if float32 model */
    int is_float = (input->desc.dtype == MARS_DTYPE_FLOAT32);

    printf("  Conv2D: %dx%dx%d -> %dx%dx%d (k=%dx%d, s=%d) [%s%s]\n",
           in_h, in_w, in_c, out_h, out_w, out_c,
           params->kernel_h, params->kernel_w, params->stride_h,
           is_float ? "F32-" : "INT8-",
           USE_MXU ? "MXU" : "SW");

#if USE_MXU
    if (is_float) {
        /* Float32 MXU-accelerated convolution */
        /* Use end of DDR buffer as scratch space for VPR stores */
        float *scratch = (float *)((char *)model->ddr_base + model->ddr_size - 256);
        conv2d_float32_mxu(
            (float *)input->vaddr, in_h, in_w, in_c,
            (float *)weight->vaddr, out_c, params->kernel_h, params->kernel_w,
            bias ? (float *)bias->vaddr : NULL,
            (float *)output->vaddr, out_h, out_w,
            params->stride_h, params->stride_w,
            pad_top, pad_left,
            scratch
        );
    } else {
        /* INT8 MXU-accelerated convolution */
        conv2d_int8_mxu(
            (int8_t *)input->vaddr, in_h, in_w, in_c,
            (int8_t *)weight->vaddr, out_c, params->kernel_h, params->kernel_w,
            bias ? (int32_t *)bias->vaddr : NULL,
            (int8_t *)output->vaddr, out_h, out_w,
            params->stride_h, params->stride_w,
            pad_top, pad_left,
            input->desc.scale, weight->desc.scale, output->desc.scale
        );
    }
#else
    if (is_float) {
        /* Float32 software convolution */
        conv2d_float32_mxu(
            (float *)input->vaddr, in_h, in_w, in_c,
            (float *)weight->vaddr, out_c, params->kernel_h, params->kernel_w,
            bias ? (float *)bias->vaddr : NULL,
            (float *)output->vaddr, out_h, out_w,
            params->stride_h, params->stride_w,
            pad_top, pad_left,
            NULL
        );
    } else {
        /* INT8 software convolution */
        conv2d_int8_sw(
            (int8_t *)input->vaddr, in_h, in_w, in_c,
            (int8_t *)weight->vaddr, out_c, params->kernel_h, params->kernel_w,
            bias ? (int32_t *)bias->vaddr : NULL,
            (int8_t *)output->vaddr, out_h, out_w,
            params->stride_h, params->stride_w,
            pad_top, pad_left,
            input->desc.scale, weight->desc.scale, output->desc.scale
        );
    }
#endif

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

/* Helper to get tensor by ID */
static mars_runtime_tensor_t* get_tensor_by_id(mars_model_t *model, uint32_t id) {
    if (id == 0xFFFFFFFF) return NULL;
    for (uint32_t i = 0; i < model->header.num_tensors; i++) {
        if (model->tensors[i].desc.id == id) {
            return &model->tensors[i];
        }
    }
    return NULL;
}

/* Execute Sigmoid: out = 1 / (1 + exp(-x)) */
static mars_error_t execute_sigmoid(mars_model_t *model, mars_runtime_layer_t *layer) {
    const mars_layer_t *desc = &layer->desc;

    mars_runtime_tensor_t *input = get_tensor_by_id(model, desc->input_tensor_ids[0]);
    mars_runtime_tensor_t *output = get_tensor_by_id(model, desc->output_tensor_ids[0]);
    if (!input || !output || !input->vaddr || !output->vaddr) {
        return MARS_ERR_INVALID_TENSOR;
    }

    /* Calculate number of elements */
    size_t numel = 1;
    for (uint32_t i = 0; i < input->desc.ndims; i++) {
        numel *= input->desc.shape[i];
    }

    /* Check if float32 model */
    int is_float = (input->desc.dtype == MARS_DTYPE_FLOAT32);

    if (is_float) {
        /* Direct float32 path - no quantization overhead */
        float *in = (float *)input->vaddr;
        float *out = (float *)output->vaddr;
        for (size_t i = 0; i < numel; i++) {
            out[i] = 1.0f / (1.0f + expf(-in[i]));
        }
        return MARS_OK;
    }

    /* INT8 path with quantization */
    int8_t *in = (int8_t *)input->vaddr;
    int8_t *out = (int8_t *)output->vaddr;
    float in_scale = input->desc.scale;
    float out_scale = output->desc.scale > 0 ? output->desc.scale : 1.0f;

    for (size_t i = 0; i < numel; i++) {
        /* Dequantize */
        float x = in[i] * in_scale;
        /* Sigmoid */
        float y = 1.0f / (1.0f + expf(-x));
        /* Requantize */
        int32_t q = (int32_t)(y / out_scale + 0.5f);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        out[i] = (int8_t)q;
    }

    return MARS_OK;
}

/* Execute element-wise Mul: out = a * b */
static mars_error_t execute_mul(mars_model_t *model, mars_runtime_layer_t *layer) {
    const mars_layer_t *desc = &layer->desc;

    mars_runtime_tensor_t *input_a = get_tensor_by_id(model, desc->input_tensor_ids[0]);
    mars_runtime_tensor_t *input_b = get_tensor_by_id(model, desc->input_tensor_ids[1]);
    mars_runtime_tensor_t *output = get_tensor_by_id(model, desc->output_tensor_ids[0]);

    if (!input_a || !input_b || !output) {
        return MARS_ERR_INVALID_TENSOR;
    }
    if (!input_a->vaddr || !input_b->vaddr || !output->vaddr) {
        return MARS_ERR_INVALID_TENSOR;
    }

    size_t numel = 1;
    for (uint32_t i = 0; i < input_a->desc.ndims; i++) {
        numel *= input_a->desc.shape[i];
    }

    /* Check if float32 model */
    int is_float = (input_a->desc.dtype == MARS_DTYPE_FLOAT32);

#if USE_MXU && defined(__mips__)
    if (is_float && mxu_is_initialized()) {
        /* Direct float32 MXU path - no quantization overhead */
        float *a = (float *)input_a->vaddr;
        float *b = (float *)input_b->vaddr;
        float *out = (float *)output->vaddr;
        mxu_mul_f32(out, a, b, numel);
        return MARS_OK;
    }
#endif

    if (is_float) {
        /* Float32 scalar fallback */
        float *a = (float *)input_a->vaddr;
        float *b = (float *)input_b->vaddr;
        float *out = (float *)output->vaddr;
        for (size_t i = 0; i < numel; i++) {
            out[i] = a[i] * b[i];
        }
        return MARS_OK;
    }

    /* INT8 path with quantization */
    int8_t *a = (int8_t *)input_a->vaddr;
    int8_t *b = (int8_t *)input_b->vaddr;
    int8_t *out = (int8_t *)output->vaddr;
    float scale_a = input_a->desc.scale;
    float scale_b = input_b->desc.scale;
    float scale_out = output->desc.scale > 0 ? output->desc.scale : 1.0f;
    float inv_scale_out = 1.0f / scale_out;

    for (size_t i = 0; i < numel; i++) {
        float va = a[i] * scale_a;
        float vb = b[i] * scale_b;
        float y = va * vb;
        int32_t q = (int32_t)(y * inv_scale_out + 0.5f);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        out[i] = (int8_t)q;
    }

    return MARS_OK;
}

/* Execute element-wise Add: out = a + b */
static mars_error_t execute_add(mars_model_t *model, mars_runtime_layer_t *layer) {
    const mars_layer_t *desc = &layer->desc;

    mars_runtime_tensor_t *input_a = get_tensor_by_id(model, desc->input_tensor_ids[0]);
    mars_runtime_tensor_t *input_b = get_tensor_by_id(model, desc->input_tensor_ids[1]);
    mars_runtime_tensor_t *output = get_tensor_by_id(model, desc->output_tensor_ids[0]);

    if (!input_a || !input_b || !output) {
        return MARS_ERR_INVALID_TENSOR;
    }
    if (!input_a->vaddr || !input_b->vaddr || !output->vaddr) {
        return MARS_ERR_INVALID_TENSOR;
    }

    size_t numel = 1;
    for (uint32_t i = 0; i < input_a->desc.ndims; i++) {
        numel *= input_a->desc.shape[i];
    }

    /* Check if float32 model */
    int is_float = (input_a->desc.dtype == MARS_DTYPE_FLOAT32);

#if USE_MXU && defined(__mips__)
    if (is_float && mxu_is_initialized()) {
        /* Direct float32 MXU path - no quantization overhead */
        float *a = (float *)input_a->vaddr;
        float *b = (float *)input_b->vaddr;
        float *out = (float *)output->vaddr;
        mxu_add_f32(out, a, b, numel);
        return MARS_OK;
    }
#endif

    if (is_float) {
        /* Float32 scalar fallback */
        float *a = (float *)input_a->vaddr;
        float *b = (float *)input_b->vaddr;
        float *out = (float *)output->vaddr;
        for (size_t i = 0; i < numel; i++) {
            out[i] = a[i] + b[i];
        }
        return MARS_OK;
    }

    /* INT8 path with quantization */
    int8_t *a = (int8_t *)input_a->vaddr;
    int8_t *b = (int8_t *)input_b->vaddr;
    int8_t *out = (int8_t *)output->vaddr;
    float scale_a = input_a->desc.scale;
    float scale_b = input_b->desc.scale;
    float scale_out = output->desc.scale > 0 ? output->desc.scale : 1.0f;
    float inv_scale_out = 1.0f / scale_out;

    for (size_t i = 0; i < numel; i++) {
        float va = a[i] * scale_a;
        float vb = b[i] * scale_b;
        float y = va + vb;
        int32_t q = (int32_t)(y * inv_scale_out + 0.5f);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        out[i] = (int8_t)q;
    }

    return MARS_OK;
}

/* Execute MaxPool */
static mars_error_t execute_maxpool(mars_model_t *model, mars_runtime_layer_t *layer) {
    const mars_layer_t *desc = &layer->desc;
    const mars_pool_params_t *params = &desc->params.pool;

    mars_runtime_tensor_t *input = get_tensor_by_id(model, desc->input_tensor_ids[0]);
    mars_runtime_tensor_t *output = get_tensor_by_id(model, desc->output_tensor_ids[0]);

    if (!input || !output || !input->vaddr || !output->vaddr) {
        return MARS_ERR_INVALID_TENSOR;
    }

    /* NHWC format */
    int in_h = input->desc.shape[1];
    int in_w = input->desc.shape[2];
    int channels = input->desc.shape[3];
    int out_h = output->desc.shape[1];
    int out_w = output->desc.shape[2];

    int kh = params->kernel_h;
    int kw = params->kernel_w;
    int sh = params->stride_h;
    int sw = params->stride_w;

    int8_t *in = (int8_t *)input->vaddr;
    int8_t *out = (int8_t *)output->vaddr;

    for (int c = 0; c < channels; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                int8_t max_val = -128;

                for (int khi = 0; khi < kh; khi++) {
                    for (int kwi = 0; kwi < kw; kwi++) {
                        int ih = oh * sh + khi;
                        int iw = ow * sw + kwi;

                        if (ih < in_h && iw < in_w) {
                            int in_idx = ih * in_w * channels + iw * channels + c;
                            if (in[in_idx] > max_val) {
                                max_val = in[in_idx];
                            }
                        }
                    }
                }

                int out_idx = oh * out_w * channels + ow * channels + c;
                out[out_idx] = max_val;
            }
        }
    }

    return MARS_OK;
}

/* Execute Concat along channel axis */
static mars_error_t execute_concat(mars_model_t *model, mars_runtime_layer_t *layer) {
    const mars_layer_t *desc = &layer->desc;

    mars_runtime_tensor_t *output = get_tensor_by_id(model, desc->output_tensor_ids[0]);
    if (!output || !output->vaddr) {
        return MARS_ERR_INVALID_TENSOR;
    }

    /* Assuming NHWC and concat along channel axis (axis=3 or -1) */
    int out_h = output->desc.shape[1];
    int out_w = output->desc.shape[2];
    int8_t *out = (int8_t *)output->vaddr;

    int channel_offset = 0;

    for (uint32_t n = 0; n < desc->num_inputs; n++) {
        mars_runtime_tensor_t *input = get_tensor_by_id(model, desc->input_tensor_ids[n]);
        if (!input || !input->vaddr) continue;

        int in_c = input->desc.shape[3];
        int8_t *in = (int8_t *)input->vaddr;
        int out_c = output->desc.shape[3];

        /* Copy each channel slice */
        for (int h = 0; h < out_h; h++) {
            for (int w = 0; w < out_w; w++) {
                for (int c = 0; c < in_c; c++) {
                    int in_idx = h * out_w * in_c + w * in_c + c;
                    int out_idx = h * out_w * out_c + w * out_c + (channel_offset + c);
                    out[out_idx] = in[in_idx];
                }
            }
        }
        channel_offset += in_c;
    }

    return MARS_OK;
}

/* Execute Upsample (nearest neighbor) */
static mars_error_t execute_upsample(mars_model_t *model, mars_runtime_layer_t *layer) {
    const mars_layer_t *desc = &layer->desc;
    const mars_upsample_params_t *params = &desc->params.upsample;

    mars_runtime_tensor_t *input = get_tensor_by_id(model, desc->input_tensor_ids[0]);
    mars_runtime_tensor_t *output = get_tensor_by_id(model, desc->output_tensor_ids[0]);

    if (!input || !output || !input->vaddr || !output->vaddr) {
        return MARS_ERR_INVALID_TENSOR;
    }

    int in_h = input->desc.shape[1];
    int in_w = input->desc.shape[2];
    int channels = input->desc.shape[3];
    int out_h = output->desc.shape[1];
    int out_w = output->desc.shape[2];

    int scale_h = params->scale_h > 0 ? params->scale_h : (out_h / in_h);
    int scale_w = params->scale_w > 0 ? params->scale_w : (out_w / in_w);

    int8_t *in = (int8_t *)input->vaddr;
    int8_t *out = (int8_t *)output->vaddr;

    /* Nearest neighbor upsampling */
    for (int oh = 0; oh < out_h; oh++) {
        int ih = oh / scale_h;
        if (ih >= in_h) ih = in_h - 1;

        for (int ow = 0; ow < out_w; ow++) {
            int iw = ow / scale_w;
            if (iw >= in_w) iw = in_w - 1;

            for (int c = 0; c < channels; c++) {
                int in_idx = ih * in_w * channels + iw * channels + c;
                int out_idx = oh * out_w * channels + ow * channels + c;
                out[out_idx] = in[in_idx];
            }
        }
    }

    return MARS_OK;
}

/* Execute ReLU/LeakyReLU: out = max(0, x) or out = x if x > 0, else alpha * x */
static mars_error_t execute_relu(mars_model_t *model, mars_runtime_layer_t *layer) {
    const mars_layer_t *desc = &layer->desc;

    mars_runtime_tensor_t *input = get_tensor_by_id(model, desc->input_tensor_ids[0]);
    mars_runtime_tensor_t *output = get_tensor_by_id(model, desc->output_tensor_ids[0]);

    if (!input || !output || !input->vaddr || !output->vaddr) {
        return MARS_ERR_INVALID_TENSOR;
    }

    size_t numel = 1;
    for (uint32_t i = 0; i < input->desc.ndims; i++) {
        numel *= input->desc.shape[i];
    }

    /* LeakyReLU uses alpha=0.01 by default */
    int is_leaky = (desc->type == MARS_LAYER_LEAKY_RELU);
    float alpha = is_leaky ? 0.01f : 0.0f;

    if (input->desc.dtype == MARS_DTYPE_FLOAT32) {
        const float *in = (const float *)input->vaddr;
        float *out = (float *)output->vaddr;
        for (size_t i = 0; i < numel; i++) {
            out[i] = in[i] > 0.0f ? in[i] : in[i] * alpha;
        }
    } else {
        const int8_t *in = (const int8_t *)input->vaddr;
        int8_t *out = (int8_t *)output->vaddr;
        for (size_t i = 0; i < numel; i++) {
            if (in[i] > 0) {
                out[i] = in[i];
            } else if (is_leaky) {
                /* Apply alpha with rounding */
                int32_t v = (int32_t)(in[i] * alpha);
                out[i] = (int8_t)(v < -128 ? -128 : v);
            } else {
                out[i] = 0;
            }
        }
    }

    return MARS_OK;
}

/* Execute BatchNorm: y = x * scale + bias (fused BN parameters) */
static mars_error_t execute_batchnorm(mars_model_t *model, mars_runtime_layer_t *layer) {
    const mars_layer_t *desc = &layer->desc;

    mars_runtime_tensor_t *input = get_tensor_by_id(model, desc->input_tensor_ids[0]);
    mars_runtime_tensor_t *scale = get_tensor_by_id(model, desc->input_tensor_ids[1]);
    mars_runtime_tensor_t *bias = get_tensor_by_id(model, desc->input_tensor_ids[2]);
    mars_runtime_tensor_t *output = get_tensor_by_id(model, desc->output_tensor_ids[0]);

    if (!input || !output || !input->vaddr || !output->vaddr) {
        return MARS_ERR_INVALID_TENSOR;
    }

    /* Get dimensions - assuming NCHW format */
    int n = input->desc.shape[0] > 0 ? input->desc.shape[0] : 1;
    int c = input->desc.shape[1] > 0 ? input->desc.shape[1] : 1;
    int h = input->desc.shape[2] > 0 ? input->desc.shape[2] : 1;
    int w = input->desc.shape[3] > 0 ? input->desc.shape[3] : 1;

    /* Get fused scale and bias from weight tensors */
    const float *s = scale && scale->vaddr ? (const float *)scale->vaddr : NULL;
    const float *b = bias && bias->vaddr ? (const float *)bias->vaddr : NULL;

    /* Float32 path */
    if (input->desc.dtype == MARS_DTYPE_FLOAT32) {
        const float *in = (const float *)input->vaddr;
        float *out = (float *)output->vaddr;

        for (int ni = 0; ni < n; ni++) {
            for (int ci = 0; ci < c; ci++) {
                float sc = s ? s[ci] : 1.0f;
                float bi = b ? b[ci] : 0.0f;
                for (int hi = 0; hi < h; hi++) {
                    for (int wi = 0; wi < w; wi++) {
                        int idx = ((ni * c + ci) * h + hi) * w + wi;
                        out[idx] = in[idx] * sc + bi;
                    }
                }
            }
        }
    } else {
        /* INT8 path with quantization */
        const int8_t *in = (const int8_t *)input->vaddr;
        int8_t *out = (int8_t *)output->vaddr;
        float in_scale = input->desc.scale > 0 ? input->desc.scale : 1.0f;
        float out_scale = output->desc.scale > 0 ? output->desc.scale : 1.0f;

        for (int ni = 0; ni < n; ni++) {
            for (int ci = 0; ci < c; ci++) {
                float sc = s ? s[ci] : 1.0f;
                float bi = b ? b[ci] : 0.0f;
                for (int hi = 0; hi < h; hi++) {
                    for (int wi = 0; wi < w; wi++) {
                        int idx = ((ni * c + ci) * h + hi) * w + wi;
                        float x = in[idx] * in_scale;
                        float y = x * sc + bi;
                        int32_t q = (int32_t)(y / out_scale + 0.5f);
                        if (q > 127) q = 127;
                        if (q < -128) q = -128;
                        out[idx] = (int8_t)q;
                    }
                }
            }
        }
    }

    return MARS_OK;
}

/* Layer execution dispatcher */
static mars_error_t execute_layer(mars_model_t *model, mars_runtime_layer_t *layer) {
    const mars_layer_t *desc = &layer->desc;

    switch (desc->type) {
        case MARS_LAYER_CONV2D:
            return execute_conv2d(model, layer);

        case MARS_LAYER_DEPTHWISE_CONV2D:
            /* TODO: implement depthwise conv */
            return MARS_OK;

        case MARS_LAYER_MAXPOOL:
            return execute_maxpool(model, layer);

        case MARS_LAYER_AVGPOOL:
            /* TODO: implement avgpool */
            return MARS_OK;

        case MARS_LAYER_RELU:
        case MARS_LAYER_RELU6:
        case MARS_LAYER_LEAKY_RELU:
            return execute_relu(model, layer);

        case MARS_LAYER_SILU:
            /* SiLU is implemented as Sigmoid + Mul in ONNX */
            return MARS_OK;

        case MARS_LAYER_SIGMOID:
            return execute_sigmoid(model, layer);

        case MARS_LAYER_CONCAT:
            return execute_concat(model, layer);

        case MARS_LAYER_ADD:
            return execute_add(model, layer);

        case MARS_LAYER_MUL:
            return execute_mul(model, layer);

        case MARS_LAYER_UPSAMPLE:
            return execute_upsample(model, layer);

        case MARS_LAYER_RESHAPE:
            /* Reshape is a no-op for data, just reinterpret shape */
            return MARS_OK;

        case MARS_LAYER_TRANSPOSE:
            /* TODO: implement transpose */
            return MARS_OK;

        case MARS_LAYER_SOFTMAX:
            /* TODO: implement softmax */
            return MARS_OK;

        case MARS_LAYER_BATCHNORM:
            return execute_batchnorm(model, layer);

        default:
            fprintf(stderr, "Mars: Unknown layer type %d\n", desc->type);
            return MARS_ERR_INVALID_LAYER;
    }

    return MARS_OK;
}

