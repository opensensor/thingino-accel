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
#include "nna_memory.h"
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

extern void conv2d_int8_nhwc_mxu(
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
    fprintf(stderr, "Mars: mars_load_memory called, data=%p size=%zu\n", data, size);
    fflush(stderr);

    if (!data || !out_model || size < sizeof(mars_header_t)) {
        return MARS_ERR_INVALID_FILE;
    }

    const uint8_t *ptr = (const uint8_t *)data;

    /* Parse header */
    fprintf(stderr, "Mars: Parsing header...\n"); fflush(stderr);
    mars_header_t header;
    memcpy(&header, ptr, sizeof(header));
    ptr += sizeof(header);

    /* Validate magic */
    fprintf(stderr, "Mars: Magic=0x%08x layers=%u tensors=%u\n",
            header.magic, header.num_layers, header.num_tensors);
    fflush(stderr);

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
    fprintf(stderr, "Mars: Allocating model context...\n"); fflush(stderr);
    mars_model_t *model = (mars_model_t *)calloc(1, sizeof(mars_model_t));
    if (!model) {
        return MARS_ERR_ALLOC_FAILED;
    }

    memcpy(&model->header, &header, sizeof(header));

    /* Allocate tensors array */
    fprintf(stderr, "Mars: Allocating %u tensors...\n", header.num_tensors); fflush(stderr);
    model->tensors = (mars_runtime_tensor_t *)calloc(header.num_tensors,
                                                       sizeof(mars_runtime_tensor_t));
    if (!model->tensors) {
        free(model);
        return MARS_ERR_ALLOC_FAILED;
    }

    /* Read tensor descriptors and initialize liveness tracking */
    for (uint32_t i = 0; i < header.num_tensors; i++) {
        memcpy(&model->tensors[i].desc, ptr, sizeof(mars_tensor_t));
        ptr += sizeof(mars_tensor_t);
        /* Initialize liveness: -1 means not yet determined */
        model->tensors[i].produced_at = -1;
        model->tensors[i].last_used_at = -1;
        model->tensors[i].buffer_idx = -1;
    }
    fprintf(stderr, "Mars: Tensors loaded\n"); fflush(stderr);

    /* Allocate layers array */
    fprintf(stderr, "Mars: Allocating %u layers...\n", header.num_layers); fflush(stderr);
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
    fprintf(stderr, "Mars: Layers loaded\n"); fflush(stderr);

    /*
     * Build tensor liveness table:
     * - produced_at: layer index where tensor is produced (output of layer)
     * - last_used_at: layer index where tensor is last consumed (input to layer)
     * This allows dynamic buffer assignment during execution.
     */
    for (uint32_t layer_idx = 0; layer_idx < header.num_layers; layer_idx++) {
        mars_layer_t *layer = &model->layers[layer_idx].desc;

        /* Mark outputs as produced at this layer */
        for (uint32_t o = 0; o < layer->num_outputs; o++) {
            uint32_t tid = layer->output_tensor_ids[o];
            if (tid < header.num_tensors) {
                model->tensors[tid].produced_at = (int32_t)layer_idx;
            }
        }

        /* Update last_used_at for all inputs */
        for (uint32_t i = 0; i < layer->num_inputs; i++) {
            uint32_t tid = layer->input_tensor_ids[i];
            if (tid < header.num_tensors) {
                model->tensors[tid].last_used_at = (int32_t)layer_idx;
            }
        }
    }
    fprintf(stderr, "Mars: Tensor liveness computed\n"); fflush(stderr);

    /* Get NNA resources */
    fprintf(stderr, "Mars: Getting NNA resources...\n"); fflush(stderr);
    /* Get DDR for weights (single chunk) */
    model->ddr_base = nna_device_get_ddr();
    fprintf(stderr, "Mars: ddr_base=%p (for weights)\n", model->ddr_base); fflush(stderr);
    model->ddr_paddr = (void *)(uintptr_t)nna_device_get_ddr_pbase();
    model->ddr_size = nna_device_get_ddr_size();
    fprintf(stderr, "Mars: ddr_size=%zu bytes (%zu MB)\n", model->ddr_size, model->ddr_size / (1024*1024)); fflush(stderr);
    model->oram_base = nna_device_get_oram();
    model->oram_paddr = model->oram_base;  /* TODO: get actual paddr */
    model->oram_size = 384 * 1024;  /* 384KB */

    /* Load weights into DDR */
    fprintf(stderr, "Mars: Loading weights (offset=%llu, size=%llu)\n",
            (unsigned long long)header.weights_offset, (unsigned long long)header.weights_size);
    fflush(stderr);

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

        fprintf(stderr, "Mars: Copying %zu bytes to DDR at %p\n",
                model->weights_size, model->ddr_base);
        memcpy(model->ddr_base, weights_src, model->weights_size);
        model->weights = model->ddr_base;
        fprintf(stderr, "Mars: Weights loaded successfully\n");
    }

    /*
     * DYNAMIC TENSOR ALLOCATION STRATEGY
     *
     * Each tensor is allocated via nna_malloc_phys() from the nmem pool.
     * This allows us to use all available nmem (~29MB) instead of being
     * limited to pre-allocated chunks.
     *
     * We allocate:
     * 1. I/O tensors (input/output) - always need dedicated buffers
     * 2. Long-lived tensors (skip connections) - need to persist across layers
     * 3. Working buffers for short-lived tensors - can be reused
     */

    /* Find max tensor size for working buffers */
    size_t max_intermediate_size = 0;
    const int32_t LONG_LIVED_THRESHOLD = 5;

    for (uint32_t i = 0; i < header.num_tensors; i++) {
        mars_runtime_tensor_t *rt = &model->tensors[i];
        if (rt->desc.data_size > 0) continue;  /* Weight tensor */
        if (rt->produced_at < 0) continue;  /* Input tensor */
        if (rt->last_used_at < 0) continue;  /* Output tensor */
        size_t sz = ALIGN_UP(tensor_byte_size(&rt->desc), 64);
        if (sz > max_intermediate_size) max_intermediate_size = sz;
    }

    fprintf(stderr, "Mars: Max intermediate tensor: %zu bytes\n", max_intermediate_size);

    /*
     * DYNAMIC ALLOCATION via nna_malloc_phys()
     * Each tensor gets its own allocation from the nmem pool.
     * This gives us access to the full ~29MB nmem instead of limited chunks.
     */

    /* Allocate input tensors */
    for (uint32_t n = 0; n < header.num_inputs; n++) {
        uint32_t tid = header.input_tensor_ids[n];
        if (tid >= header.num_tensors) continue;
        mars_runtime_tensor_t *rt = &model->tensors[tid];
        if (rt->desc.data_size > 0) continue;

        size_t sz = ALIGN_UP(tensor_byte_size(&rt->desc), 64);
        void *paddr = NULL;
        rt->vaddr = nna_malloc_phys(sz, &paddr);
        if (rt->vaddr == NULL) {
            fprintf(stderr, "Mars: Failed to allocate input tensor %u (%zu bytes)\n", tid, sz);
            free(model->layers);
            free(model->tensors);
            free(model);
            return MARS_ERR_ALLOC_FAILED;
        }
        rt->paddr = paddr;
        rt->alloc_size = sz;
        rt->buffer_idx = -1;  /* Dedicated buffer */
        fprintf(stderr, "Mars: Input tensor %u: %zu bytes at vaddr=%p paddr=%p\n",
                tid, sz, rt->vaddr, rt->paddr);
    }

    /* Allocate output tensors */
    for (uint32_t n = 0; n < header.num_outputs; n++) {
        uint32_t tid = header.output_tensor_ids[n];
        if (tid >= header.num_tensors) continue;
        mars_runtime_tensor_t *rt = &model->tensors[tid];
        if (rt->desc.data_size > 0) continue;
        if (rt->vaddr != NULL) continue;

        size_t sz = ALIGN_UP(tensor_byte_size(&rt->desc), 64);
        void *paddr = NULL;
        rt->vaddr = nna_malloc_phys(sz, &paddr);
        if (rt->vaddr == NULL) {
            fprintf(stderr, "Mars: Failed to allocate output tensor %u (%zu bytes)\n", tid, sz);
            /* TODO: cleanup already allocated tensors */
            free(model->layers);
            free(model->tensors);
            free(model);
            return MARS_ERR_ALLOC_FAILED;
        }
        rt->paddr = paddr;
        rt->alloc_size = sz;
        rt->buffer_idx = -1;
        fprintf(stderr, "Mars: Output tensor %u: %zu bytes at vaddr=%p paddr=%p\n",
                tid, sz, rt->vaddr, rt->paddr);
    }

    /* Allocate long-lived intermediate tensors (skip connections) */
    uint32_t skip_alloc_count = 0;
    for (uint32_t i = 0; i < header.num_tensors; i++) {
        mars_runtime_tensor_t *rt = &model->tensors[i];
        if (rt->vaddr != NULL) continue;  /* Already allocated */
        if (rt->desc.data_size > 0) continue;  /* Weight tensor */
        if (rt->produced_at < 0) continue;  /* Input tensor */
        if (rt->last_used_at < 0) continue;  /* Output tensor */

        int32_t span = rt->last_used_at - rt->produced_at;
        if (span > LONG_LIVED_THRESHOLD) {
            size_t sz = ALIGN_UP(tensor_byte_size(&rt->desc), 64);
            void *paddr = NULL;
            rt->vaddr = nna_malloc_phys(sz, &paddr);
            if (rt->vaddr == NULL) {
                fprintf(stderr, "Mars: Failed to allocate skip tensor %u (%zu bytes)\n", i, sz);
                /* TODO: cleanup already allocated tensors */
                free(model->layers);
                free(model->tensors);
                free(model);
                return MARS_ERR_ALLOC_FAILED;
            }
            rt->paddr = paddr;
            rt->alloc_size = sz;
            rt->buffer_idx = -1;
            skip_alloc_count++;
        }
    }
    fprintf(stderr, "Mars: Allocated %u skip-connection tensors\n", skip_alloc_count);

    /* Allocate working buffers for short-lived tensors */
    size_t buffer_size = ALIGN_UP(max_intermediate_size, 64);
    if (buffer_size < 256 * 1024) buffer_size = 256 * 1024;  /* 256KB minimum */

    model->num_work_buffers = MARS_MAX_WORK_BUFFERS;
    model->work_buffer_size = buffer_size;

    for (uint32_t b = 0; b < model->num_work_buffers; b++) {
        void *paddr = NULL;
        model->work_buffers[b] = nna_malloc_phys(buffer_size, &paddr);
        if (model->work_buffers[b] == NULL) {
            fprintf(stderr, "Mars: Failed to allocate working buffer %u (%zu bytes)\n", b, buffer_size);
            /* TODO: cleanup already allocated */
            free(model->layers);
            free(model->tensors);
            free(model);
            return MARS_ERR_ALLOC_FAILED;
        }
        model->work_buffers_paddr[b] = paddr;
        model->buffer_tensor[b] = -1;  /* Initially free */
    }
    fprintf(stderr, "Mars: Allocated %u working buffers of %zu bytes each\n",
            model->num_work_buffers, buffer_size);

    /* Assign weight tensors - these have fixed memory from the weights section */
    for (uint32_t i = 0; i < header.num_tensors; i++) {
        mars_runtime_tensor_t *rt = &model->tensors[i];
        if (rt->vaddr != NULL) continue;  /* Already allocated */
        if (rt->desc.data_size > 0) {
            /* Weight tensor - assign from weights section */
            rt->vaddr = (uint8_t *)model->ddr_base + rt->desc.data_offset;
            rt->paddr = (uint8_t *)model->ddr_paddr + rt->desc.data_offset;
            rt->alloc_size = rt->desc.data_size;
            rt->buffer_idx = -1;  /* Dedicated (weight) buffer */
        }
        /* Short-lived intermediate tensors will be assigned dynamically during execution */
    }

    fprintf(stderr, "Mars: Using dynamic nna_malloc() for tensor allocation\n");
    fprintf(stderr, "Mars: %u working buffers available for short-lived tensors\n",
            model->num_work_buffers);

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
    fprintf(stderr, "Mars: Opening file %s\n", path); fflush(stderr);

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Mars: Cannot open %s\n", path);
        return MARS_ERR_INVALID_FILE;
    }
    fprintf(stderr, "Mars: File opened, getting size...\n"); fflush(stderr);

    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    fprintf(stderr, "Mars: File size = %ld bytes\n", size); fflush(stderr);

    void *data = malloc(size);
    if (!data) {
        fclose(fp);
        return MARS_ERR_ALLOC_FAILED;
    }
    fprintf(stderr, "Mars: Allocated buffer at %p\n", data); fflush(stderr);

    fprintf(stderr, "Mars: Reading file...\n"); fflush(stderr);
    if (fread(data, 1, size, fp) != (size_t)size) {
        free(data);
        fclose(fp);
        return MARS_ERR_INVALID_FILE;
    }
    fclose(fp);
    fprintf(stderr, "Mars: File read complete\n"); fflush(stderr);

    mars_error_t err = mars_load_memory(data, size, model);
    fprintf(stderr, "Mars: mars_load_memory returned %d\n", err); fflush(stderr);
    free(data);  /* Model copies what it needs */
    return err;
}

void mars_free(mars_model_t *model) {
    if (!model) return;

    /* Free dynamically allocated tensors (I/O, skip connections, runtime) */
    if (model->tensors) {
        for (uint32_t i = 0; i < model->header.num_tensors; i++) {
            mars_runtime_tensor_t *rt = &model->tensors[i];
            /* buffer_idx == -1 means dedicated buffer via nna_malloc
             * buffer_idx == -3 means runtime dynamic allocation
             * Skip weight tensors (they point into DDR weight section) */
            if (rt->vaddr != NULL && rt->desc.data_size == 0 &&
                (rt->buffer_idx == -1 || rt->buffer_idx == -3)) {
                /* Don't free if it's part of the weights DDR block */
                if (model->ddr_base == NULL ||
                    rt->vaddr < model->ddr_base ||
                    (uintptr_t)rt->vaddr >= (uintptr_t)model->ddr_base + model->ddr_size) {
                    nna_free(rt->vaddr);
                    rt->vaddr = NULL;
                }
            }
        }
    }

    /* Free working buffers */
    for (uint32_t b = 0; b < model->num_work_buffers; b++) {
        if (model->work_buffers[b] != NULL) {
            nna_free(model->work_buffers[b]);
            model->work_buffers[b] = NULL;
        }
    }

    if (model->layers) free(model->layers);
    if (model->tensors) free(model->tensors);
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

/*
 * Dynamic buffer assignment: Find a free buffer for a tensor at the given layer
 *
 * A buffer is "free" if:
 * - It has never been used (buffer_tensor[b] == -1), OR
 * - The tensor it holds is "dead" (last_used_at < current_layer_idx)
 *
 * We also must avoid buffers that hold any input to the current layer.
 */
static int find_free_buffer(mars_model_t *model, uint32_t layer_idx, mars_layer_t *layer) {
    /* First, build a set of buffer indices that are "forbidden" (holding layer inputs) */
    int forbidden[MARS_MAX_WORK_BUFFERS] = {0};

    for (uint32_t i = 0; i < layer->num_inputs; i++) {
        uint32_t tid = layer->input_tensor_ids[i];
        if (tid >= model->header.num_tensors) continue;
        mars_runtime_tensor_t *rt = &model->tensors[tid];
        if (rt->buffer_idx >= 0 && rt->buffer_idx < (int8_t)model->num_work_buffers) {
            forbidden[rt->buffer_idx] = 1;
        }
    }

    /* Find a buffer that is not forbidden and is either free or holds a dead tensor */
    for (uint32_t b = 0; b < model->num_work_buffers; b++) {
        if (forbidden[b]) continue;

        int32_t held_tensor = model->buffer_tensor[b];
        if (held_tensor < 0) {
            /* Buffer never used - it's free */
            return (int)b;
        }

        /* Check if the held tensor is dead (last_used_at < current layer) */
        mars_runtime_tensor_t *held_rt = &model->tensors[held_tensor];
        if (held_rt->last_used_at < (int32_t)layer_idx) {
            /* Tensor is dead - buffer is available */
            return (int)b;
        }
    }

    /* No free buffer found - this shouldn't happen with proper liveness analysis */
    return -1;
}

/*
 * Assign buffers to output tensors before layer execution
 * Uses dynamic allocation via nna_malloc_phys() when working buffers are full
 */
static void assign_output_buffers(mars_model_t *model, uint32_t layer_idx) {
    mars_layer_t *layer = &model->layers[layer_idx].desc;

    for (uint32_t o = 0; o < layer->num_outputs; o++) {
        uint32_t tid = layer->output_tensor_ids[o];
        if (tid >= model->header.num_tensors) continue;

        mars_runtime_tensor_t *rt = &model->tensors[tid];

        /* Skip if already has dedicated memory (I/O or weight tensor) */
        if (rt->vaddr != NULL) continue;

        /* Try to find a free working buffer first */
        int buf_idx = find_free_buffer(model, layer_idx, layer);
        if (buf_idx >= 0) {
            /* Assign the working buffer */
            rt->vaddr = model->work_buffers[buf_idx];
            rt->paddr = model->work_buffers_paddr[buf_idx];
            rt->alloc_size = model->work_buffer_size;
            rt->buffer_idx = (int8_t)buf_idx;
            model->buffer_tensor[buf_idx] = (int32_t)tid;
        } else {
            /* No free working buffer - allocate dynamically */
            size_t sz = ALIGN_UP(tensor_byte_size(&rt->desc), 64);
            void *paddr = NULL;
            rt->vaddr = nna_malloc_phys(sz, &paddr);
            if (rt->vaddr == NULL) {
                fprintf(stderr, "Mars: ERROR - Failed to allocate tensor %u (%zu bytes) at layer %u!\n",
                        tid, sz, layer_idx);
                return;
            }
            rt->paddr = paddr;
            rt->alloc_size = sz;
            rt->buffer_idx = -3;  /* Mark as dynamically allocated at runtime */
        }
    }
}

/*
 * Free dynamically allocated tensors that are no longer needed after a layer
 */
static void free_dead_tensors(mars_model_t *model, uint32_t layer_idx) {
    for (uint32_t i = 0; i < model->header.num_tensors; i++) {
        mars_runtime_tensor_t *rt = &model->tensors[i];

        /* Only free tensors that were dynamically allocated at runtime */
        if (rt->buffer_idx != -3) continue;

        /* Check if this tensor is now dead (last_used_at <= current layer) */
        if (rt->last_used_at >= 0 && rt->last_used_at <= (int32_t)layer_idx) {
            /* This tensor is no longer needed - free it */
            nna_free(rt->vaddr);
            rt->vaddr = NULL;
            rt->paddr = NULL;
            rt->alloc_size = 0;
            rt->buffer_idx = -1;
        }
    }
}

mars_error_t mars_run(mars_model_t *model) {
    if (!model) return MARS_ERR_INVALID_FILE;

    /* Reset execution flags and buffer assignments */
    for (uint32_t i = 0; i < model->header.num_layers; i++) {
        model->layers[i].is_executed = false;
    }

    /* Reset working buffer tracking */
    for (uint32_t b = 0; b < model->num_work_buffers; b++) {
        model->buffer_tensor[b] = -1;  /* All buffers free */
    }

    /* Reset intermediate tensor assignments (keep I/O and weight assignments) */
    for (uint32_t i = 0; i < model->header.num_tensors; i++) {
        mars_runtime_tensor_t *rt = &model->tensors[i];
        if (rt->buffer_idx >= 0) {
            /* This was a dynamically assigned tensor - clear it */
            rt->vaddr = NULL;
            rt->paddr = NULL;
            rt->buffer_idx = -1;
        }
    }

    /* Execute layers in order with dynamic buffer assignment */
    for (uint32_t i = 0; i < model->header.num_layers; i++) {
        /* Dynamically assign output buffers for this layer */
        assign_output_buffers(model, i);

        /* Progress indicator every 20 layers */
        if ((i % 20 == 0) || i == model->header.num_layers - 1) {
            fprintf(stderr, "  Layer %u/%u...\r", i, model->header.num_layers);
            fflush(stderr);
        }

        mars_error_t err = execute_layer(model, &model->layers[i]);
        if (err != MARS_OK) {
            fprintf(stderr, "\nMars: Layer %u execution failed (type=%u)\n", i, model->layers[i].desc.type);
            return err;
        }
        model->layers[i].is_executed = true;

        /* Free dynamically allocated tensors that are no longer needed */
        free_dead_tensors(model, i);
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

    /* Check data format - NHWC (7) or NCHW (0) */
    int is_nhwc = (input->desc.format == MARS_FORMAT_NHWC);
    int out_is_nhwc = (output->desc.format == MARS_FORMAT_NHWC);

    /* Extract dimensions based on format */
    int in_c, in_h, in_w, out_c, out_h, out_w;
    if (is_nhwc) {
        /* NHWC: [N, H, W, C] */
        in_h = input->desc.shape[1];
        in_w = input->desc.shape[2];
        in_c = input->desc.shape[3];
    } else {
        /* NCHW: [N, C, H, W] */
        in_c = input->desc.shape[1];
        in_h = input->desc.shape[2];
        in_w = input->desc.shape[3];
    }

    /* Output format should match input format */
    if (out_is_nhwc) {
        /* NHWC: [N, H, W, C] */
        out_h = output->desc.shape[1];
        out_w = output->desc.shape[2];
        out_c = output->desc.shape[3];
    } else {
        /* NCHW: [N, C, H, W] */
        out_c = output->desc.shape[1];
        out_h = output->desc.shape[2];
        out_w = output->desc.shape[3];
    }

    /* Calculate padding based on mode */
    int pad_top = 0, pad_left = 0;
    if (params->padding == MARS_PAD_SAME) {
        int pad_h = (out_h - 1) * params->stride_h + params->kernel_h - in_h;
        int pad_w = (out_w - 1) * params->stride_w + params->kernel_w - in_w;
        pad_top = pad_h / 2;
        pad_left = pad_w / 2;
    } else if (params->padding == MARS_PAD_EXPLICIT) {
        /* Use explicit padding values from the model */
        pad_top = params->pad_top;
        pad_left = params->pad_left;
    }

    /* Check if float32 model */
    int is_float = (input->desc.dtype == MARS_DTYPE_FLOAT32);

    /* Debug: print scales for first few convs */
    static int conv_count = 0;
    if (conv_count < 3) {
        printf("  Conv2D[%d]: %dx%dx%d -> %dx%dx%d (k=%dx%d, s=%d) [%s%s%s]\n",
               conv_count, in_h, in_w, in_c, out_h, out_w, out_c,
               params->kernel_h, params->kernel_w, params->stride_h,
               is_float ? "F32-" : "INT8-",
               USE_MXU ? "MXU" : "SW",
               is_nhwc ? "-NHWC" : "");
        printf("    Scales: in=%.6f w=%.6f out=%.6f\n",
               input->desc.scale, weight->desc.scale, output->desc.scale);
        printf("    Combined scale: %.10f\n",
               (input->desc.scale * weight->desc.scale) / output->desc.scale);
        conv_count++;
    } else {
        printf("  Conv2D: %dx%dx%d -> %dx%dx%d (k=%dx%d, s=%d) [%s%s%s]\n",
               in_h, in_w, in_c, out_h, out_w, out_c,
               params->kernel_h, params->kernel_w, params->stride_h,
               is_float ? "F32-" : "INT8-",
               USE_MXU ? "MXU" : "SW",
               is_nhwc ? "-NHWC" : "");
    }

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
    } else if (is_nhwc) {
        /* INT8 NHWC MXU-accelerated convolution - faster gather */
        conv2d_int8_nhwc_mxu(
            (int8_t *)input->vaddr, in_h, in_w, in_c,
            (int8_t *)weight->vaddr, out_c, params->kernel_h, params->kernel_w,
            bias ? (int32_t *)bias->vaddr : NULL,
            (int8_t *)output->vaddr, out_h, out_w,
            params->stride_h, params->stride_w,
            pad_top, pad_left,
            input->desc.scale, weight->desc.scale, output->desc.scale
        );
    } else {
        /* INT8 NCHW MXU-accelerated convolution */
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
    } else if (is_nhwc) {
        /* INT8 NHWC software convolution */
        conv2d_int8_nhwc_mxu(
            (int8_t *)input->vaddr, in_h, in_w, in_c,
            (int8_t *)weight->vaddr, out_c, params->kernel_h, params->kernel_w,
            bias ? (int32_t *)bias->vaddr : NULL,
            (int8_t *)output->vaddr, out_h, out_w,
            params->stride_h, params->stride_w,
            pad_top, pad_left,
            input->desc.scale, weight->desc.scale, output->desc.scale
        );
    } else {
        /* INT8 NCHW software convolution */
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
    static int sigmoid_count = 0;
    sigmoid_count++;

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

    /* Debug output disabled for performance */
    (void)sigmoid_count;

    return MARS_OK;
}

/* Execute element-wise Mul: out = a * b */
static mars_error_t execute_mul(mars_model_t *model, mars_runtime_layer_t *layer) {
    const mars_layer_t *desc = &layer->desc;
    static int mul_count = 0;
    mul_count++;

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
        int32_t q = (int32_t)roundf(y * inv_scale_out);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        out[i] = (int8_t)q;
    }

    /* Debug output disabled for performance */
    (void)mul_count;

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
        int32_t q = (int32_t)roundf(y * inv_scale_out);
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

/* Execute Concat - generic implementation for any axis */
static mars_error_t execute_concat(mars_model_t *model, mars_runtime_layer_t *layer) {
    const mars_layer_t *desc = &layer->desc;
    const mars_concat_params_t *params = &desc->params.concat;

    mars_runtime_tensor_t *output = get_tensor_by_id(model, desc->output_tensor_ids[0]);
    if (!output || !output->vaddr) {
        return MARS_ERR_INVALID_TENSOR;
    }

    int8_t *out = (int8_t *)output->vaddr;
    uint32_t axis = params->axis;

    /* Debug: count concats to identify FPN layers */
    static int concat_count = 0;
    concat_count++;

    /* Debug: print key FPN Concat layers */
    /* In NHWC YOLOv5n:
     * Concat around layer 44-45: 80x80 first FPN (model.17)
     * Concat around layer 52-53: 40x40 second FPN (model.20)
     * Concat around layer 59-60: 20x20 third FPN (model.23)
     */
    int32_t out_h = output->desc.shape[1];
    int32_t out_w = output->desc.shape[2];
    int32_t out_c = output->desc.shape[3];

    /* Debug output disabled for performance */
    (void)out_h; (void)out_w; (void)out_c; (void)concat_count;

    /*
     * Generic concat: for each input, copy data at the right offset along the axis.
     * For axis=N, we need to interleave data from different inputs along that axis.
     *
     * Optimization: if axis is the last non-trivial dimension, we can do sequential copies.
     */

    /* Get output dimensions */
    int32_t out_shape[4];
    for (int i = 0; i < 4; i++) {
        out_shape[i] = output->desc.shape[i] > 0 ? output->desc.shape[i] : 1;
    }

    /* Calculate stride for the axis dimension */
    int64_t axis_stride = 1;
    for (uint32_t i = axis + 1; i < 4; i++) {
        axis_stride *= out_shape[i];
    }

    /* Calculate stride for dimensions before axis */
    int64_t outer_count = 1;
    for (uint32_t i = 0; i < axis; i++) {
        outer_count *= out_shape[i];
    }

    /* For each input, calculate where it goes in the output */
    int32_t axis_offset = 0;
    float out_scale = output->desc.scale > 0.0f ? output->desc.scale : 1.0f;

    for (uint32_t n = 0; n < desc->num_inputs; n++) {
        mars_runtime_tensor_t *input = get_tensor_by_id(model, desc->input_tensor_ids[n]);
        if (!input || !input->vaddr) continue;

        int8_t *in = (int8_t *)input->vaddr;
        float in_scale = input->desc.scale > 0.0f ? input->desc.scale : 1.0f;

        /* Check if we need to rescale (input and output scales differ significantly) */
        float scale_ratio = in_scale / out_scale;
        int needs_rescale = (fabsf(scale_ratio - 1.0f) > 0.01f);

        /* Get input shape */
        int32_t in_shape[4];
        for (int i = 0; i < 4; i++) {
            in_shape[i] = input->desc.shape[i] > 0 ? input->desc.shape[i] : 1;
        }

        int32_t in_axis_size = in_shape[axis];

        /* Calculate input element count */
        int64_t in_axis_stride = 1;
        for (uint32_t i = axis + 1; i < 4; i++) {
            in_axis_stride *= in_shape[i];
        }

        /* Copy data (with optional rescaling) */
        /* For each position in the outer dimensions */
        for (int64_t outer = 0; outer < outer_count; outer++) {
            /* For each position along the axis in this input */
            for (int32_t a = 0; a < in_axis_size; a++) {
                /* Calculate offsets for this slice */
                int64_t in_offset = outer * (in_axis_size * in_axis_stride) + a * in_axis_stride;
                int64_t out_offset = outer * (out_shape[axis] * axis_stride) + (axis_offset + a) * axis_stride;

                if (needs_rescale) {
                    /* Rescale each element from input scale to output scale */
                    for (int64_t j = 0; j < axis_stride; j++) {
                        float val = in[in_offset + j] * scale_ratio;
                        int32_t q = (int32_t)roundf(val);
                        if (q > 127) q = 127;
                        if (q < -128) q = -128;
                        out[out_offset + j] = (int8_t)q;
                    }
                } else {
                    /* Direct copy - scales match */
                    memcpy(out + out_offset, in + in_offset, axis_stride);
                }
            }
        }

        axis_offset += in_axis_size;
    }

    /* Debug: print concat output for FPN layers */
    if (out_c >= 64 && out_c <= 256) {
        fprintf(stderr, "    Output first 16 NHWC: ");
        for (int i = 0; i < 16; i++) fprintf(stderr, "%d ", out[i]);
        fprintf(stderr, "\n");
    }

    return MARS_OK;
}

/* Execute Reshape: copy data to output with new shape interpretation */
static mars_error_t execute_reshape(mars_model_t *model, mars_runtime_layer_t *layer) {
    const mars_layer_t *desc = &layer->desc;

    mars_runtime_tensor_t *input = get_tensor_by_id(model, desc->input_tensor_ids[0]);
    mars_runtime_tensor_t *output = get_tensor_by_id(model, desc->output_tensor_ids[0]);

    if (!input || !output || !input->vaddr || !output->vaddr) {
        return MARS_ERR_INVALID_TENSOR;
    }

    /* Calculate output size - use this as the copy size since input may have
     * truncated shape (5D stored as 4D loses a dimension) */
    size_t output_size = 1;
    for (int i = 0; i < 4; i++) {
        if (output->desc.shape[i] > 0) {
            output_size *= output->desc.shape[i];
        }
    }

    /* Use output size as copy size - the input buffer should have this much data
     * even if its stored shape is truncated */
    size_t copy_size = output_size;

    printf("  [DEBUG Reshape] in_id=%u vaddr=%p -> out_id=%u vaddr=%p copy_size=%zu\n",
           desc->input_tensor_ids[0], input->vaddr,
           desc->output_tensor_ids[0], output->vaddr, copy_size);
    printf("    Input shape: [%d,%d,%d,%d], Output shape: [%d,%d,%d,%d]\n",
           input->desc.shape[0], input->desc.shape[1], input->desc.shape[2], input->desc.shape[3],
           output->desc.shape[0], output->desc.shape[1], output->desc.shape[2], output->desc.shape[3]);
    int8_t *in8 = (int8_t*)input->vaddr;
    printf("    Input first 16: ");
    for (int i = 0; i < 16; i++) printf("%d ", in8[i]);
    printf("\n");

    /* Simple memcpy - reshape is just reinterpretation */
    memcpy(output->vaddr, input->vaddr, copy_size);

    return MARS_OK;
}

/* Execute Transpose: permute dimensions */
static mars_error_t execute_transpose(mars_model_t *model, mars_runtime_layer_t *layer) {
    const mars_layer_t *desc = &layer->desc;
    const mars_transpose_params_t *params = &desc->params.transpose;

    mars_runtime_tensor_t *input = get_tensor_by_id(model, desc->input_tensor_ids[0]);
    mars_runtime_tensor_t *output = get_tensor_by_id(model, desc->output_tensor_ids[0]);

    if (!input || !output || !input->vaddr || !output->vaddr) {
        return MARS_ERR_INVALID_TENSOR;
    }

    uint32_t ndims = params->ndims;
    if (ndims > 6) ndims = 6;

    /* Get input shape and compute strides */
    int32_t in_shape[6] = {1, 1, 1, 1, 1, 1};
    int32_t in_stride[6] = {1, 1, 1, 1, 1, 1};
    int32_t total = 1;

    /* Copy input shape - use raw shape data for high-dim tensors */
    for (uint32_t i = 0; i < ndims; i++) {
        /* For tensors with ndims > 4, the shape may be packed differently */
        /* Check if there's shape info in the raw params or use output shape */
        if (i < 4) {
            in_shape[i] = input->desc.shape[i];
        } else {
            /* For dims 4+, we need to get from reshape context */
            /* YOLOv5: [1, 3, 85, H, W] - dims 3,4 come from spatial dims */
            in_shape[i] = 1;  /* Will be overwritten from context */
        }
    }

    /* YOLOv5 specific: perm=[0,1,3,4,2] on [1, 3, 85, H, W]
     * Since we store tensors as 4D max in mars_tensor_t, we need special handling
     * For YOLOv5 head outputs: the Conv produces [H, W, 255] in NHWC
     * After reshape [1, 3, 85, H, W] -> transpose [0,1,3,4,2] -> [1, 3, H, W, 85]
     *
     * But due to reshape aliasing, the actual data layout is still [H, W, 255]
     * and we need to reorganize to [H, W, 3, 85] for proper anchor/class interleaving
     */

    /* Get tensor sizes */
    size_t in_size = 1;
    for (uint32_t i = 0; i < input->desc.ndims && i < 4; i++) {
        if (input->desc.shape[i] > 0)
            in_size *= input->desc.shape[i];
    }

    int8_t *src = (int8_t *)input->vaddr;
    int8_t *dst = (int8_t *)output->vaddr;

    /* For 5D transpose perm=[0,1,3,4,2] on [1, 3, 85, H, W]:
     * This transposes the last 3 dims: [85, H, W] -> [H, W, 85]
     * In our NHWC layout with reshape aliasing, the tensor is [H, W, 255]
     * where 255 = 3 * 85 (3 anchors, 85 values per anchor)
     *
     * After transpose+reshape, we need [H*W*3, 85] = [num_boxes, 85]
     *
     * The data in NHWC is already [H, W, C] where C = 255 = 3*85
     * We need to reinterpret as [H, W, 3, 85] then reshape to [H*W*3, 85]
     * This is just a reshape - no actual data movement needed!
     */

    /* Check if this is just a no-op reshape situation */
    /* For YOLOv5 heads, the transpose just reinterprets the anchor/class ordering */
    if (ndims == 5 && params->perm[0] == 0 && params->perm[1] == 1 &&
        params->perm[2] == 3 && params->perm[3] == 4 && params->perm[4] == 2) {
        /* YOLOv5 specific: [1, 3, 85, H, W] -> [1, 3, H, W, 85]
         * In NHWC this is stored as [1, H, W, 255] -> [1, H, W, 255]
         * Just copy the data - the layout is already correct for row-major access
         */
        memcpy(dst, src, in_size * sizeof(int8_t));
        return MARS_OK;
    }

    /* General transpose - compute strides and permute */
    for (int i = (int)ndims - 1; i >= 0; i--) {
        in_stride[i] = total;
        total *= in_shape[i];
    }

    /* Compute output strides based on permutation */
    int32_t out_shape[6], out_stride[6];
    for (uint32_t i = 0; i < ndims; i++) {
        out_shape[i] = in_shape[params->perm[i]];
    }
    total = 1;
    for (int i = (int)ndims - 1; i >= 0; i--) {
        out_stride[i] = total;
        total *= out_shape[i];
    }

    /* Simple element-wise transpose (slow but correct) */
    int32_t coords[6] = {0};
    for (int32_t idx = 0; idx < total; idx++) {
        /* Convert flat index to output coordinates */
        int32_t tmp = idx;
        for (int d = 0; d < (int)ndims; d++) {
            coords[d] = tmp / out_stride[d];
            tmp = tmp % out_stride[d];
        }

        /* Map output coords to input coords via inverse perm */
        int32_t in_idx = 0;
        for (uint32_t d = 0; d < ndims; d++) {
            in_idx += coords[d] * in_stride[params->perm[d]];
        }

        dst[idx] = src[in_idx];
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
            return execute_reshape(model, layer);

        case MARS_LAYER_TRANSPOSE:
            return execute_transpose(model, layer);

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

