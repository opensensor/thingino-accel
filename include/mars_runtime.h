/*
 * Mars Runtime - Execute .mars models on Ingenic T41 NNA
 *
 * Copyright (c) 2024 OpenSensor Project
 * SPDX-License-Identifier: MIT
 */

#ifndef MARS_RUNTIME_H
#define MARS_RUNTIME_H

#include "mars.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes */
typedef enum {
    MARS_OK = 0,
    MARS_ERR_INVALID_MAGIC = -1,
    MARS_ERR_VERSION_MISMATCH = -2,
    MARS_ERR_ALLOC_FAILED = -3,
    MARS_ERR_INVALID_FILE = -4,
    MARS_ERR_NNA_INIT_FAILED = -5,
    MARS_ERR_LAYER_FAILED = -6,
    MARS_ERR_INVALID_TENSOR = -7,
    MARS_ERR_INVALID_LAYER = -8,
} mars_error_t;

/* Runtime tensor - holds actual data */
typedef struct {
    mars_tensor_t desc;          /* Tensor descriptor from file */
    void *vaddr;                 /* Virtual address (CPU accessible) */
    void *paddr;                 /* Physical address (NNA accessible) */
    size_t alloc_size;           /* Allocated size (may be larger due to alignment) */
    bool is_external;            /* True if memory is externally managed */
} mars_runtime_tensor_t;

/* Runtime layer */
typedef struct {
    mars_layer_t desc;           /* Layer descriptor from file */
    bool is_executed;            /* Track if layer has been executed this run */
} mars_runtime_layer_t;

/* Runtime model context */
typedef struct {
    mars_header_t header;
    mars_runtime_tensor_t *tensors;
    mars_runtime_layer_t *layers;

    /* NNA hardware resources */
    void *ddr_base;              /* DDR buffer for weights/activations */
    void *ddr_paddr;             /* Physical address of DDR buffer */
    size_t ddr_size;
    void *oram_base;             /* On-chip SRAM buffer */
    void *oram_paddr;
    size_t oram_size;

    /* Weight data (loaded from file) */
    void *weights;
    size_t weights_size;

    /* Statistics */
    uint64_t total_inference_us;
    uint32_t inference_count;
} mars_model_t;

/*
 * API Functions
 */

/**
 * Load a .mars model from a file path
 * @param path Path to .mars file
 * @param model Output model handle
 * @return MARS_OK on success
 */
mars_error_t mars_load_file(const char *path, mars_model_t **model);

/**
 * Load a .mars model from memory
 * @param data Pointer to model data
 * @param size Size of model data in bytes
 * @param model Output model handle
 * @return MARS_OK on success
 */
mars_error_t mars_load_memory(const void *data, size_t size, mars_model_t **model);

/**
 * Free a loaded model
 * @param model Model to free
 */
void mars_free(mars_model_t *model);

/**
 * Get input tensor by index
 * @param model Model handle
 * @param index Input tensor index
 * @return Tensor pointer or NULL if invalid
 */
mars_runtime_tensor_t* mars_get_input(mars_model_t *model, int index);

/**
 * Get output tensor by index
 * @param model Model handle
 * @param index Output tensor index
 * @return Tensor pointer or NULL if invalid
 */
mars_runtime_tensor_t* mars_get_output(mars_model_t *model, int index);

/**
 * Run inference
 * @param model Model handle
 * @return MARS_OK on success
 */
mars_error_t mars_run(mars_model_t *model);

/**
 * Get last error message
 * @return Human-readable error string
 */
const char* mars_get_error_string(mars_error_t err);

/**
 * Get number of inputs
 */
int mars_get_num_inputs(mars_model_t *model);

/**
 * Get number of outputs
 */
int mars_get_num_outputs(mars_model_t *model);

/**
 * Print model summary
 */
void mars_print_summary(mars_model_t *model);

#ifdef __cplusplus
}
#endif

#endif /* MARS_RUNTIME_H */

