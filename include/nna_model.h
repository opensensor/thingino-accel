/*
 * thingino-accel - Model Loading and Inference
 * 
 * Functions for loading .mgk models and running inference
 */

#ifndef THINGINO_ACCEL_NNA_MODEL_H
#define THINGINO_ACCEL_NNA_MODEL_H

#include <stdint.h>
#include <stddef.h>
#include "nna_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Model handle (opaque) */
typedef struct nna_model nna_model_t;

/* Model loading options */
typedef struct {
    int use_file_mapping;    /* Use mmap instead of loading to memory */
    int enable_profiling;    /* Enable per-layer profiling */
    void *forward_memory;    /* Pre-allocated forward pass memory (optional) */
    size_t forward_mem_size; /* Size of forward memory */
} nna_model_options_t;

/* Model information */
typedef struct {
    uint32_t num_inputs;     /* Number of input tensors */
    uint32_t num_outputs;    /* Number of output tensors */
    uint32_t num_layers;     /* Number of layers */
    size_t model_size;       /* Model file size in bytes */
    size_t forward_mem_req;  /* Required forward pass memory */
} nna_model_info_t;

/**
 * Load a .mgk model from file
 * 
 * @param path Path to .mgk model file
 * @param options Loading options (NULL for defaults)
 * @return Model handle or NULL on error
 */
nna_model_t* nna_model_load(const char *path, const nna_model_options_t *options);

/**
 * Load a .mgk model from memory buffer
 * 
 * @param buffer Pointer to model data in memory
 * @param size Size of model data
 * @param options Loading options (NULL for defaults)
 * @return Model handle or NULL on error
 */
nna_model_t* nna_model_load_from_memory(const void *buffer, size_t size,
                                         const nna_model_options_t *options);

/**
 * Get model information
 * 
 * @param model Model handle
 * @param info Pointer to info structure to fill
 * @return NNA_SUCCESS or error code
 */
int nna_model_get_info(nna_model_t *model, nna_model_info_t *info);

/**
 * Get input tensor by index
 * 
 * @param model Model handle
 * @param index Input index (0-based)
 * @return Tensor handle or NULL if index invalid
 */
nna_tensor_t* nna_model_get_input(nna_model_t *model, uint32_t index);

/**
 * Get input tensor by name
 * 
 * @param model Model handle
 * @param name Input tensor name
 * @return Tensor handle or NULL if not found
 */
nna_tensor_t* nna_model_get_input_by_name(nna_model_t *model, const char *name);

/**
 * Get output tensor by index
 * 
 * @param model Model handle
 * @param index Output index (0-based)
 * @return Tensor handle or NULL if index invalid
 */
const nna_tensor_t* nna_model_get_output(nna_model_t *model, uint32_t index);

/**
 * Get output tensor by name
 * 
 * @param model Model handle
 * @param name Output tensor name
 * @return Tensor handle or NULL if not found
 */
const nna_tensor_t* nna_model_get_output_by_name(nna_model_t *model, const char *name);

/**
 * Run inference on the model
 * 
 * @param model Model handle
 * @return NNA_SUCCESS or error code
 */
int nna_model_run(nna_model_t *model);

/**
 * Unload model and free resources
 * 
 * @param model Model handle
 */
void nna_model_unload(nna_model_t *model);

#ifdef __cplusplus
}
#endif

#endif /* THINGINO_ACCEL_NNA_MODEL_H */

