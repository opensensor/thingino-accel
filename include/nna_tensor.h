/*
 * thingino-accel - Open Source NNA Library for Ingenic T41/T31
 * 
 * Tensor operations API
 */

#ifndef THINGINO_ACCEL_NNA_TENSOR_H
#define THINGINO_ACCEL_NNA_TENSOR_H

#include "nna_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Create a new tensor with specified shape
 * 
 * Memory is allocated automatically.
 * 
 * Args:
 *   shape: Tensor shape (N, H, W, C)
 *   dtype: Data type
 *   format: Data format
 * 
 * Returns:
 *   Pointer to tensor, or NULL on failure
 */
nna_tensor_t* nna_tensor_create(const nna_shape_t *shape, 
                                 nna_dtype_t dtype,
                                 nna_format_t format);

/*
 * Create a tensor from existing data
 * 
 * The tensor does not own the data - caller must manage memory.
 * 
 * Args:
 *   data: Pointer to existing data
 *   shape: Tensor shape
 *   dtype: Data type
 *   format: Data format
 * 
 * Returns:
 *   Pointer to tensor, or NULL on failure
 */
nna_tensor_t* nna_tensor_from_data(void *data,
                                    const nna_shape_t *shape,
                                    nna_dtype_t dtype,
                                    nna_format_t format);

/*
 * Destroy a tensor and free its memory
 * 
 * If the tensor owns its data, the data is freed.
 * 
 * Args:
 *   tensor: Tensor to destroy
 */
void nna_tensor_destroy(nna_tensor_t *tensor);

/*
 * Get tensor data pointer
 * 
 * Args:
 *   tensor: Tensor
 * 
 * Returns:
 *   Pointer to tensor data
 */
void* nna_tensor_data(const nna_tensor_t *tensor);

/*
 * Get tensor shape
 * 
 * Args:
 *   tensor: Tensor
 * 
 * Returns:
 *   Pointer to shape structure
 */
const nna_shape_t* nna_tensor_shape(const nna_tensor_t *tensor);

/*
 * Get tensor data type
 * 
 * Args:
 *   tensor: Tensor
 * 
 * Returns:
 *   Data type
 */
nna_dtype_t nna_tensor_dtype(const nna_tensor_t *tensor);

/*
 * Get total number of elements in tensor
 * 
 * Args:
 *   tensor: Tensor
 * 
 * Returns:
 *   Number of elements (N * H * W * C)
 */
size_t nna_tensor_numel(const nna_tensor_t *tensor);

/*
 * Get total size in bytes
 * 
 * Args:
 *   tensor: Tensor
 * 
 * Returns:
 *   Size in bytes
 */
size_t nna_tensor_bytes(const nna_tensor_t *tensor);

/*
 * Reshape tensor (does not reallocate memory)
 * 
 * Total number of elements must remain the same.
 * 
 * Args:
 *   tensor: Tensor to reshape
 *   new_shape: New shape
 * 
 * Returns:
 *   NNA_SUCCESS on success
 *   NNA_ERROR_INVALID if shapes are incompatible
 */
int nna_tensor_reshape(nna_tensor_t *tensor, const nna_shape_t *new_shape);

/*
 * Helper: Create shape from dimensions
 * 
 * Args:
 *   n, h, w, c: Dimensions
 * 
 * Returns:
 *   Shape structure
 */
nna_shape_t nna_shape_make(int32_t n, int32_t h, int32_t w, int32_t c);

#ifdef __cplusplus
}
#endif

#endif /* THINGINO_ACCEL_NNA_TENSOR_H */

