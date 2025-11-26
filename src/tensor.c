/*
 * thingino-accel - Tensor implementation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nna_tensor.h"
#include "nna_memory.h"

/* Get size of data type in bytes */
static size_t dtype_size(nna_dtype_t dtype) {
    switch (dtype) {
        case NNA_DTYPE_FLOAT32: return 4;
        case NNA_DTYPE_FLOAT16: return 2;
        case NNA_DTYPE_INT8:    return 1;
        case NNA_DTYPE_UINT8:   return 1;
        case NNA_DTYPE_INT16:   return 2;
        case NNA_DTYPE_UINT16:  return 2;
        case NNA_DTYPE_INT32:   return 4;
        case NNA_DTYPE_UINT32:  return 4;
        default: return 0;
    }
}

/* Calculate total number of elements */
static size_t shape_numel(const nna_shape_t *shape) {
    size_t numel = 1;
    for (int i = 0; i < shape->ndim; i++) {
        numel *= shape->dims[i];
    }
    return numel;
}

nna_tensor_t* nna_tensor_create(const nna_shape_t *shape,
                                 nna_dtype_t dtype,
                                 nna_format_t format) {
    if (shape == NULL || shape->ndim <= 0 || shape->ndim > 4) {
        return NULL;
    }

    nna_tensor_t *tensor = malloc(sizeof(nna_tensor_t));
    if (tensor == NULL) {
        return NULL;
    }

    /* Copy shape */
    memcpy(&tensor->shape, shape, sizeof(nna_shape_t));
    tensor->dtype = dtype;
    tensor->format = format;

    /* Calculate size and allocate memory */
    size_t numel = shape_numel(shape);
    size_t elem_size = dtype_size(dtype);
    tensor->bytes = numel * elem_size;

    tensor->data = nna_malloc(tensor->bytes);
    if (tensor->data == NULL) {
        free(tensor);
        return NULL;
    }

    tensor->owns_data = 1;

    return tensor;
}

nna_tensor_t* nna_tensor_from_data(void *data,
                                    const nna_shape_t *shape,
                                    nna_dtype_t dtype,
                                    nna_format_t format) {
    if (data == NULL || shape == NULL || shape->ndim <= 0 || shape->ndim > 4) {
        return NULL;
    }

    nna_tensor_t *tensor = malloc(sizeof(nna_tensor_t));
    if (tensor == NULL) {
        return NULL;
    }

    memcpy(&tensor->shape, shape, sizeof(nna_shape_t));
    tensor->dtype = dtype;
    tensor->format = format;
    tensor->data = data;
    tensor->owns_data = 0;

    size_t numel = shape_numel(shape);
    size_t elem_size = dtype_size(dtype);
    tensor->bytes = numel * elem_size;

    return tensor;
}

void nna_tensor_destroy(nna_tensor_t *tensor) {
    if (tensor == NULL) {
        return;
    }

    if (tensor->owns_data && tensor->data != NULL) {
        nna_free(tensor->data);
    }

    free(tensor);
}

void* nna_tensor_data(const nna_tensor_t *tensor) {
    return tensor ? tensor->data : NULL;
}

const nna_shape_t* nna_tensor_shape(const nna_tensor_t *tensor) {
    return tensor ? &tensor->shape : NULL;
}

nna_dtype_t nna_tensor_dtype(const nna_tensor_t *tensor) {
    return tensor ? tensor->dtype : NNA_DTYPE_FLOAT32;
}

size_t nna_tensor_numel(const nna_tensor_t *tensor) {
    return tensor ? shape_numel(&tensor->shape) : 0;
}

size_t nna_tensor_bytes(const nna_tensor_t *tensor) {
    return tensor ? tensor->bytes : 0;
}

int nna_tensor_reshape(nna_tensor_t *tensor, const nna_shape_t *new_shape) {
    if (tensor == NULL || new_shape == NULL) {
        return NNA_ERROR_INVALID;
    }

    /* Check that total elements match */
    size_t old_numel = shape_numel(&tensor->shape);
    size_t new_numel = shape_numel(new_shape);

    if (old_numel != new_numel) {
        return NNA_ERROR_INVALID;
    }

    memcpy(&tensor->shape, new_shape, sizeof(nna_shape_t));
    return NNA_SUCCESS;
}

nna_shape_t nna_shape_make(int32_t n, int32_t h, int32_t w, int32_t c) {
    nna_shape_t shape;
    shape.dims[0] = n;
    shape.dims[1] = h;
    shape.dims[2] = w;
    shape.dims[3] = c;
    shape.ndim = 4;
    return shape;
}

