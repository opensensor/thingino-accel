/*
 * thingino-accel - Open Source NNA Library for Ingenic T41/T31
 * 
 * Basic type definitions for NNA operations
 */

#ifndef THINGINO_ACCEL_NNA_TYPES_H
#define THINGINO_ACCEL_NNA_TYPES_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Return codes */
#define NNA_SUCCESS           0
#define NNA_ERROR_INIT       -1
#define NNA_ERROR_DEVICE     -2
#define NNA_ERROR_MEMORY     -3
#define NNA_ERROR_INVALID    -4
#define NNA_ERROR_TIMEOUT    -5

/* Tensor data types */
typedef enum {
    NNA_DTYPE_FLOAT32 = 0,
    NNA_DTYPE_FLOAT16 = 1,
    NNA_DTYPE_INT8    = 2,
    NNA_DTYPE_UINT8   = 3,
    NNA_DTYPE_INT16   = 4,
    NNA_DTYPE_UINT16  = 5,
    NNA_DTYPE_INT32   = 6,
    NNA_DTYPE_UINT32  = 7,
} nna_dtype_t;

/* Tensor format */
typedef enum {
    NNA_FORMAT_NHWC = 1,  /* Batch, Height, Width, Channels */
    NNA_FORMAT_NV12 = 5,  /* YUV 4:2:0 format */
} nna_format_t;

/* Memory allocation flags */
typedef enum {
    NNA_MEM_DDR  = 0,  /* DDR memory (cacheable) */
    NNA_MEM_ORAM = 1,  /* On-chip RAM (384KB on T41) */
} nna_mem_type_t;

/* Tensor shape (max 4 dimensions: NHWC) */
typedef struct {
    int32_t dims[4];   /* N, H, W, C */
    int32_t ndim;      /* Number of dimensions */
} nna_shape_t;

/* Tensor descriptor */
typedef struct {
    void *data;           /* Pointer to data */
    nna_shape_t shape;    /* Tensor shape */
    nna_dtype_t dtype;    /* Data type */
    nna_format_t format;  /* Data format */
    size_t bytes;         /* Total bytes */
    int owns_data;        /* Whether tensor owns the data */
} nna_tensor_t;

/* Hardware info */
typedef struct {
    uint32_t oram_vbase;      /* ORAM virtual base address */
    uint32_t oram_pbase;      /* ORAM physical base address */
    uint32_t oram_size;       /* ORAM size in bytes */
    uint32_t version;         /* NNA hardware version */
} nna_hw_info_t;

/* Model handle (opaque) */
typedef struct nna_model nna_model_t;

#ifdef __cplusplus
}
#endif

#endif /* THINGINO_ACCEL_NNA_TYPES_H */

