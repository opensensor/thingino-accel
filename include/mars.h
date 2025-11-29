/*
 * Mars - Open Neural Network Compiler for Ingenic T41 NNA
 *
 * This is a clean-room implementation of a neural network format and runtime
 * for the Ingenic T41 Neural Network Accelerator.
 *
 * Copyright (c) 2024 OpenSensor Project
 * SPDX-License-Identifier: MIT
 */

#ifndef MARS_H
#define MARS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Magic number: "MARS" in little-endian */
#define MARS_MAGIC 0x5352414D

/* Current format version */
#define MARS_VERSION_MAJOR 1
#define MARS_VERSION_MINOR 0

/* Maximum supported values */
#define MARS_MAX_DIMS 6
#define MARS_MAX_NAME_LEN 64
#define MARS_MAX_LAYERS 256
#define MARS_MAX_TENSORS 512

/* Data types - matches NNA hardware capabilities */
typedef enum {
    MARS_DTYPE_FLOAT32 = 0,
    MARS_DTYPE_INT32 = 1,
    MARS_DTYPE_INT16 = 2,
    MARS_DTYPE_INT8 = 3,
    MARS_DTYPE_UINT8 = 4,
    MARS_DTYPE_UINT4 = 5,  /* 4-bit packed */
} mars_dtype_t;

/* Tensor format - matching NNA hardware expectations */
typedef enum {
    MARS_FORMAT_NHWC = 0,       /* Feature: [N, H, W, C] - basic layout */
    MARS_FORMAT_NDHWC32 = 1,    /* Feature: [N, D_C32, H, W, 32] - NNA native, 32-channel groups */
    MARS_FORMAT_HWIO = 2,       /* Weight: [H, W, I, O] */
    MARS_FORMAT_NMHWSOIB2 = 3,  /* Weight: [N_OFP, M_IFP, H, W, S, OFP, IFP] - NNA packed */
    MARS_FORMAT_NMC32 = 4,      /* Bias/BN: [N_OFP, M_BT, 32] */
    MARS_FORMAT_D1 = 5,         /* Scale/LUT: [d1] */
    MARS_FORMAT_OHWI = 6,       /* Weight: [O, H, W, I] */
    MARS_FORMAT_NCHW = 7,       /* Feature: [N, C, H, W] - ONNX default */
    MARS_FORMAT_OIHW = 8,       /* Weight: [O, I, H, W] - ONNX default */
} mars_format_t;

/* Layer types */
typedef enum {
    MARS_LAYER_CONV2D = 0,
    MARS_LAYER_DEPTHWISE_CONV2D = 1,
    MARS_LAYER_MAXPOOL = 2,
    MARS_LAYER_AVGPOOL = 3,
    MARS_LAYER_GLOBAL_AVGPOOL = 4,
    MARS_LAYER_RELU = 5,
    MARS_LAYER_RELU6 = 6,
    MARS_LAYER_LEAKY_RELU = 7,
    MARS_LAYER_SILU = 8,
    MARS_LAYER_SIGMOID = 9,
    MARS_LAYER_CONCAT = 10,
    MARS_LAYER_ADD = 11,
    MARS_LAYER_MUL = 12,
    MARS_LAYER_UPSAMPLE = 13,
    MARS_LAYER_RESHAPE = 14,
    MARS_LAYER_SOFTMAX = 15,
    MARS_LAYER_FC = 16,     /* Fully connected */
} mars_layer_type_t;

/* Activation type (fused with conv/fc) */
typedef enum {
    MARS_ACT_NONE = 0,
    MARS_ACT_RELU = 1,
    MARS_ACT_RELU6 = 2,
    MARS_ACT_LEAKY_RELU = 3,
    MARS_ACT_SILU = 4,
    MARS_ACT_SIGMOID = 5,
    MARS_ACT_TANH = 6,
    MARS_ACT_HARD_SWISH = 7,
} mars_activation_t;

/* Padding mode */
typedef enum {
    MARS_PAD_VALID = 0,     /* No padding */
    MARS_PAD_SAME = 1,      /* Pad to keep same size with stride=1 */
    MARS_PAD_EXPLICIT = 2,  /* Use explicit padding values */
} mars_padding_t;

/*
 * File Header (64 bytes)
 */
typedef struct __attribute__((packed)) {
    uint32_t magic;              /* MARS_MAGIC */
    uint16_t version_major;
    uint16_t version_minor;
    uint32_t flags;              /* Reserved for future use */
    uint32_t num_layers;
    uint32_t num_tensors;
    uint32_t num_inputs;
    uint32_t num_outputs;
    uint64_t weights_offset;     /* Offset to weight data */
    uint64_t weights_size;       /* Total weight data size */
    uint32_t input_tensor_ids[4];   /* Up to 4 input tensors */
    uint32_t output_tensor_ids[4];  /* Up to 4 output tensors */
} mars_header_t;

/*
 * Tensor Descriptor (64 bytes)
 */
typedef struct __attribute__((packed)) {
    uint32_t id;
    char name[MARS_MAX_NAME_LEN - 4];  /* Tensor name */
    mars_dtype_t dtype;
    mars_format_t format;
    uint32_t ndims;
    int32_t shape[MARS_MAX_DIMS];
    uint64_t data_offset;        /* Offset in weights section (0 if runtime) */
    uint64_t data_size;          /* Size in bytes */
    float scale;                 /* Quantization scale */
    int32_t zero_point;          /* Quantization zero point */
} mars_tensor_t;

/*
 * Layer-specific parameters
 */

/* Conv2D / DepthwiseConv2D parameters */
typedef struct __attribute__((packed)) {
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t dilation_h;
    uint32_t dilation_w;
    mars_padding_t padding;
    uint32_t pad_top;
    uint32_t pad_bottom;
    uint32_t pad_left;
    uint32_t pad_right;
    uint32_t groups;             /* 1 for conv, in_channels for depthwise */
    mars_activation_t activation;
    uint32_t weight_tensor_id;
    uint32_t bias_tensor_id;     /* 0xFFFFFFFF if no bias */
} mars_conv_params_t;

/* Pooling parameters */
typedef struct __attribute__((packed)) {
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t stride_h;
    uint32_t stride_w;
    mars_padding_t padding;
    uint32_t pad_top;
    uint32_t pad_bottom;
    uint32_t pad_left;
    uint32_t pad_right;
} mars_pool_params_t;

/* Activation parameters */
typedef struct __attribute__((packed)) {
    float alpha;                 /* For LeakyReLU slope */
} mars_act_params_t;

/* Concat parameters */
typedef struct __attribute__((packed)) {
    uint32_t axis;
    uint32_t num_inputs;
} mars_concat_params_t;

/* Upsample parameters */
typedef struct __attribute__((packed)) {
    uint32_t scale_h;
    uint32_t scale_w;
    uint32_t mode;               /* 0=nearest, 1=bilinear */
} mars_upsample_params_t;

/* Reshape parameters */
typedef struct __attribute__((packed)) {
    int32_t new_shape[MARS_MAX_DIMS];
    uint32_t ndims;
} mars_reshape_params_t;

/* Fully connected parameters */
typedef struct __attribute__((packed)) {
    uint32_t weight_tensor_id;
    uint32_t bias_tensor_id;
    mars_activation_t activation;
} mars_fc_params_t;

/*
 * Layer Descriptor (128 bytes)
 */
typedef struct __attribute__((packed)) {
    uint32_t id;
    mars_layer_type_t type;
    uint32_t num_inputs;
    uint32_t num_outputs;
    uint32_t input_tensor_ids[4];
    uint32_t output_tensor_ids[4];
    union {
        mars_conv_params_t conv;
        mars_pool_params_t pool;
        mars_act_params_t act;
        mars_concat_params_t concat;
        mars_upsample_params_t upsample;
        mars_reshape_params_t reshape;
        mars_fc_params_t fc;
        uint8_t raw[64];
    } params;
} mars_layer_t;

/*
 * Complete .mars file structure:
 *
 * +------------------+
 * | mars_header_t    |  64 bytes
 * +------------------+
 * | mars_tensor_t[]  |  num_tensors * 64 bytes
 * +------------------+
 * | mars_layer_t[]   |  num_layers * 128 bytes
 * +------------------+
 * | weight data      |  variable size (aligned to 64 bytes)
 * +------------------+
 */

#ifdef __cplusplus
}
#endif

#endif /* MARS_H */

