/*
 * MXUv3-accelerated element-wise operations
 *
 * Copyright (c) 2024 OpenSensor Project
 * SPDX-License-Identifier: MIT
 */

#ifndef MXU_OPS_H
#define MXU_OPS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Initialize MXU for compute operations.
 * Must be called once with NNA-allocated memory before using MXU ops.
 */
void mxu_init(void *nna_mem);

/*
 * Check if MXU is initialized
 */
int mxu_is_initialized(void);

/*
 * Element-wise multiply: out[i] = a[i] * b[i]
 * Processes 16 floats at a time using VPR registers.
 */
void mxu_mul_f32(float *out, const float *a, const float *b, size_t count);

/*
 * Element-wise add: out[i] = a[i] + b[i]
 */
void mxu_add_f32(float *out, const float *a, const float *b, size_t count);

/*
 * Element-wise subtract: out[i] = a[i] - b[i]
 */
void mxu_sub_f32(float *out, const float *a, const float *b, size_t count);

/*
 * ReLU activation: out[i] = max(0, in[i])
 */
void mxu_relu_f32(float *out, const float *in, size_t count);

/*
 * INT8 convolution with MXU acceleration
 * Input/Output: NHWC format, Weights: OHWI format
 */
void conv2d_int8_mxu(
    const signed char *input, int in_h, int in_w, int in_c,
    const signed char *weight, int out_c, int kh, int kw,
    const int *bias,
    signed char *output, int out_h, int out_w,
    int stride_h, int stride_w,
    int pad_top, int pad_left,
    float in_scale, float w_scale, float out_scale);

/*
 * Float32 convolution with MXU SIMD acceleration
 * Uses VPR registers to process 16 floats at a time
 * Input/Output: NHWC format, Weights: OHWI format
 * scratch: 64-byte aligned buffer for intermediate VPR stores (min 64 bytes)
 */
void conv2d_float32_mxu(
    const float *input, int in_h, int in_w, int in_c,
    const float *weight, int out_c, int kh, int kw,
    const float *bias,
    float *output, int out_h, int out_w,
    int stride_h, int stride_w,
    int pad_top, int pad_left,
    float *scratch);

#ifdef __cplusplus
}
#endif

#endif /* MXU_OPS_H */

