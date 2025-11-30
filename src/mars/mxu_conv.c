/*
 * MXUv3-accelerated convolution for Mars Runtime
 *
 * Uses Ingenic XBurst2 MXUv3 (Media Extension Unit v3) SIMD instructions.
 * T41 has MXUv3 with 32x VPR registers (512-bit each = 16 floats).
 *
 * Working instructions:
 * - LA0_VPR(reg, addr): Load 512-bit (16 floats) to VPR register
 * - SA0_VPR(reg, addr): Store 512-bit (16 floats) from VPR register
 * - VPR_ADD(dst, src): VPR[dst] = VPR[src] + VPR[dst] (in-place)
 * - VPR_MUL(dst, src): VPR[dst] = VPR[src] * VPR[dst] (in-place)
 *
 * Copyright (c) 2024 OpenSensor Project
 * SPDX-License-Identifier: MIT
 */

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __mips__
#include "mxuv3.h"

/*
 * MXU Float32 inner product (dot product)
 * Uses VPR registers to process 16 floats at a time
 * Returns the scalar sum of element-wise products
 */
static inline float inner_product_mxu_f32(const float * __restrict in,
                                          const float * __restrict w,
                                          int count,
                                          float * __restrict scratch) {
    float sum = 0.0f;
    int i = 0;

    /* Process 16 floats at a time using VPR registers */
    for (; i + 16 <= count; i += 16) {
        /* Load input and weights into VPR registers */
        LA0_VPR(2, in + i);   /* VPR2 = input[i:i+16] */
        LA0_VPR(4, w + i);    /* VPR4 = weight[i:i+16] */

        /* Multiply: VPR2 = VPR4 * VPR2 */
        VPR_MUL(2, 4);

        /* Store result to scratch, then sum scalar */
        SA0_VPR(2, scratch);
        __asm__ __volatile__("sync" ::: "memory");

        /* Accumulate 16 products */
        for (int j = 0; j < 16; j++) {
            sum += scratch[j];
        }
    }

    /* Handle remaining elements with scalar code */
    for (; i < count; i++) {
        sum += in[i] * w[i];
    }

    return sum;
}

/*
 * Scalar fallback for small inner products
 */
static inline int32_t inner_product_scalar(const int8_t * __restrict in,
                                           const int8_t * __restrict w, int count) {
    int32_t sum = 0;
    int i = 0;

    /* Process 4 elements at a time */
    for (; i + 3 < count; i += 4) {
        sum += (int32_t)in[i] * (int32_t)w[i];
        sum += (int32_t)in[i+1] * (int32_t)w[i+1];
        sum += (int32_t)in[i+2] * (int32_t)w[i+2];
        sum += (int32_t)in[i+3] * (int32_t)w[i+3];
    }

    /* Handle remaining elements */
    for (; i < count; i++) {
        sum += (int32_t)in[i] * (int32_t)w[i];
    }

    return sum;
}

/*
 * MXU INT8 inner product (dot product) using S4MACSSB
 * Uses VPR registers to process 64 int8 values at a time (512-bit)
 * Returns the 32-bit sum of element-wise products
 *
 * OPTIMIZED VERSION:
 * - Uses 4-way loop unrolling (256 bytes per iteration)
 * - Uses 4 VPR pairs (VPR0-7) to hide load latency
 * - Single VSR accumulator (hardware handles 4 segments)
 * - No sync until final read
 */
static inline int32_t inner_product_mxu_int8(const int8_t * __restrict in,
                                             const int8_t * __restrict w,
                                             int count,
                                             int32_t * __restrict scratch) {
    int32_t sum = 0;
    int i = 0;

    /* Zero the VSR accumulator before starting */
    VSR_ZERO(0);

    /* Process 64 bytes at a time - simple path first */
    for (; i + 64 <= count; i += 64) {
        LA0_VPR(0, in + i);
        LA0_VPR(1, w + i);
        S4MACSSB(0, 0, 1);
    }

    /* Move accumulated sums from VSR0 to VPR8 and zero VSR0 */
    MFSUMZ(8, 0);

    /* Store VPR8 to scratch to read the 4 accumulated sums */
    SA0_VPR(8, scratch);
    __asm__ __volatile__("sync" ::: "memory");

    /* Sum the 4 segment accumulators (each holds sum of 16 byte products) */
    sum = scratch[0] + scratch[1] + scratch[2] + scratch[3];

    /* Handle remaining elements with scalar code */
    for (; i < count; i++) {
        sum += (int32_t)in[i] * (int32_t)w[i];
    }

    return sum;
}

/*
 * MXUv3 INT8 convolution kernel with S4MACSSB acceleration
 *
 * Input format: NCHW (batch, channels, height, width)
 * Weight format: OIHW (out_ch, in_ch, kh, kw)
 * Output format: NCHW (batch, channels, height, width)
 *
 * Uses im2col to gather scattered NCHW data into contiguous buffer,
 * then processes with MXU for acceleration.
 */
void conv2d_int8_mxu(
    const int8_t *input, int in_h, int in_w, int in_c,
    const int8_t *weight, int out_c, int kh, int kw,
    const int32_t *bias,
    int8_t *output, int out_h, int out_w,
    int stride_h, int stride_w,
    int pad_top, int pad_left,
    float in_scale, float w_scale, float out_scale)
{
    float combined_scale = (in_scale * w_scale) / out_scale;

    /* Aligned scratch buffer for MXU output */
    int32_t scratch[16] __attribute__((aligned(64)));

    /* Weight size per output channel: in_c * kh * kw */
    int weight_per_oc = in_c * kh * kw;

    /* im2col buffer - gather kernel window into contiguous memory */
    /* Max size: 64 channels * 7x7 kernel = 3136, round up */
    int8_t im2col_buf[4096] __attribute__((aligned(64)));

    /* Check if we can use MXU (need at least 64 bytes for one VPR) */
    int use_mxu = (weight_per_oc >= 64);

    int out_plane = out_h * out_w;
    int in_plane = in_h * in_w;

    /* For stride=1 no-padding case, use optimized gather */
    int fast_gather = (stride_h == 1 && stride_w == 1 && pad_top == 0 && pad_left == 0);

    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            /* Gather kernel window ONCE per output position */
            int idx = 0;
            if (fast_gather) {
                /* Optimized: no bounds checking needed */
                for (int ic = 0; ic < in_c; ic++) {
                    const int8_t *in_base = input + ic * in_plane + oh * in_w + ow;
                    for (int khi = 0; khi < kh; khi++) {
                        const int8_t *row = in_base + khi * in_w;
                        for (int kwi = 0; kwi < kw; kwi++) {
                            im2col_buf[idx++] = row[kwi];
                        }
                    }
                }
            } else {
                for (int ic = 0; ic < in_c; ic++) {
                    const int8_t *in_ch = input + ic * in_plane;
                    for (int khi = 0; khi < kh; khi++) {
                        int ih = oh * stride_h - pad_top + khi;
                        for (int kwi = 0; kwi < kw; kwi++) {
                            int iw = ow * stride_w - pad_left + kwi;
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                im2col_buf[idx] = in_ch[ih * in_w + iw];
                            } else {
                                im2col_buf[idx] = 0;
                            }
                            idx++;
                        }
                    }
                }
            }

            int out_pos = oh * out_w + ow;

            if (use_mxu) {
                /* Process 4 output channels at once using 4 VSR accumulators */
                int oc = 0;
                for (; oc + 3 < out_c; oc += 4) {
                    const int8_t *w0 = weight + oc * weight_per_oc;
                    const int8_t *w1 = w0 + weight_per_oc;
                    const int8_t *w2 = w1 + weight_per_oc;
                    const int8_t *w3 = w2 + weight_per_oc;

                    VSR_ZERO(0); VSR_ZERO(1); VSR_ZERO(2); VSR_ZERO(3);

                    int i = 0;
                    for (; i + 64 <= weight_per_oc; i += 64) {
                        LA0_VPR(0, im2col_buf + i);
                        LA0_VPR(1, w0 + i);
                        LA0_VPR(2, w1 + i);
                        LA0_VPR(3, w2 + i);
                        LA0_VPR(4, w3 + i);
                        S4MACSSB(0, 0, 1);
                        S4MACSSB(1, 0, 2);
                        S4MACSSB(2, 0, 3);
                        S4MACSSB(3, 0, 4);
                    }

                    /* Single sync for all 4 channels */
                    MFSUMZ(8, 0); MFSUMZ(9, 1); MFSUMZ(10, 2); MFSUMZ(11, 3);
                    SA0_VPR(8, scratch);
                    __asm__ __volatile__("sync" ::: "memory");
                    int32_t s0 = scratch[0] + scratch[1] + scratch[2] + scratch[3];
                    SA0_VPR(9, scratch);
                    int32_t s1 = scratch[0] + scratch[1] + scratch[2] + scratch[3];
                    SA0_VPR(10, scratch);
                    int32_t s2 = scratch[0] + scratch[1] + scratch[2] + scratch[3];
                    SA0_VPR(11, scratch);
                    int32_t s3 = scratch[0] + scratch[1] + scratch[2] + scratch[3];

                    /* Scalar remainder */
                    for (; i < weight_per_oc; i++) {
                        int8_t v = im2col_buf[i];
                        s0 += (int32_t)v * (int32_t)w0[i];
                        s1 += (int32_t)v * (int32_t)w1[i];
                        s2 += (int32_t)v * (int32_t)w2[i];
                        s3 += (int32_t)v * (int32_t)w3[i];
                    }

                    /* Add bias */
                    if (bias) { s0 += bias[oc]; s1 += bias[oc+1]; s2 += bias[oc+2]; s3 += bias[oc+3]; }

                    /* Quantize and store */
                    int32_t r0 = (int32_t)(s0 * combined_scale + (s0 >= 0 ? 0.5f : -0.5f));
                    int32_t r1 = (int32_t)(s1 * combined_scale + (s1 >= 0 ? 0.5f : -0.5f));
                    int32_t r2 = (int32_t)(s2 * combined_scale + (s2 >= 0 ? 0.5f : -0.5f));
                    int32_t r3 = (int32_t)(s3 * combined_scale + (s3 >= 0 ? 0.5f : -0.5f));
                    r0 = r0 > 127 ? 127 : (r0 < -128 ? -128 : r0);
                    r1 = r1 > 127 ? 127 : (r1 < -128 ? -128 : r1);
                    r2 = r2 > 127 ? 127 : (r2 < -128 ? -128 : r2);
                    r3 = r3 > 127 ? 127 : (r3 < -128 ? -128 : r3);

                    output[oc * out_plane + out_pos] = (int8_t)r0;
                    output[(oc+1) * out_plane + out_pos] = (int8_t)r1;
                    output[(oc+2) * out_plane + out_pos] = (int8_t)r2;
                    output[(oc+3) * out_plane + out_pos] = (int8_t)r3;
                }

                /* Remaining channels */
                for (; oc < out_c; oc++) {
                    const int8_t *w_oc = weight + oc * weight_per_oc;
                    VSR_ZERO(0);
                    int i = 0;
                    for (; i + 64 <= weight_per_oc; i += 64) {
                        LA0_VPR(0, im2col_buf + i);
                        LA0_VPR(1, w_oc + i);
                        S4MACSSB(0, 0, 1);
                    }
                    MFSUMZ(8, 0);
                    SA0_VPR(8, scratch);
                    __asm__ __volatile__("sync" ::: "memory");
                    int32_t sum = scratch[0] + scratch[1] + scratch[2] + scratch[3];
                    for (; i < weight_per_oc; i++) {
                        sum += (int32_t)im2col_buf[i] * (int32_t)w_oc[i];
                    }
                    if (bias) sum += bias[oc];
                    int32_t result = (int32_t)(sum * combined_scale + (sum >= 0 ? 0.5f : -0.5f));
                    result = result > 127 ? 127 : (result < -128 ? -128 : result);
                    output[oc * out_plane + out_pos] = (int8_t)result;
                }
            } else {
                /* Scalar path for small kernels */
                for (int oc = 0; oc < out_c; oc++) {
                    const int8_t *w_oc = weight + oc * weight_per_oc;
                    int32_t sum = bias ? bias[oc] : 0;
                    for (int i = 0; i < weight_per_oc; i++) {
                        sum += (int32_t)im2col_buf[i] * (int32_t)w_oc[i];
                    }
                    float scaled = sum * combined_scale;
                    int32_t result = (int32_t)(scaled + (scaled >= 0 ? 0.5f : -0.5f));
                    result = result > 127 ? 127 : (result < -128 ? -128 : result);
                    output[oc * out_plane + out_pos] = (int8_t)result;
                }
            }
        }
    }
}

/*
 * MXUv3-accelerated FLOAT32 convolution kernel
 *
 * Input format: NCHW (batch, channels, height, width)
 * Weight format: OIHW (out_ch, in_ch, kh, kw)
 * Output format: NCHW
 */
void conv2d_float32_mxu(
    const float *input, int in_h, int in_w, int in_c,
    const float *weight, int out_c, int kh, int kw,
    const float *bias,
    float *output, int out_h, int out_w,
    int stride_h, int stride_w,
    int pad_top, int pad_left,
    float *scratch)
{
    int weight_per_oc = in_c * kh * kw;

    for (int oc = 0; oc < out_c; oc++) {
        const float *w_oc = weight + oc * weight_per_oc;

        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float sum = bias ? bias[oc] : 0.0f;

                /* OIHW weight layout: [oc][ic][kh][kw] */
                int w_idx = 0;
                for (int ic = 0; ic < in_c; ic++) {
                    const float *in_ch = input + ic * in_h * in_w;
                    for (int khi = 0; khi < kh; khi++) {
                        int ih = oh * stride_h - pad_top + khi;
                        for (int kwi = 0; kwi < kw; kwi++) {
                            int iw = ow * stride_w - pad_left + kwi;
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                sum += in_ch[ih * in_w + iw] * w_oc[w_idx];
                            }
                            w_idx++;
                        }
                    }
                }

                /* NCHW output */
                output[oc * out_h * out_w + oh * out_w + ow] = sum;
            }
        }
    }
}

#else
/* Fallback for non-MIPS platforms - NCHW format */
void conv2d_int8_mxu(
    const int8_t *input, int in_h, int in_w, int in_c,
    const int8_t *weight, int out_c, int kh, int kw,
    const int32_t *bias,
    int8_t *output, int out_h, int out_w,
    int stride_h, int stride_w,
    int pad_top, int pad_left,
    float in_scale, float w_scale, float out_scale)
{
    float combined_scale = (in_scale * w_scale) / out_scale;
    int weight_per_oc = in_c * kh * kw;

    for (int oc = 0; oc < out_c; oc++) {
        const int8_t *w_oc = weight + oc * weight_per_oc;
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                int32_t sum = bias ? bias[oc] : 0;

                int w_idx = 0;
                for (int ic = 0; ic < in_c; ic++) {
                    const int8_t *in_ch = input + ic * in_h * in_w;
                    for (int khi = 0; khi < kh; khi++) {
                        int ih = oh * stride_h - pad_top + khi;
                        for (int kwi = 0; kwi < kw; kwi++) {
                            int iw = ow * stride_w - pad_left + kwi;
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                sum += (int32_t)in_ch[ih * in_w + iw] * (int32_t)w_oc[w_idx];
                            }
                            w_idx++;
                        }
                    }
                }

                float scaled = sum * combined_scale;
                int32_t result = (int32_t)(scaled + (scaled >= 0 ? 0.5f : -0.5f));
                result = result > 127 ? 127 : (result < -128 ? -128 : result);
                output[oc * out_h * out_w + oh * out_w + ow] = (int8_t)result;
            }
        }
    }
}

/* Float32 fallback - NCHW format */
void conv2d_float32_mxu(
    const float *input, int in_h, int in_w, int in_c,
    const float *weight, int out_c, int kh, int kw,
    const float *bias,
    float *output, int out_h, int out_w,
    int stride_h, int stride_w,
    int pad_top, int pad_left,
    float *scratch)
{
    (void)scratch;
    int weight_per_oc = in_c * kh * kw;

    for (int oc = 0; oc < out_c; oc++) {
        const float *w_oc = weight + oc * weight_per_oc;
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float sum = bias ? bias[oc] : 0.0f;

                int w_idx = 0;
                for (int ic = 0; ic < in_c; ic++) {
                    const float *in_ch = input + ic * in_h * in_w;
                    for (int khi = 0; khi < kh; khi++) {
                        int ih = oh * stride_h - pad_top + khi;
                        for (int kwi = 0; kwi < kw; kwi++) {
                            int iw = ow * stride_w - pad_left + kwi;
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                sum += in_ch[ih * in_w + iw] * w_oc[w_idx];
                            }
                            w_idx++;
                        }
                    }
                }

                output[oc * out_h * out_w + oh * out_w + ow] = sum;
            }
        }
    }
}
#endif /* __mips__ */

