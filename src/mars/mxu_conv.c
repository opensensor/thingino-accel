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

    /* Sum the 4 segment accumulators at positions 0, 4, 8, 12 (stride=4) */
    sum = scratch[0] + scratch[4] + scratch[8] + scratch[12];

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

    int out_plane = out_h * out_w;
    int in_plane = in_h * in_w;

    /*
     * Special fast path for 1x1 convolutions (pointwise)
     * No im2col needed - process 4 spatial positions at once
     */
    if (kh == 1 && kw == 1 && stride_h == 1 && stride_w == 1 && pad_top == 0 && pad_left == 0) {
        int8_t gather_buf[256] __attribute__((aligned(64)));  /* 4 positions × 64 channels max */

        for (int pos = 0; pos < out_plane; pos++) {
            int oh = pos / out_w;
            int ow = pos % out_w;

            /* Gather in_c values for this position from scattered channels */
            for (int ic = 0; ic < in_c; ic++) {
                gather_buf[ic] = input[ic * in_plane + oh * in_w + ow];
            }

            /* Process 4 output channels at a time */
            int oc = 0;
            if (in_c >= 64) {
                for (; oc + 3 < out_c; oc += 4) {
                    const int8_t *w0 = weight + oc * in_c;
                    const int8_t *w1 = w0 + in_c;
                    const int8_t *w2 = w1 + in_c;
                    const int8_t *w3 = w2 + in_c;

                    VSR_ZERO(0); VSR_ZERO(1); VSR_ZERO(2); VSR_ZERO(3);
                    int i = 0;
                    for (; i + 64 <= in_c; i += 64) {
                        LA0_VPR(0, gather_buf + i);
                        LA0_VPR(1, w0 + i);
                        LA0_VPR(2, w1 + i);
                        LA0_VPR(3, w2 + i);
                        LA0_VPR(4, w3 + i);
                        S4MACSSB(0, 0, 1);
                        S4MACSSB(1, 0, 2);
                        S4MACSSB(2, 0, 3);
                        S4MACSSB(3, 0, 4);
                    }
                    /* Extract accumulated sums from VSR to VPR and zero VSR */
                    MFSUMZ(8, 0); MFSUMZ(9, 1); MFSUMZ(10, 2); MFSUMZ(11, 3);

                    /* Read each VPR with sync before reading
                     * S4MACSSB places segment sums at positions 0, 4, 8, 12 */
                    SA0_VPR(8, scratch);
                    __asm__ __volatile__("sync" ::: "memory");
                    int32_t s0 = scratch[0] + scratch[4] + scratch[8] + scratch[12];

                    SA0_VPR(9, scratch);
                    __asm__ __volatile__("sync" ::: "memory");
                    int32_t s1 = scratch[0] + scratch[4] + scratch[8] + scratch[12];

                    SA0_VPR(10, scratch);
                    __asm__ __volatile__("sync" ::: "memory");
                    int32_t s2 = scratch[0] + scratch[4] + scratch[8] + scratch[12];

                    SA0_VPR(11, scratch);
                    __asm__ __volatile__("sync" ::: "memory");
                    int32_t s3 = scratch[0] + scratch[4] + scratch[8] + scratch[12];

                    for (; i < in_c; i++) {
                        int8_t v = gather_buf[i];
                        s0 += (int32_t)v * (int32_t)w0[i];
                        s1 += (int32_t)v * (int32_t)w1[i];
                        s2 += (int32_t)v * (int32_t)w2[i];
                        s3 += (int32_t)v * (int32_t)w3[i];
                    }
                    if (bias) { s0 += bias[oc]; s1 += bias[oc+1]; s2 += bias[oc+2]; s3 += bias[oc+3]; }

                    int32_t r0 = (int32_t)(s0 * combined_scale + (s0 >= 0 ? 0.5f : -0.5f));
                    int32_t r1 = (int32_t)(s1 * combined_scale + (s1 >= 0 ? 0.5f : -0.5f));
                    int32_t r2 = (int32_t)(s2 * combined_scale + (s2 >= 0 ? 0.5f : -0.5f));
                    int32_t r3 = (int32_t)(s3 * combined_scale + (s3 >= 0 ? 0.5f : -0.5f));
                    r0 = r0 > 127 ? 127 : (r0 < -128 ? -128 : r0);
                    r1 = r1 > 127 ? 127 : (r1 < -128 ? -128 : r1);
                    r2 = r2 > 127 ? 127 : (r2 < -128 ? -128 : r2);
                    r3 = r3 > 127 ? 127 : (r3 < -128 ? -128 : r3);

                    output[oc * out_plane + pos] = (int8_t)r0;
                    output[(oc+1) * out_plane + pos] = (int8_t)r1;
                    output[(oc+2) * out_plane + pos] = (int8_t)r2;
                    output[(oc+3) * out_plane + pos] = (int8_t)r3;
                }
            }

            /* Remaining channels - scalar */
            for (; oc < out_c; oc++) {
                const int8_t *w_oc = weight + oc * in_c;
                int32_t sum = bias ? bias[oc] : 0;
                for (int i = 0; i < in_c; i++) {
                    sum += (int32_t)gather_buf[i] * (int32_t)w_oc[i];
                }
                float scaled = sum * combined_scale;
                int32_t result = (int32_t)(scaled + (scaled >= 0 ? 0.5f : -0.5f));
                result = result > 127 ? 127 : (result < -128 ? -128 : result);
                output[oc * out_plane + pos] = (int8_t)result;
            }
        }
        return;
    }

    /* im2col buffer for larger kernels */
    int8_t im2col_buf[4096] __attribute__((aligned(64)));

    /* Check if we can use MXU (need at least 64 bytes for one VPR) */
    int use_mxu = (weight_per_oc >= 64);

    /* For stride=1 no-padding case, use optimized gather */
    int fast_gather = (stride_h == 1 && stride_w == 1 && pad_top == 0 && pad_left == 0);

    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            /* Gather kernel window ONCE per output position */
            int8_t *dst = im2col_buf;
            if (fast_gather && kw == 3) {
                /* Optimized 3x3: copy 3 bytes at a time */
                for (int ic = 0; ic < in_c; ic++) {
                    const int8_t *in_base = input + ic * in_plane + oh * in_w + ow;
                    for (int khi = 0; khi < kh; khi++) {
                        const int8_t *row = in_base + khi * in_w;
                        dst[0] = row[0]; dst[1] = row[1]; dst[2] = row[2];
                        dst += 3;
                    }
                }
            } else if (fast_gather) {
                for (int ic = 0; ic < in_c; ic++) {
                    const int8_t *in_base = input + ic * in_plane + oh * in_w + ow;
                    for (int khi = 0; khi < kh; khi++) {
                        const int8_t *row = in_base + khi * in_w;
                        for (int kwi = 0; kwi < kw; kwi++) {
                            *dst++ = row[kwi];
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
                                *dst++ = in_ch[ih * in_w + iw];
                            } else {
                                *dst++ = 0;
                            }
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

                    /* Extract accumulated sums from VSR to VPR and zero VSR */
                    MFSUMZ(8, 0); MFSUMZ(9, 1); MFSUMZ(10, 2); MFSUMZ(11, 3);

                    /* Read each VPR with sync before reading
                     * S4MACSSB places segment sums at positions 0, 4, 8, 12 */
                    SA0_VPR(8, scratch);
                    __asm__ __volatile__("sync" ::: "memory");
                    int32_t s0 = scratch[0] + scratch[4] + scratch[8] + scratch[12];

                    SA0_VPR(9, scratch);
                    __asm__ __volatile__("sync" ::: "memory");
                    int32_t s1 = scratch[0] + scratch[4] + scratch[8] + scratch[12];

                    SA0_VPR(10, scratch);
                    __asm__ __volatile__("sync" ::: "memory");
                    int32_t s2 = scratch[0] + scratch[4] + scratch[8] + scratch[12];

                    SA0_VPR(11, scratch);
                    __asm__ __volatile__("sync" ::: "memory");
                    int32_t s3 = scratch[0] + scratch[4] + scratch[8] + scratch[12];

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
                    /* S4MACSSB places segment sums at positions 0, 4, 8, 12 */
                    int32_t sum = scratch[0] + scratch[4] + scratch[8] + scratch[12];
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

/*
 * MXUv3 INT8 convolution kernel for NHWC format with fused activation
 *
 * Input format: NHWC (batch, height, width, channels) - channels contiguous
 * Weight format: OHWI (out_ch, kh, kw, in_ch) - input channels contiguous per kernel position
 * Output format: NHWC (batch, height, width, channels)
 *
 * Activation is applied BEFORE quantization to INT8, preserving magnitude info.
 * activation: 0=none, 1=relu, 3=leaky_relu (0.1 slope)
 *
 * NHWC is much faster because:
 * - Channels are contiguous at each spatial position
 * - Gather is just copying kh*kw rows of in_c bytes each
 * - No scattered memory access across channel planes
 */
void conv2d_int8_nhwc_mxu_act(
    const int8_t *input, int in_h, int in_w, int in_c,
    const int8_t *weight, int out_c, int kh, int kw,
    const int32_t *bias,
    int8_t *output, int out_h, int out_w,
    int stride_h, int stride_w,
    int pad_top, int pad_left,
    float in_scale, float w_scale, float out_scale,
    int activation)
{
    static int debug_count = 0;
    if (debug_count < 3) {
        fprintf(stderr, "    [DEBUG] conv act=%d (3=leaky) combined_scale=%f\n",
                activation, (in_scale * w_scale) / out_scale);
        debug_count++;
    }

    float combined_scale = (in_scale * w_scale) / out_scale;

    /* For LeakyReLU, compute scale for negative values: scale * 0.1 */
    float neg_scale = combined_scale * 0.1f;

    /* Aligned scratch buffer for MXU output */
    int32_t scratch[16] __attribute__((aligned(64)));

    /* Weight size per output channel: kh * kw * in_c (OHWI layout) */
    int weight_per_oc = kh * kw * in_c;

    /* Input row stride in bytes */
    int in_row_stride = in_w * in_c;

    /* im2col buffer - gather kernel window */
    int8_t im2col_buf[4096] __attribute__((aligned(64)));

    /* Check if we can use MXU (need at least 64 bytes for one VPR) */
    int use_mxu = (weight_per_oc >= 64);

    /* Progress tracking for debugging */
    static int first_call = 1;
    (void)first_call; /* Suppress unused warning */

    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {

            /* Gather kernel window - NHWC makes this fast!
             * For each kernel row, copy in_c contiguous bytes
             */
            int8_t *dst = im2col_buf;
            int base_ih = oh * stride_h - pad_top;
            int base_iw = ow * stride_w - pad_left;

            for (int khi = 0; khi < kh; khi++) {
                int ih = base_ih + khi;
                for (int kwi = 0; kwi < kw; kwi++) {
                    int iw = base_iw + kwi;
                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                        /* Copy in_c contiguous bytes */
                        const int8_t *src = input + ih * in_row_stride + iw * in_c;
                        memcpy(dst, src, in_c);
                    } else {
                        /* Zero padding */
                        memset(dst, 0, in_c);
                    }
                    dst += in_c;
                }
            }

            /* Output position in NHWC format */
            int8_t *out_pos = output + (oh * out_w + ow) * out_c;

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

                    /* Extract accumulated sums from VSR to VPR and zero VSR */
                    MFSUMZ(8, 0); MFSUMZ(9, 1); MFSUMZ(10, 2); MFSUMZ(11, 3);

                    /* Read each VPR with sync before reading
                     * S4MACSSB places 4 segment sums at positions 0, 4, 8, 12 (stride=4)
                     * NOT positions 0, 1, 2, 3 as originally assumed!
                     */
                    SA0_VPR(8, scratch);
                    __asm__ __volatile__("sync" ::: "memory");
                    int32_t s0 = scratch[0] + scratch[4] + scratch[8] + scratch[12];

                    SA0_VPR(9, scratch);
                    __asm__ __volatile__("sync" ::: "memory");
                    int32_t s1 = scratch[0] + scratch[4] + scratch[8] + scratch[12];

                    SA0_VPR(10, scratch);
                    __asm__ __volatile__("sync" ::: "memory");
                    int32_t s2 = scratch[0] + scratch[4] + scratch[8] + scratch[12];

                    SA0_VPR(11, scratch);
                    __asm__ __volatile__("sync" ::: "memory");
                    int32_t s3 = scratch[0] + scratch[4] + scratch[8] + scratch[12];

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

                    /* Apply activation and quantize - NHWC output: channels contiguous */
                    /* LeakyReLU applied BEFORE quantization to preserve magnitude */
                    float sc0 = (s0 >= 0 || activation != 3) ? combined_scale : neg_scale;
                    float sc1 = (s1 >= 0 || activation != 3) ? combined_scale : neg_scale;
                    float sc2 = (s2 >= 0 || activation != 3) ? combined_scale : neg_scale;
                    float sc3 = (s3 >= 0 || activation != 3) ? combined_scale : neg_scale;

                    int32_t r0 = (int32_t)(s0 * sc0 + (s0 >= 0 ? 0.5f : -0.5f));
                    int32_t r1 = (int32_t)(s1 * sc1 + (s1 >= 0 ? 0.5f : -0.5f));
                    int32_t r2 = (int32_t)(s2 * sc2 + (s2 >= 0 ? 0.5f : -0.5f));
                    int32_t r3 = (int32_t)(s3 * sc3 + (s3 >= 0 ? 0.5f : -0.5f));

                    /* ReLU: clamp negatives to 0 */
                    if (activation == 1) {
                        if (r0 < 0) r0 = 0;
                        if (r1 < 0) r1 = 0;
                        if (r2 < 0) r2 = 0;
                        if (r3 < 0) r3 = 0;
                    }

                    r0 = r0 > 127 ? 127 : (r0 < -128 ? -128 : r0);
                    r1 = r1 > 127 ? 127 : (r1 < -128 ? -128 : r1);
                    r2 = r2 > 127 ? 127 : (r2 < -128 ? -128 : r2);
                    r3 = r3 > 127 ? 127 : (r3 < -128 ? -128 : r3);

                    out_pos[oc] = (int8_t)r0;
                    out_pos[oc+1] = (int8_t)r1;
                    out_pos[oc+2] = (int8_t)r2;
                    out_pos[oc+3] = (int8_t)r3;
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
                    /* S4MACSSB places segment sums at positions 0, 4, 8, 12 */
                    int32_t sum = scratch[0] + scratch[4] + scratch[8] + scratch[12];
                    for (; i < weight_per_oc; i++) {
                        sum += (int32_t)im2col_buf[i] * (int32_t)w_oc[i];
                    }
                    if (bias) sum += bias[oc];
                    float sc = (sum >= 0 || activation != 3) ? combined_scale : neg_scale;
                    int32_t result = (int32_t)(sum * sc + (sum >= 0 ? 0.5f : -0.5f));
                    if (activation == 1 && result < 0) result = 0;
                    result = result > 127 ? 127 : (result < -128 ? -128 : result);
                    out_pos[oc] = (int8_t)result;
                }
            } else {
                /* Scalar path for small kernels */
                for (int oc = 0; oc < out_c; oc++) {
                    const int8_t *w_oc = weight + oc * weight_per_oc;
                    int32_t sum = bias ? bias[oc] : 0;
                    for (int i = 0; i < weight_per_oc; i++) {
                        sum += (int32_t)im2col_buf[i] * (int32_t)w_oc[i];
                    }
                    float sc = (sum >= 0 || activation != 3) ? combined_scale : neg_scale;
                    int32_t result = (int32_t)(sum * sc + (sum >= 0 ? 0.5f : -0.5f));
                    if (activation == 1 && result < 0) result = 0;
                    result = result > 127 ? 127 : (result < -128 ? -128 : result);
                    out_pos[oc] = (int8_t)result;
                }
            }
        }
    }
    first_call = 0;
}

/*
 * Optimized NHWC Conv2D with weight-stationary processing
 * Load each 64-byte weight chunk ONCE, then apply to all tile positions
 * This reduces weight memory bandwidth by TILE_W factor
 */
#define TILE_W 4  /* Process 4 output columns - balances reuse vs register pressure */

void conv2d_int8_nhwc_mxu_tiled(
    const int8_t *input, int in_h, int in_w, int in_c,
    const int8_t *weight, int out_c, int kh, int kw,
    const int32_t *bias,
    int8_t *output, int out_h, int out_w,
    int stride_h, int stride_w,
    int pad_top, int pad_left,
    float in_scale, float w_scale, float out_scale,
    int activation)
{
    float combined_scale = (in_scale * w_scale) / out_scale;
    float neg_scale = combined_scale * 0.1f;

    int32_t scratch[16] __attribute__((aligned(64)));
    int weight_per_oc = kh * kw * in_c;
    int in_row_stride = in_w * in_c;

    /* im2col buffers for tile positions - each is weight_per_oc bytes */
    int8_t im2col_buf[TILE_W * 4096] __attribute__((aligned(64)));

    /* Accumulators for tile positions (4 channels × TILE_W positions) */
    int32_t tile_sums[TILE_W][4];

    int use_mxu = (weight_per_oc >= 64);

    for (int oh = 0; oh < out_h; oh++) {
        for (int ow_base = 0; ow_base < out_w; ow_base += TILE_W) {
            int tile_w = (ow_base + TILE_W <= out_w) ? TILE_W : (out_w - ow_base);

            /* Gather im2col for all tile positions at once */
            for (int t = 0; t < tile_w; t++) {
                int ow = ow_base + t;
                int8_t *dst = im2col_buf + t * weight_per_oc;
                int base_ih = oh * stride_h - pad_top;
                int base_iw = ow * stride_w - pad_left;

                for (int khi = 0; khi < kh; khi++) {
                    int ih = base_ih + khi;
                    for (int kwi = 0; kwi < kw; kwi++) {
                        int iw = base_iw + kwi;
                        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            memcpy(dst, input + ih * in_row_stride + iw * in_c, in_c);
                        } else {
                            memset(dst, 0, in_c);
                        }
                        dst += in_c;
                    }
                }
            }

            /* Process 4 output channels at a time */
            for (int oc = 0; oc < out_c; oc += 4) {
                int oc_count = (oc + 4 <= out_c) ? 4 : (out_c - oc);
                const int8_t *w0 = weight + oc * weight_per_oc;
                const int8_t *w1 = w0 + weight_per_oc;
                const int8_t *w2 = w1 + weight_per_oc;
                const int8_t *w3 = w2 + weight_per_oc;

                /* Initialize accumulators for all tile positions */
                for (int t = 0; t < tile_w; t++) {
                    tile_sums[t][0] = tile_sums[t][1] = tile_sums[t][2] = tile_sums[t][3] = 0;
                }

                if (use_mxu && oc_count == 4) {
                    /* WEIGHT-STATIONARY: Load each weight chunk once, apply to all tiles */
                    int i = 0;
                    for (; i + 64 <= weight_per_oc; i += 64) {
                        /* Load weights ONCE per 64-byte chunk */
                        LA0_VPR(1, w0 + i);
                        LA0_VPR(2, w1 + i);
                        LA0_VPR(3, w2 + i);
                        LA0_VPR(4, w3 + i);

                        /* Apply to each tile position, extracting results each time */
                        for (int t = 0; t < tile_w; t++) {
                            int8_t *im2col = im2col_buf + t * weight_per_oc;

                            VSR_ZERO(0); VSR_ZERO(1); VSR_ZERO(2); VSR_ZERO(3);
                            LA0_VPR(0, im2col + i);
                            S4MACSSB(0, 0, 1);
                            S4MACSSB(1, 0, 2);
                            S4MACSSB(2, 0, 3);
                            S4MACSSB(3, 0, 4);

                            MFSUMZ(8, 0); MFSUMZ(9, 1); MFSUMZ(10, 2); MFSUMZ(11, 3);

                            SA0_VPR(8, scratch);
                            __asm__ __volatile__("sync" ::: "memory");
                            tile_sums[t][0] += scratch[0] + scratch[4] + scratch[8] + scratch[12];

                            SA0_VPR(9, scratch);
                            __asm__ __volatile__("sync" ::: "memory");
                            tile_sums[t][1] += scratch[0] + scratch[4] + scratch[8] + scratch[12];

                            SA0_VPR(10, scratch);
                            __asm__ __volatile__("sync" ::: "memory");
                            tile_sums[t][2] += scratch[0] + scratch[4] + scratch[8] + scratch[12];

                            SA0_VPR(11, scratch);
                            __asm__ __volatile__("sync" ::: "memory");
                            tile_sums[t][3] += scratch[0] + scratch[4] + scratch[8] + scratch[12];
                        }
                    }

                    /* Scalar remainder for non-64-byte-aligned portion */
                    for (; i < weight_per_oc; i++) {
                        int8_t wv0 = w0[i], wv1 = w1[i], wv2 = w2[i], wv3 = w3[i];
                        for (int t = 0; t < tile_w; t++) {
                            int8_t v = im2col_buf[t * weight_per_oc + i];
                            tile_sums[t][0] += (int32_t)v * (int32_t)wv0;
                            tile_sums[t][1] += (int32_t)v * (int32_t)wv1;
                            tile_sums[t][2] += (int32_t)v * (int32_t)wv2;
                            tile_sums[t][3] += (int32_t)v * (int32_t)wv3;
                        }
                    }
                } else {
                    /* Scalar fallback */
                    for (int c = 0; c < oc_count; c++) {
                        const int8_t *w_oc = weight + (oc + c) * weight_per_oc;
                        for (int t = 0; t < tile_w; t++) {
                            int8_t *im2col = im2col_buf + t * weight_per_oc;
                            for (int i = 0; i < weight_per_oc; i++) {
                                tile_sums[t][c] += (int32_t)im2col[i] * (int32_t)w_oc[i];
                            }
                        }
                    }
                }

                /* Quantize and store all tile results */
                for (int t = 0; t < tile_w; t++) {
                    int ow = ow_base + t;
                    int8_t *out_pos = output + (oh * out_w + ow) * out_c;

                    for (int c = 0; c < oc_count; c++) {
                        int32_t sum = tile_sums[t][c];
                        if (bias) sum += bias[oc + c];

                        float sc = (sum >= 0 || activation != 3) ? combined_scale : neg_scale;
                        int32_t result = (int32_t)(sum * sc + (sum >= 0 ? 0.5f : -0.5f));
                        if (activation == 1 && result < 0) result = 0;
                        result = result > 127 ? 127 : (result < -128 ? -128 : result);
                        out_pos[oc + c] = (int8_t)result;
                    }
                }
            }
        }
    }
}

/* Wrapper without activation for backward compatibility */
void conv2d_int8_nhwc_mxu(
    const int8_t *input, int in_h, int in_w, int in_c,
    const int8_t *weight, int out_c, int kh, int kw,
    const int32_t *bias,
    int8_t *output, int out_h, int out_w,
    int stride_h, int stride_w,
    int pad_top, int pad_left,
    float in_scale, float w_scale, float out_scale)
{
    conv2d_int8_nhwc_mxu_act(input, in_h, in_w, in_c, weight, out_c, kh, kw,
                              bias, output, out_h, out_w, stride_h, stride_w,
                              pad_top, pad_left, in_scale, w_scale, out_scale, 0);
}

/*
 * Float32 convolution for NHWC format with fused activation
 * MXU-OPTIMIZED VERSION
 *
 * Input format: NHWC (batch, height, width, channels)
 * Weight format: OHWI (out_ch, kh, kw, in_ch) - compatible with NHWC gather
 * Output format: NHWC (batch, height, width, channels)
 *
 * activation: 0=none, 1=relu, 3=leaky_relu (0.1 slope)
 * dilation_h, dilation_w: dilation factors (1 = no dilation)
 *
 * Optimization strategy:
 * 1. Gather kernel window once per output position into im2col buffer
 * 2. Use MXU VPR for vectorized multiply-accumulate (16 floats/iteration)
 * 3. Process multiple output channels per gathered window
 */
void conv2d_float32_nhwc(
    const float *input, int in_h, int in_w, int in_c,
    const float *weight, int out_c, int kh, int kw,
    const float *bias,
    float *output, int out_h, int out_w,
    int stride_h, int stride_w,
    int pad_top, int pad_left,
    int dilation_h, int dilation_w,
    int activation)
{
    /* Weight layout: OHWI - [out_c][kh][kw][in_c] */
    int weight_per_oc = kh * kw * in_c;
    int in_row_stride = in_w * in_c;

    /* Allocate im2col buffer for kernel window (64-byte aligned for MXU) */
    int im2col_size = ((weight_per_oc + 15) & ~15) * sizeof(float);
    float *im2col_buf = (float *)aligned_alloc(64, im2col_size);
    if (!im2col_buf) {
        /* Fallback to non-aligned allocation */
        im2col_buf = (float *)malloc(im2col_size);
    }

    /* Scratch buffer for MXU horizontal sum (64-byte aligned) */
    float scratch[16] __attribute__((aligned(64)));

    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            float *out_pos = output + (oh * out_w + ow) * out_c;

            /* Gather kernel window into im2col_buf ONCE per output position */
            float *dst = im2col_buf;
            for (int khi = 0; khi < kh; khi++) {
                int ih = oh * stride_h - pad_top + khi * dilation_h;
                for (int kwi = 0; kwi < kw; kwi++) {
                    int iw = ow * stride_w - pad_left + kwi * dilation_w;
                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                        const float *in_pos = input + ih * in_row_stride + iw * in_c;
                        memcpy(dst, in_pos, in_c * sizeof(float));
                    } else {
                        memset(dst, 0, in_c * sizeof(float));
                    }
                    dst += in_c;
                }
            }

            /* Now compute all output channels using the gathered window */
            for (int oc = 0; oc < out_c; oc++) {
                const float *w_oc = weight + oc * weight_per_oc;
                float sum = bias ? bias[oc] : 0.0f;

                /* MXU-accelerated dot product: process 16 floats at a time */
                int i = 0;
                for (; i + 16 <= weight_per_oc; i += 16) {
                    LA0_VPR(2, im2col_buf + i);  /* VPR2 = input patch */
                    LA0_VPR(4, w_oc + i);        /* VPR4 = weights */
                    VPR_MUL(2, 4);               /* VPR2 = VPR2 * VPR4 */
                    SA0_VPR(2, scratch);
                    __asm__ __volatile__("sync" ::: "memory");

                    /* Horizontal sum of 16 products */
                    sum += scratch[0] + scratch[1] + scratch[2] + scratch[3] +
                           scratch[4] + scratch[5] + scratch[6] + scratch[7] +
                           scratch[8] + scratch[9] + scratch[10] + scratch[11] +
                           scratch[12] + scratch[13] + scratch[14] + scratch[15];
                }

                /* Scalar tail for remaining elements */
                for (; i < weight_per_oc; i++) {
                    sum += im2col_buf[i] * w_oc[i];
                }

                /* Apply activation */
                if (activation == 1 && sum < 0) {
                    sum = 0.0f;  /* ReLU */
                } else if (activation == 3 && sum < 0) {
                    sum *= 0.1f;  /* LeakyReLU */
                }

                out_pos[oc] = sum;
            }
        }
    }

    free(im2col_buf);
}

/*
 * ORAM-accelerated Float32 NHWC convolution
 *
 * Stages weights in ORAM for 5-20x faster access.
 * Use when weight_size <= oram_size.
 *
 * Benchmark shows: ORAM is 7.6x faster for reads, 20x faster for writes.
 */
void conv2d_float32_nhwc_oram(
    const float *input, int in_h, int in_w, int in_c,
    const float *weight, int out_c, int kh, int kw,
    const float *bias,
    float *output, int out_h, int out_w,
    int stride_h, int stride_w,
    int pad_top, int pad_left,
    int dilation_h, int dilation_w,
    int activation,
    void *oram_base, uint32_t oram_size)
{
    /* Weight layout: OHWI - [out_c][kh][kw][in_c] */
    int weight_per_oc = kh * kw * in_c;
    uint32_t weight_size = (uint32_t)(out_c * weight_per_oc * sizeof(float));
    int in_row_stride = in_w * in_c;

    /* Check if weights fit in ORAM */
    int use_oram = (oram_base != NULL && weight_size <= oram_size);
    volatile float *oram_weights = NULL;

    if (use_oram) {
        /* Stage weights to ORAM for faster access */
        oram_weights = (volatile float *)oram_base;
        memcpy((void *)oram_weights, weight, weight_size);
        __sync_synchronize();
    }

    /* Allocate im2col buffer for kernel window (64-byte aligned for MXU) */
    int im2col_size = ((weight_per_oc + 15) & ~15) * sizeof(float);
    float *im2col_buf = (float *)aligned_alloc(64, im2col_size);
    if (!im2col_buf) {
        im2col_buf = (float *)malloc(im2col_size);
    }

    /* Scratch buffer for MXU horizontal sum (64-byte aligned) */
    float scratch[16] __attribute__((aligned(64)));

    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            float *out_pos = output + (oh * out_w + ow) * out_c;

            /* Gather kernel window into im2col_buf ONCE per output position */
            float *dst = im2col_buf;
            for (int khi = 0; khi < kh; khi++) {
                int ih = oh * stride_h - pad_top + khi * dilation_h;
                for (int kwi = 0; kwi < kw; kwi++) {
                    int iw = ow * stride_w - pad_left + kwi * dilation_w;
                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                        const float *in_pos = input + ih * in_row_stride + iw * in_c;
                        memcpy(dst, in_pos, in_c * sizeof(float));
                    } else {
                        memset(dst, 0, in_c * sizeof(float));
                    }
                    dst += in_c;
                }
            }

            /* Compute all output channels using the gathered window */
            for (int oc = 0; oc < out_c; oc++) {
                /* Use ORAM weights if staged, otherwise DDR weights */
                const float *w_oc = use_oram
                    ? (const float *)(oram_weights + oc * weight_per_oc)
                    : (weight + oc * weight_per_oc);
                float sum = bias ? bias[oc] : 0.0f;

                /* MXU-accelerated dot product: process 16 floats at a time */
                int i = 0;
                for (; i + 16 <= weight_per_oc; i += 16) {
                    LA0_VPR(2, im2col_buf + i);  /* VPR2 = input patch */
                    LA0_VPR(4, w_oc + i);        /* VPR4 = weights (from ORAM!) */
                    VPR_MUL(2, 4);               /* VPR2 = VPR2 * VPR4 */
                    SA0_VPR(2, scratch);
                    __asm__ __volatile__("sync" ::: "memory");

                    /* Horizontal sum of 16 products */
                    sum += scratch[0] + scratch[1] + scratch[2] + scratch[3] +
                           scratch[4] + scratch[5] + scratch[6] + scratch[7] +
                           scratch[8] + scratch[9] + scratch[10] + scratch[11] +
                           scratch[12] + scratch[13] + scratch[14] + scratch[15];
                }

                /* Scalar tail for remaining elements */
                for (; i < weight_per_oc; i++) {
                    sum += im2col_buf[i] * w_oc[i];
                }

                /* Apply activation */
                if (activation == 1 && sum < 0) {
                    sum = 0.0f;  /* ReLU */
                } else if (activation == 3 && sum < 0) {
                    sum *= 0.1f;  /* LeakyReLU */
                }

                out_pos[oc] = sum;
            }
        }
    }

    free(im2col_buf);
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

/*
 * Float32 convolution for NHWC format with fused activation
 * SCALAR FALLBACK for non-MIPS platforms
 */
void conv2d_float32_nhwc(
    const float *input, int in_h, int in_w, int in_c,
    const float *weight, int out_c, int kh, int kw,
    const float *bias,
    float *output, int out_h, int out_w,
    int stride_h, int stride_w,
    int pad_top, int pad_left,
    int dilation_h, int dilation_w,
    int activation)
{
    int weight_per_oc = kh * kw * in_c;
    int in_row_stride = in_w * in_c;

    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            float *out_pos = output + (oh * out_w + ow) * out_c;

            for (int oc = 0; oc < out_c; oc++) {
                const float *w_oc = weight + oc * weight_per_oc;
                float sum = bias ? bias[oc] : 0.0f;

                int w_idx = 0;
                for (int khi = 0; khi < kh; khi++) {
                    int ih = oh * stride_h - pad_top + khi * dilation_h;
                    for (int kwi = 0; kwi < kw; kwi++) {
                        int iw = ow * stride_w - pad_left + kwi * dilation_w;
                        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            const float *in_pos = input + ih * in_row_stride + iw * in_c;
                            for (int ic = 0; ic < in_c; ic++) {
                                sum += in_pos[ic] * w_oc[w_idx++];
                            }
                        } else {
                            w_idx += in_c;
                        }
                    }
                }

                if (activation == 1 && sum < 0) sum = 0.0f;
                else if (activation == 3 && sum < 0) sum *= 0.1f;
                out_pos[oc] = sum;
            }
        }
    }
}

/* ORAM fallback for non-MIPS platforms - just calls regular version */
void conv2d_float32_nhwc_oram(
    const float *input, int in_h, int in_w, int in_c,
    const float *weight, int out_c, int kh, int kw,
    const float *bias,
    float *output, int out_h, int out_w,
    int stride_h, int stride_w,
    int pad_top, int pad_left,
    int dilation_h, int dilation_w,
    int activation,
    void *oram_base, uint32_t oram_size)
{
    (void)oram_base;
    (void)oram_size;
    conv2d_float32_nhwc(input, in_h, in_w, in_c, weight, out_c, kh, kw,
                        bias, output, out_h, out_w, stride_h, stride_w,
                        pad_top, pad_left, dilation_h, dilation_w, activation);
}

/* NHWC fallback for non-MIPS platforms with activation */
void conv2d_int8_nhwc_mxu_act(
    const int8_t *input, int in_h, int in_w, int in_c,
    const int8_t *weight, int out_c, int kh, int kw,
    const int32_t *bias,
    int8_t *output, int out_h, int out_w,
    int stride_h, int stride_w,
    int pad_top, int pad_left,
    float in_scale, float w_scale, float out_scale,
    int activation)
{
    float combined_scale = (in_scale * w_scale) / out_scale;
    float neg_scale = combined_scale * 0.1f;
    int weight_per_oc = kh * kw * in_c;  /* OHWI layout */
    int in_row_stride = in_w * in_c;

    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            int8_t *out_pos = output + (oh * out_w + ow) * out_c;

            for (int oc = 0; oc < out_c; oc++) {
                const int8_t *w_oc = weight + oc * weight_per_oc;
                int32_t sum = bias ? bias[oc] : 0;

                int w_idx = 0;
                for (int khi = 0; khi < kh; khi++) {
                    int ih = oh * stride_h - pad_top + khi;
                    for (int kwi = 0; kwi < kw; kwi++) {
                        int iw = ow * stride_w - pad_left + kwi;
                        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            const int8_t *in_pos = input + ih * in_row_stride + iw * in_c;
                            for (int ic = 0; ic < in_c; ic++) {
                                sum += (int32_t)in_pos[ic] * (int32_t)w_oc[w_idx++];
                            }
                        } else {
                            w_idx += in_c;  /* Skip zero-padded region */
                        }
                    }
                }

                float sc = (sum >= 0 || activation != 3) ? combined_scale : neg_scale;
                int32_t result = (int32_t)(sum * sc + (sum >= 0 ? 0.5f : -0.5f));
                if (activation == 1 && result < 0) result = 0;
                result = result > 127 ? 127 : (result < -128 ? -128 : result);
                out_pos[oc] = (int8_t)result;
            }
        }
    }
}

/* Wrapper without activation for backward compatibility */
void conv2d_int8_nhwc_mxu(
    const int8_t *input, int in_h, int in_w, int in_c,
    const int8_t *weight, int out_c, int kh, int kw,
    const int32_t *bias,
    int8_t *output, int out_h, int out_w,
    int stride_h, int stride_w,
    int pad_top, int pad_left,
    float in_scale, float w_scale, float out_scale)
{
    conv2d_int8_nhwc_mxu_act(input, in_h, in_w, in_c, weight, out_c, kh, kw,
                              bias, output, out_h, out_w, stride_h, stride_w,
                              pad_top, pad_left, in_scale, w_scale, out_scale, 0);
}

/* Tiled version fallback for non-MIPS - just call the regular version */
void conv2d_int8_nhwc_mxu_tiled(
    const int8_t *input, int in_h, int in_w, int in_c,
    const int8_t *weight, int out_c, int kh, int kw,
    const int32_t *bias,
    int8_t *output, int out_h, int out_w,
    int stride_h, int stride_w,
    int pad_top, int pad_left,
    float in_scale, float w_scale, float out_scale,
    int activation)
{
    conv2d_int8_nhwc_mxu_act(input, in_h, in_w, in_c, weight, out_c, kh, kw,
                              bias, output, out_h, out_w, stride_h, stride_w,
                              pad_top, pad_left, in_scale, w_scale, out_scale, activation);
}
#endif /* __mips__ */

