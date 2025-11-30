/*
 * Mars Conv2D Benchmark
 *
 * Benchmarks 3x3 convolution using MXU inner product vs scalar.
 * Tests various input sizes typical for neural network inference.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mars_nn_hw.h"
#include "mars_math.h"

/* Test configurations: height, width, input_channels, output_channels
 * Sized to fit in ~640KB ORAM with input + weights + output + patch buffer
 */
typedef struct {
    int h, w, ic, oc;
    const char *name;
} conv_config_t;

static const conv_config_t CONFIGS[] = {
    { 10, 10,   8,   8, "10x10x8->8" },     /* tiny: ~4KB total */
    { 14, 14,  16,  16, "14x14x16->16" },   /* small: ~26KB total */
    { 20, 20,  16,  32, "20x20x16->32" },   /* medium: ~82KB total */
    { 20, 20,  32,  32, "20x20x32->32" },   /* medium+: ~133KB total */
};
#define NUM_CONFIGS (sizeof(CONFIGS) / sizeof(CONFIGS[0]))

#define KERNEL_SIZE 3
#define WARMUP_ITERS 2
#define BENCH_ITERS 3

/* Scalar 3x3 convolution (one output channel, one spatial location) */
static float scalar_conv3x3(const float *input, const float *weight,
                            int h, int w, int ic, int oh, int ow) {
    float sum = 0.0f;
    for (int kh = 0; kh < 3; kh++) {
        for (int kw = 0; kw < 3; kw++) {
            int ih = oh + kh;
            int iw = ow + kw;
            if (ih < h && iw < w) {
                for (int c = 0; c < ic; c++) {
                    int in_idx = (ih * w + iw) * ic + c;
                    int wt_idx = (kh * 3 + kw) * ic + c;
                    sum += input[in_idx] * weight[wt_idx];
                }
            }
        }
    }
    return sum;
}

/* Run scalar conv2d for all output locations */
static void scalar_conv2d_full(const float *input, const float *weight, float *output,
                               int h, int w, int ic, int oc) {
    int oh_size = h - 2;
    int ow_size = w - 2;
    
    for (int oci = 0; oci < oc; oci++) {
        const float *wt = weight + oci * (9 * ic);
        for (int oh = 0; oh < oh_size; oh++) {
            for (int ow = 0; ow < ow_size; ow++) {
                int out_idx = (oh * ow_size + ow) * oc + oci;
                output[out_idx] = scalar_conv3x3(input, wt, h, w, ic, oh, ow);
            }
        }
    }
}

/* MXU conv2d - uses inner product for each output */
static void mxu_conv2d_full(mars_nn_hw_ctx_t *ctx,
                            uint32_t input_off, uint32_t weight_off, uint32_t output_off,
                            uint32_t patch_off, /* temp buffer for im2col patch */
                            int h, int w, int ic, int oc) {
    int oh_size = h - 2;
    int ow_size = w - 2;
    int kernel_elems = 9 * ic;  /* 3x3 * channels */
    
    float *input = (float *)((char *)ctx->oram_vaddr + input_off);
    float *output = (float *)((char *)ctx->oram_vaddr + output_off);
    float *patch = (float *)((char *)ctx->oram_vaddr + patch_off);
    
    for (int oh = 0; oh < oh_size; oh++) {
        for (int ow = 0; ow < ow_size; ow++) {
            /* Extract im2col patch (3x3 x ic) */
            int pi = 0;
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    int ih = oh + kh;
                    int iw = ow + kw;
                    for (int c = 0; c < ic; c++) {
                        patch[pi++] = input[(ih * w + iw) * ic + c];
                    }
                }
            }
            
            /* Compute all output channels using MXU dot product */
            for (int oci = 0; oci < oc; oci++) {
                uint32_t wt_off = weight_off + oci * kernel_elems * sizeof(float);
                float val = mars_nn_hw_mxu_dot(ctx, patch_off, wt_off, kernel_elems);
                int out_idx = (oh * ow_size + ow) * oc + oci;
                output[out_idx] = val;
            }
        }
    }
}

static void run_conv_benchmark(mars_nn_hw_ctx_t *ctx, const conv_config_t *cfg) {
    mars_nn_timing_t t;
    int h = cfg->h, w = cfg->w, ic = cfg->ic, oc = cfg->oc;
    int oh = h - 2, ow = w - 2;
    
    size_t in_size = h * w * ic * sizeof(float);
    size_t wt_size = oc * 9 * ic * sizeof(float);  /* oc * 3x3 * ic */
    size_t out_size = oh * ow * oc * sizeof(float);
    size_t patch_size = 9 * ic * sizeof(float);
    
    printf("\n=== %s (in=%zuKB, wt=%zuKB, out=%zuKB) ===\n",
           cfg->name, in_size/1024, wt_size/1024, out_size/1024);
    
    /* Check if we fit in ORAM */
    size_t total = in_size + wt_size + out_size + patch_size + 256;
    if (total > ctx->oram_size) {
        printf("  SKIP: needs %zuKB, only %uKB available\n", total/1024, ctx->oram_size/1024);
        return;
    }
    
    /* Allocate ORAM regions */
    mars_nn_hw_oram_reset(ctx);
    uint32_t in_off = mars_nn_hw_oram_alloc(ctx, in_size, 64);
    uint32_t wt_off = mars_nn_hw_oram_alloc(ctx, wt_size, 64);
    uint32_t out_off = mars_nn_hw_oram_alloc(ctx, out_size, 64);
    uint32_t patch_off = mars_nn_hw_oram_alloc(ctx, ((patch_size + 63) / 64) * 64, 64);
    
    /* Initialize input and weights */
    float *input = (float *)((char *)ctx->oram_vaddr + in_off);
    float *weight = (float *)((char *)ctx->oram_vaddr + wt_off);
    float *output_mxu = (float *)((char *)ctx->oram_vaddr + out_off);
    
    for (size_t i = 0; i < in_size / sizeof(float); i++)
        input[i] = (float)(i % 256) * 0.01f;
    for (size_t i = 0; i < wt_size / sizeof(float); i++)
        weight[i] = (float)((i * 7) % 256) * 0.001f - 0.128f;
    
    /* Allocate host buffers for scalar version */
    float *host_out = (float *)aligned_alloc(64, out_size);
    memset(host_out, 0, out_size);

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERS; i++) {
        scalar_conv2d_full(input, weight, host_out, h, w, ic, oc);
    }

    /* Benchmark scalar */
    mars_nn_timing_start(&t);
    for (int i = 0; i < BENCH_ITERS; i++) {
        scalar_conv2d_full(input, weight, host_out, h, w, ic, oc);
    }
    mars_nn_timing_end(&t);
    double scalar_us = t.elapsed_us / BENCH_ITERS;

    /* Warmup MXU */
    for (int i = 0; i < WARMUP_ITERS; i++) {
        mxu_conv2d_full(ctx, in_off, wt_off, out_off, patch_off, h, w, ic, oc);
    }

    /* Benchmark MXU */
    mars_nn_timing_start(&t);
    for (int i = 0; i < BENCH_ITERS; i++) {
        mxu_conv2d_full(ctx, in_off, wt_off, out_off, patch_off, h, w, ic, oc);
    }
    mars_nn_timing_end(&t);
    double mxu_us = t.elapsed_us / BENCH_ITERS;

    /* Verify first few outputs */
    int mismatches = 0;
    for (int i = 0; i < (int)(out_size / sizeof(float)) && i < 100; i++) {
        float diff = host_out[i] - output_mxu[i];
        if (diff < 0) diff = -diff;
        if (diff > 0.01f) mismatches++;
    }

    /* Calculate FLOPs: each output = 9 * ic * 2 (mul + add) */
    double flops_per_out = 9.0 * ic * 2.0;
    double total_flops = (double)oh * ow * oc * flops_per_out;
    double scalar_gflops = total_flops / (scalar_us * 1000.0);
    double mxu_gflops = total_flops / (mxu_us * 1000.0);
    double speedup = scalar_us / mxu_us;

    printf("  Scalar: %.1f ms (%.3f GFLOPS)\n", scalar_us/1000.0, scalar_gflops);
    printf("  MXU:    %.1f ms (%.3f GFLOPS), speedup=%.2fx %s\n",
           mxu_us/1000.0, mxu_gflops, speedup, mismatches ? "MISMATCH" : "OK");

    free(host_out);
}

int main(void) {
    printf("=============================================\n");
    printf("Mars Conv2D Benchmark - MXU vs Scalar\n");
    printf("3x3 convolution, valid padding\n");
    printf("=============================================\n");

    mars_nn_hw_ctx_t ctx;
    if (mars_nn_hw_init(&ctx) < 0) {
        fprintf(stderr, "Failed to initialize NNA hardware\n");
        return 1;
    }

    for (size_t i = 0; i < NUM_CONFIGS; i++) {
        run_conv_benchmark(&ctx, &CONFIGS[i]);
    }

    printf("\n=============================================\n");
    printf("Benchmark complete!\n");
    printf("=============================================\n");

    mars_nn_hw_cleanup(&ctx);
    return 0;
}

