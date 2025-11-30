/*
 * Mars NN Layer Benchmark
 *
 * Benchmarks new layer operations: ReLU, LeakyReLU, BatchNorm, MaxPool, AvgPool
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mars_nn_hw.h"

#define WARMUP_ITERS 3
#define BENCH_ITERS 10

/* Scalar reference implementations */
static void scalar_relu(const float *src, float *dst, size_t n) {
    for (size_t i = 0; i < n; i++)
        dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
}

static void scalar_leaky_relu(const float *src, float *dst, size_t n, float alpha) {
    for (size_t i = 0; i < n; i++) {
        float x = src[i];
        dst[i] = x > 0.0f ? x : alpha * x;
    }
}

static void scalar_batchnorm(const float *src, const float *scale, const float *bias,
                             float *dst, size_t n) {
    for (size_t i = 0; i < n; i++)
        dst[i] = src[i] * scale[i] + bias[i];
}

static void scalar_maxpool2x2(const float *src, float *dst, int h, int w, int c) {
    int oh = h / 2, ow = w / 2;
    for (int y = 0; y < oh; y++) {
        for (int x = 0; x < ow; x++) {
            int iy = y * 2, ix = x * 2;
            for (int ch = 0; ch < c; ch++) {
                float v00 = src[(iy * w + ix) * c + ch];
                float v01 = src[(iy * w + ix + 1) * c + ch];
                float v10 = src[((iy + 1) * w + ix) * c + ch];
                float v11 = src[((iy + 1) * w + ix + 1) * c + ch];
                float mx = v00;
                if (v01 > mx) mx = v01;
                if (v10 > mx) mx = v10;
                if (v11 > mx) mx = v11;
                dst[(y * ow + x) * c + ch] = mx;
            }
        }
    }
}

static void run_activation_bench(mars_nn_hw_ctx_t *ctx, const char *name, size_t count) {
    mars_nn_timing_t t;
    size_t size = count * sizeof(float);
    
    printf("\n=== %s: %zu floats (%.1f KB) ===\n", name, count, size / 1024.0);
    
    mars_nn_hw_oram_reset(ctx);
    uint32_t src_off = mars_nn_hw_oram_alloc(ctx, size, 64);
    uint32_t dst_off = mars_nn_hw_oram_alloc(ctx, size, 64);
    
    float *src = (float *)((char *)ctx->oram_vaddr + src_off);
    float *dst_mxu = (float *)((char *)ctx->oram_vaddr + dst_off);
    float *dst_scalar = aligned_alloc(64, size);
    
    /* Init with mixed pos/neg values */
    for (size_t i = 0; i < count; i++)
        src[i] = (float)(i % 256) * 0.01f - 1.28f;
    
    /* Warmup + benchmark scalar */
    for (int i = 0; i < WARMUP_ITERS; i++) scalar_relu(src, dst_scalar, count);
    mars_nn_timing_start(&t);
    for (int i = 0; i < BENCH_ITERS; i++) scalar_relu(src, dst_scalar, count);
    mars_nn_timing_end(&t);
    double scalar_us = t.elapsed_us / BENCH_ITERS;
    
    /* Warmup + benchmark MXU */
    for (int i = 0; i < WARMUP_ITERS; i++) mars_nn_hw_relu(ctx, dst_off, src_off, count);
    mars_nn_timing_start(&t);
    for (int i = 0; i < BENCH_ITERS; i++) mars_nn_hw_relu(ctx, dst_off, src_off, count);
    mars_nn_timing_end(&t);
    double mxu_us = t.elapsed_us / BENCH_ITERS;
    
    /* Verify */
    int errors = 0;
    for (size_t i = 0; i < count && i < 100; i++)
        if (fabsf(dst_mxu[i] - dst_scalar[i]) > 0.001f) errors++;
    
    double gb_per_sec = (2.0 * size) / (mxu_us * 1000.0); /* read + write */
    printf("  Scalar: %.1f us\n", scalar_us);
    printf("  MXU:    %.1f us, speedup=%.2fx, %.2f GB/s %s\n",
           mxu_us, scalar_us / mxu_us, gb_per_sec, errors ? "MISMATCH" : "OK");
    
    free(dst_scalar);
}

static void run_batchnorm_bench(mars_nn_hw_ctx_t *ctx, size_t count) {
    mars_nn_timing_t t;
    size_t size = count * sizeof(float);
    size_t total_needed = size * 4 + 256;  /* 4 buffers + alignment */

    printf("\n=== BatchNorm: %zu floats (%.1f KB, need %.1f KB) ===\n",
           count, size / 1024.0, total_needed / 1024.0);

    if (total_needed > ctx->oram_size) {
        printf("  SKIP: exceeds ORAM size\n");
        return;
    }

    mars_nn_hw_oram_reset(ctx);
    uint32_t src_off = mars_nn_hw_oram_alloc(ctx, size, 64);
    uint32_t scale_off = mars_nn_hw_oram_alloc(ctx, size, 64);
    uint32_t bias_off = mars_nn_hw_oram_alloc(ctx, size, 64);
    uint32_t dst_off = mars_nn_hw_oram_alloc(ctx, size, 64);

    printf("  Offsets: src=%u, scale=%u, bias=%u, dst=%u\n",
           src_off, scale_off, bias_off, dst_off);
    
    float *src = (float *)((char *)ctx->oram_vaddr + src_off);
    float *scale = (float *)((char *)ctx->oram_vaddr + scale_off);
    float *bias = (float *)((char *)ctx->oram_vaddr + bias_off);
    float *dst_mxu = (float *)((char *)ctx->oram_vaddr + dst_off);
    float *dst_scalar = aligned_alloc(64, size);
    
    for (size_t i = 0; i < count; i++) {
        src[i] = (float)(i % 256) * 0.01f;
        scale[i] = 1.0f + (float)(i % 10) * 0.01f;
        bias[i] = (float)(i % 20) * 0.001f;
    }
    
    /* Benchmark scalar */
    for (int i = 0; i < WARMUP_ITERS; i++) scalar_batchnorm(src, scale, bias, dst_scalar, count);
    mars_nn_timing_start(&t);
    for (int i = 0; i < BENCH_ITERS; i++) scalar_batchnorm(src, scale, bias, dst_scalar, count);
    mars_nn_timing_end(&t);
    double scalar_us = t.elapsed_us / BENCH_ITERS;
    
    /* Benchmark MXU */
    for (int i = 0; i < WARMUP_ITERS; i++)
        mars_nn_hw_batchnorm(ctx, dst_off, src_off, scale_off, bias_off, count);
    mars_nn_timing_start(&t);
    for (int i = 0; i < BENCH_ITERS; i++)
        mars_nn_hw_batchnorm(ctx, dst_off, src_off, scale_off, bias_off, count);
    mars_nn_timing_end(&t);
    double mxu_us = t.elapsed_us / BENCH_ITERS;
    
    int errors = 0;
    for (size_t i = 0; i < count && i < 100; i++)
        if (fabsf(dst_mxu[i] - dst_scalar[i]) > 0.01f) errors++;
    
    double gb_per_sec = (4.0 * size) / (mxu_us * 1000.0);
    printf("  Scalar: %.1f us\n", scalar_us);
    printf("  MXU:    %.1f us, speedup=%.2fx, %.2f GB/s %s\n",
           mxu_us, scalar_us / mxu_us, gb_per_sec, errors ? "MISMATCH" : "OK");
    
    free(dst_scalar);
}

static void run_maxpool_bench(mars_nn_hw_ctx_t *ctx, int h, int w, int c) {
    mars_nn_timing_t t;
    size_t in_size = h * w * c * sizeof(float);
    size_t out_size = (h/2) * (w/2) * c * sizeof(float);
    size_t total_needed = in_size + out_size + 128;

    printf("\n=== MaxPool2x2: %dx%dx%d -> %dx%dx%d (need %.1f KB) ===\n",
           h, w, c, h/2, w/2, c, total_needed / 1024.0);

    if (total_needed > ctx->oram_size) {
        printf("  SKIP: exceeds ORAM size\n");
        return;
    }

    mars_nn_hw_oram_reset(ctx);
    uint32_t src_off = mars_nn_hw_oram_alloc(ctx, in_size, 64);
    uint32_t dst_off = mars_nn_hw_oram_alloc(ctx, out_size, 64);

    printf("  Offsets: src=%u, dst=%u\n", src_off, dst_off);

    float *src = (float *)((char *)ctx->oram_vaddr + src_off);
    float *dst_mxu = (float *)((char *)ctx->oram_vaddr + dst_off);
    float *dst_scalar = aligned_alloc(64, out_size);

    for (size_t i = 0; i < in_size/sizeof(float); i++)
        src[i] = (float)(i % 1024) * 0.001f;

    /* Benchmark scalar */
    for (int i = 0; i < WARMUP_ITERS; i++) scalar_maxpool2x2(src, dst_scalar, h, w, c);
    mars_nn_timing_start(&t);
    for (int i = 0; i < BENCH_ITERS; i++) scalar_maxpool2x2(src, dst_scalar, h, w, c);
    mars_nn_timing_end(&t);
    double scalar_us = t.elapsed_us / BENCH_ITERS;

    /* Benchmark MXU */
    for (int i = 0; i < WARMUP_ITERS; i++)
        mars_nn_hw_maxpool2x2(ctx, dst_off, src_off, h, w, c);
    mars_nn_timing_start(&t);
    for (int i = 0; i < BENCH_ITERS; i++)
        mars_nn_hw_maxpool2x2(ctx, dst_off, src_off, h, w, c);
    mars_nn_timing_end(&t);
    double mxu_us = t.elapsed_us / BENCH_ITERS;

    int errors = 0;
    for (size_t i = 0; i < out_size/sizeof(float) && i < 100; i++)
        if (fabsf(dst_mxu[i] - dst_scalar[i]) > 0.001f) errors++;

    printf("  Scalar: %.1f us\n", scalar_us);
    printf("  MXU:    %.1f us, speedup=%.2fx %s\n",
           mxu_us, scalar_us / mxu_us, errors ? "MISMATCH" : "OK");

    free(dst_scalar);
}

int main(void) {
    printf("=============================================\n");
    printf("Mars NN Layer Benchmark\n");
    printf("=============================================\n");

    mars_nn_hw_ctx_t ctx;
    if (mars_nn_hw_init(&ctx) < 0) {
        fprintf(stderr, "Failed to initialize NNA hardware\n");
        return 1;
    }

    /* ReLU benchmarks - max 2 buffers: 2 * size <= 640KB, so max ~80K floats */
    run_activation_bench(&ctx, "ReLU", 1024);
    run_activation_bench(&ctx, "ReLU", 16384);
    run_activation_bench(&ctx, "ReLU", 32768);  /* 128KB * 2 = 256KB */

    /* BatchNorm benchmarks - 4 buffers: 4 * size <= 640KB, so max ~40K floats */
    run_batchnorm_bench(&ctx, 1024);   /* 4KB * 4 = 16KB */
    run_batchnorm_bench(&ctx, 4096);   /* 16KB * 4 = 64KB */
    run_batchnorm_bench(&ctx, 16384);  /* 64KB * 4 = 256KB */

    /* MaxPool benchmarks - sized to fit in ORAM conservatively */
    run_maxpool_bench(&ctx, 16, 16, 16);    /* 16KB in, 4KB out */
    run_maxpool_bench(&ctx, 20, 20, 32);    /* 50KB in, 12KB out */
    run_maxpool_bench(&ctx, 28, 28, 32);    /* 100KB in, 25KB out */

    printf("\n=============================================\n");
    printf("Benchmark complete!\n");
    printf("=============================================\n");

    mars_nn_hw_cleanup(&ctx);
    return 0;
}

