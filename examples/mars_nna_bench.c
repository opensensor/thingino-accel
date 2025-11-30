/*
 * Mars NNA Benchmark
 *
 * Benchmarks MXU vector operations vs scalar baseline using NNA ORAM.
 * Demonstrates the Mars -> NNA hardware path.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mars_nn_hw.h"
#include "mars_math.h"

#define ALIGN64 __attribute__((aligned(64)))

/* Benchmark sizes (in floats) - 16K floats is sweet spot before overhead/stability issues */
static const size_t SIZES[] = { 64, 256, 1024, 4096, 16384 };
#define NUM_SIZES (sizeof(SIZES) / sizeof(SIZES[0]))
#define WARMUP_ITERS 3
#define BENCH_ITERS 10

/* Scalar baseline add */
static void scalar_vec_add(float *dst, const float *a, const float *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = a[i] + b[i];
    }
}

/* Scalar baseline mul */
static void scalar_vec_mul(float *dst, const float *a, const float *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = a[i] * b[i];
    }
}

/* Verify results match */
static int verify(const float *a, const float *b, size_t n, float eps) {
    for (size_t i = 0; i < n; i++) {
        float diff = a[i] - b[i];
        if (diff < 0) diff = -diff;
        if (diff > eps) {
            printf("  MISMATCH at [%zu]: got %f, expected %f\n", i, a[i], b[i]);
            return 0;
        }
    }
    return 1;
}

static void run_benchmark(mars_nn_hw_ctx_t *ctx, size_t count) {
    size_t bytes = count * sizeof(float);
    mars_nn_timing_t t;
    double scalar_add_us = 0, mxu_add_us = 0;
    double scalar_mul_us = 0, mxu_mul_us = 0;

    printf("\n=== Benchmark: %zu floats (%zu bytes) ===\n", count, bytes);

    /* Allocate ORAM regions */
    mars_nn_hw_oram_reset(ctx);
    uint32_t off_a = mars_nn_hw_oram_alloc(ctx, bytes, 64);
    uint32_t off_b = mars_nn_hw_oram_alloc(ctx, bytes, 64);
    uint32_t off_dst = mars_nn_hw_oram_alloc(ctx, bytes, 64);
    
    if (off_a == (uint32_t)-1 || off_b == (uint32_t)-1 || off_dst == (uint32_t)-1) {
        printf("  SKIP: not enough ORAM for this size\n");
        return;
    }

    /* Prepare test data */
    float *host_a = (float *)aligned_alloc(64, bytes);
    float *host_b = (float *)aligned_alloc(64, bytes);
    float *host_result = (float *)aligned_alloc(64, bytes);
    float *host_expected = (float *)aligned_alloc(64, bytes);

    for (size_t i = 0; i < count; i++) {
        host_a[i] = (float)(i % 100) * 0.01f + 1.0f;
        host_b[i] = (float)((i * 7) % 100) * 0.01f + 0.5f;
    }

    /* Copy to ORAM */
    mars_nn_hw_to_oram(ctx, host_a, off_a, bytes);
    mars_nn_hw_to_oram(ctx, host_b, off_b, bytes);

    /* === Vector ADD benchmark === */
    
    /* Warmup scalar */
    for (int i = 0; i < WARMUP_ITERS; i++) {
        scalar_vec_add(host_expected, host_a, host_b, count);
    }

    /* Benchmark scalar add */
    mars_nn_timing_start(&t);
    for (int i = 0; i < BENCH_ITERS; i++) {
        scalar_vec_add(host_expected, host_a, host_b, count);
    }
    mars_nn_timing_end(&t);
    scalar_add_us = t.elapsed_us / BENCH_ITERS;

    /* Warmup MXU */
    for (int i = 0; i < WARMUP_ITERS; i++) {
        mars_nn_hw_mxu_vec_op(ctx, off_dst, off_a, off_b, count, 0);
    }

    /* Benchmark MXU add */
    mars_nn_timing_start(&t);
    for (int i = 0; i < BENCH_ITERS; i++) {
        mars_nn_hw_mxu_vec_op(ctx, off_dst, off_a, off_b, count, 0);
    }
    mars_nn_timing_end(&t);
    mxu_add_us = t.elapsed_us / BENCH_ITERS;

    /* Verify ADD */
    mars_nn_hw_from_oram(ctx, off_dst, host_result, bytes);
    int add_ok = verify(host_result, host_expected, count, 1e-5f);

    /* === Vector MUL benchmark === */

    /* Benchmark scalar mul */
    mars_nn_timing_start(&t);
    for (int i = 0; i < BENCH_ITERS; i++) {
        scalar_vec_mul(host_expected, host_a, host_b, count);
    }
    mars_nn_timing_end(&t);
    scalar_mul_us = t.elapsed_us / BENCH_ITERS;

    /* Benchmark MXU mul */
    mars_nn_timing_start(&t);
    for (int i = 0; i < BENCH_ITERS; i++) {
        mars_nn_hw_mxu_vec_op(ctx, off_dst, off_a, off_b, count, 1);
    }
    mars_nn_timing_end(&t);
    mxu_mul_us = t.elapsed_us / BENCH_ITERS;

    /* Verify MUL */
    mars_nn_hw_from_oram(ctx, off_dst, host_result, bytes);
    int mul_ok = verify(host_result, host_expected, count, 1e-4f);

    /* Print results */
    double add_speedup = scalar_add_us / mxu_add_us;
    double mul_speedup = scalar_mul_us / mxu_mul_us;
    double add_gflops = (count * BENCH_ITERS) / (mxu_add_us * 1000.0);
    double mul_gflops = (count * BENCH_ITERS) / (mxu_mul_us * 1000.0);

    printf("  ADD: scalar=%.1f us, MXU=%.1f us, speedup=%.2fx, %.3f GFLOPS %s\n",
           scalar_add_us, mxu_add_us, add_speedup, add_gflops, add_ok ? "OK" : "FAIL");
    printf("  MUL: scalar=%.1f us, MXU=%.1f us, speedup=%.2fx, %.3f GFLOPS %s\n",
           scalar_mul_us, mxu_mul_us, mul_speedup, mul_gflops, mul_ok ? "OK" : "FAIL");

    free(host_a); free(host_b); free(host_result); free(host_expected);
}

int main(void) {
    printf("========================================\n");
    printf("Mars NNA Benchmark - MXU vs Scalar\n");
    printf("========================================\n");

    mars_nn_hw_ctx_t ctx;
    if (mars_nn_hw_init(&ctx) < 0) {
        fprintf(stderr, "Failed to initialize NNA hardware\n");
        return 1;
    }

    printf("\nRunning benchmarks (%d warmup, %d iterations each)...\n",
           WARMUP_ITERS, BENCH_ITERS);

    for (size_t i = 0; i < NUM_SIZES; i++) {
        run_benchmark(&ctx, SIZES[i]);
    }

    printf("\n========================================\n");
    printf("Benchmark complete!\n");
    printf("========================================\n");

    mars_nn_hw_cleanup(&ctx);
    return 0;
}

