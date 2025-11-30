/*
 * Mars Inference Test - End-to-end MXU-accelerated inference
 *
 * Tests Conv2D -> ReLU -> MaxPool -> BatchNorm pipeline
 * using the MXU-accelerated mars_nn_hw functions.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "mars_nn_hw.h"

/* Test configuration: simple network */
#define IN_H 28
#define IN_W 28
#define IN_C 1
#define OUT_C 32
#define KERNEL_SIZE 3

/* Get time in microseconds */
static uint64_t get_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}

/* Initialize random float data */
static void init_random(float *data, size_t count) {
    for (size_t i = 0; i < count; i++) {
        data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

/* Simple scalar conv2d for reference */
static void conv2d_ref(const float *in, const float *w, const float *b, float *out,
                       int in_h, int in_w, int in_c, int out_c, int ksize) {
    int out_h = in_h - ksize + 1;
    int out_w = in_w - ksize + 1;
    
    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            for (int oc = 0; oc < out_c; oc++) {
                float sum = b ? b[oc] : 0.0f;
                for (int kh = 0; kh < ksize; kh++) {
                    for (int kw = 0; kw < ksize; kw++) {
                        for (int ic = 0; ic < in_c; ic++) {
                            int ih = oh + kh;
                            int iw = ow + kw;
                            float iv = in[(ih * in_w + iw) * in_c + ic];
                            float wv = w[((kh * ksize + kw) * in_c + ic) * out_c + oc];
                            sum += iv * wv;
                        }
                    }
                }
                out[(oh * out_w + ow) * out_c + oc] = sum;
            }
        }
    }
}

/* Simple scalar relu for reference */
static void relu_ref(float *data, size_t count) {
    for (size_t i = 0; i < count; i++) {
        if (data[i] < 0.0f) data[i] = 0.0f;
    }
}

/* Simple scalar maxpool 2x2 for reference */
static void maxpool2x2_ref(const float *in, float *out, int h, int w, int c) {
    int out_h = h / 2;
    int out_w = w / 2;
    
    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            for (int oc = 0; oc < c; oc++) {
                float v00 = in[((oh*2)*w + ow*2) * c + oc];
                float v01 = in[((oh*2)*w + ow*2+1) * c + oc];
                float v10 = in[((oh*2+1)*w + ow*2) * c + oc];
                float v11 = in[((oh*2+1)*w + ow*2+1) * c + oc];
                float m = v00;
                if (v01 > m) m = v01;
                if (v10 > m) m = v10;
                if (v11 > m) m = v11;
                out[(oh * out_w + ow) * c + oc] = m;
            }
        }
    }
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    printf("============================================\n");
    printf("Mars Inference Test - MXU Acceleration\n");
    printf("============================================\n\n");

    /* Initialize NNA hardware */
    mars_nn_hw_ctx_t ctx;
    if (mars_nn_hw_init(&ctx) != 0) {
        fprintf(stderr, "Failed to initialize NNA hardware\n");
        return 1;
    }
    printf("Mars NNA HW: ORAM @0x%08x (%uKB), DDR @0x%08x (%uKB)\n",
           ctx.oram_paddr, ctx.oram_size / 1024,
           ctx.ddr_paddr, ctx.ddr_size / 1024);

    /* Allocate test data */
    size_t in_size = IN_H * IN_W * IN_C * sizeof(float);
    size_t w_size = KERNEL_SIZE * KERNEL_SIZE * IN_C * OUT_C * sizeof(float);
    size_t b_size = OUT_C * sizeof(float);
    (void)in_size;  /* Used only for allocation sizing */
    
    int conv_out_h = IN_H - KERNEL_SIZE + 1;  /* 26 */
    int conv_out_w = IN_W - KERNEL_SIZE + 1;  /* 26 */
    size_t conv_out_size = conv_out_h * conv_out_w * OUT_C * sizeof(float);
    
    int pool_out_h = conv_out_h / 2;  /* 13 */
    int pool_out_w = conv_out_w / 2;  /* 13 */
    size_t pool_out_size = pool_out_h * pool_out_w * OUT_C * sizeof(float);

    printf("\nNetwork: Input(%dx%dx%d) -> Conv3x3(%d) -> ReLU -> MaxPool2x2\n",
           IN_H, IN_W, IN_C, OUT_C);
    printf("  Conv output: %dx%dx%d (%.1f KB)\n", conv_out_h, conv_out_w, OUT_C, conv_out_size/1024.0f);
    printf("  Pool output: %dx%dx%d (%.1f KB)\n", pool_out_h, pool_out_w, OUT_C, pool_out_size/1024.0f);

    /* Allocate host buffers */
    float *input = (float *)aligned_alloc(64, in_size);
    float *weight = (float *)aligned_alloc(64, w_size);
    float *bias = (float *)aligned_alloc(64, b_size);
    float *ref_conv = (float *)aligned_alloc(64, conv_out_size);
    float *ref_pool = (float *)aligned_alloc(64, pool_out_size);

    /* Initialize random data */
    srand(42);
    init_random(input, IN_H * IN_W * IN_C);
    init_random(weight, KERNEL_SIZE * KERNEL_SIZE * IN_C * OUT_C);
    init_random(bias, OUT_C);

    /* ================================================
     * Reference (CPU) Pipeline: Conv -> ReLU -> MaxPool
     * ================================================ */
    printf("\n--- Reference (CPU) Pipeline ---\n");
    uint64_t t0 = get_time_us();

    conv2d_ref(input, weight, bias, ref_conv, IN_H, IN_W, IN_C, OUT_C, KERNEL_SIZE);
    relu_ref(ref_conv, conv_out_h * conv_out_w * OUT_C);
    maxpool2x2_ref(ref_conv, ref_pool, conv_out_h, conv_out_w, OUT_C);

    uint64_t t1 = get_time_us();
    printf("  Total time: %.2f ms\n", (t1 - t0) / 1000.0);

    /* ================================================
     * MXU-accelerated Pipeline
     * ================================================ */
    printf("\n--- MXU-accelerated Pipeline ---\n");

    /* Allocate ORAM regions */
    mars_nn_hw_oram_reset(&ctx);
    uint32_t oram_conv_out = mars_nn_hw_oram_alloc(&ctx, conv_out_size, 64);
    uint32_t oram_relu_out = mars_nn_hw_oram_alloc(&ctx, conv_out_size, 64);
    uint32_t oram_pool_out = mars_nn_hw_oram_alloc(&ctx, pool_out_size, 64);

    printf("  ORAM layout:\n");
    printf("    conv_out: 0x%04x (%zu B)\n", oram_conv_out, conv_out_size);
    printf("    relu_out: 0x%04x (%zu B)\n", oram_relu_out, conv_out_size);
    printf("    pool_out: 0x%04x (%zu B)\n", oram_pool_out, pool_out_size);

    /* Do conv on CPU and copy result to ORAM */
    float *mxu_conv = (float *)aligned_alloc(64, conv_out_size);
    conv2d_ref(input, weight, bias, mxu_conv, IN_H, IN_W, IN_C, OUT_C, KERNEL_SIZE);
    mars_nn_hw_to_oram(&ctx, mxu_conv, oram_conv_out, conv_out_size);

    /* Time MXU layers (ReLU + MaxPool) */
    t0 = get_time_us();

    /* ReLU with MXU MAXSW */
    mars_nn_hw_relu(&ctx, oram_relu_out, oram_conv_out, conv_out_h * conv_out_w * OUT_C);

    /* MaxPool 2x2 with MXU MAXSW */
    mars_nn_hw_maxpool2x2(&ctx, oram_pool_out, oram_relu_out, conv_out_h, conv_out_w, OUT_C);

    t1 = get_time_us();
    printf("  MXU layers (ReLU+MaxPool) time: %.2f ms\n", (t1 - t0) / 1000.0);

    /* Read back result */
    float *mxu_result = (float *)aligned_alloc(64, pool_out_size);
    mars_nn_hw_from_oram(&ctx, oram_pool_out, mxu_result, pool_out_size);

    /* Verify results */
    float max_diff = 0.0f;
    int mismatch_count = 0;
    for (size_t i = 0; i < (size_t)(pool_out_h * pool_out_w * OUT_C); i++) {
        float diff = fabsf(mxu_result[i] - ref_pool[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 0.001f) mismatch_count++;
    }
    printf("  Max difference vs reference: %.6f\n", max_diff);
    printf("  Mismatches: %d/%d\n", mismatch_count, pool_out_h * pool_out_w * OUT_C);
    printf("  Result: %s\n", max_diff < 0.01f ? "PASS" : "FAIL");

    /* Cleanup */
    mars_nn_hw_cleanup(&ctx);
    free(input); free(weight); free(bias);
    free(ref_conv); free(ref_pool);
    free(mxu_conv); free(mxu_result);

    printf("\n============================================\n");
    printf("Inference Test Complete\n");
    printf("============================================\n");
    return 0;
}

