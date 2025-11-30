/*
 * Mars NN Hardware Interface
 *
 * Concrete hardware interface for executing Mars NN programs on the T41 NNA.
 * Uses ORAM for data staging, NNDMA for transfers, and MXU for compute.
 *
 * MXU Compute Operations (from MXUv3 PDF):
 * - VPR_ADD, VPR_SUB, VPR_MUL: 16×float32 vector ops
 * - LA0_VPR, SA0_VPR: 512-bit VPR load/store
 * - VSR0-VSR3: Sum registers for MAC accumulation
 */

#pragma once

#include <stddef.h>
#include <stdint.h>
#include "mars_nn_cmd.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Hardware context for NNA operations */
typedef struct {
    int fd;                     /* /dev/soc-nna file descriptor */
    int memfd;                  /* /dev/mem file descriptor */
    void *oram_vaddr;           /* Mapped ORAM pointer */
    uint32_t oram_paddr;        /* ORAM physical address */
    uint32_t oram_size;         /* Available ORAM size in bytes */
    void *ddr_vaddr;            /* Mapped DDR DMA buffer */
    uint32_t ddr_paddr;         /* DDR physical address */
    uint32_t ddr_size;          /* DDR buffer size */
    void *desram_vaddr;         /* Mapped descriptor RAM */
    void *dma_io_vaddr;         /* Mapped DMA I/O registers */
    uint32_t desram_rd_idx;     /* Current read descriptor index */
    uint32_t desram_wr_idx;     /* Current write descriptor index */
    int initialized;
} mars_nn_hw_ctx_t;

/* Initialize NNA hardware context */
int mars_nn_hw_init(mars_nn_hw_ctx_t *ctx);

/* Cleanup NNA hardware context */
void mars_nn_hw_cleanup(mars_nn_hw_ctx_t *ctx);

/* Allocate region in ORAM (simple bump allocator) */
uint32_t mars_nn_hw_oram_alloc(mars_nn_hw_ctx_t *ctx, uint32_t size, uint32_t align);

/* Reset ORAM allocator */
void mars_nn_hw_oram_reset(mars_nn_hw_ctx_t *ctx);

/* Copy data to ORAM (host -> ORAM) */
int mars_nn_hw_to_oram(mars_nn_hw_ctx_t *ctx, const void *src, 
                       uint32_t oram_offset, uint32_t size);

/* Copy data from ORAM (ORAM -> host) */
int mars_nn_hw_from_oram(mars_nn_hw_ctx_t *ctx, uint32_t oram_offset,
                         void *dst, uint32_t size);

/* DMA transfer DDR -> ORAM (uses NNDMA hardware) */
int mars_nn_hw_dma_to_oram(mars_nn_hw_ctx_t *ctx, uint32_t ddr_offset,
                           uint32_t oram_offset, uint32_t size);

/* DMA transfer ORAM -> DDR (uses NNDMA hardware) */
int mars_nn_hw_dma_from_oram(mars_nn_hw_ctx_t *ctx, uint32_t oram_offset,
                             uint32_t ddr_offset, uint32_t size);

/* Wait for DMA completion */
int mars_nn_hw_dma_wait(mars_nn_hw_ctx_t *ctx);

/* Execute MXU vector operation on ORAM data
 * This uses MXU VPR registers with ORAM-resident data.
 * op: 0=add, 1=mul, 2=sub
 */
int mars_nn_hw_mxu_vec_op(mars_nn_hw_ctx_t *ctx,
                          uint32_t dst_oram_offset,
                          uint32_t src_a_oram_offset,
                          uint32_t src_b_oram_offset,
                          uint32_t count_floats,
                          int op);

/* MXU inner product: dst[0] = sum(a[i] * b[i])
 * Uses VPR registers for SIMD multiply, VSR for accumulation
 * Returns single float result
 */
float mars_nn_hw_mxu_dot(mars_nn_hw_ctx_t *ctx,
                         uint32_t src_a_oram_offset,
                         uint32_t src_b_oram_offset,
                         uint32_t count_floats);

/* MXU fused multiply-add: dst = a * b + c (element-wise) */
int mars_nn_hw_mxu_fma(mars_nn_hw_ctx_t *ctx,
                       uint32_t dst_oram_offset,
                       uint32_t src_a_oram_offset,
                       uint32_t src_b_oram_offset,
                       uint32_t src_c_oram_offset,
                       uint32_t count_floats);

/* Convolution kernel: single output channel
 * Computes one output pixel from input patch and filter weights
 * Uses MXU inner product for kernel × patch
 */
float mars_nn_hw_conv_kernel(mars_nn_hw_ctx_t *ctx,
                             uint32_t input_oram_offset,
                             uint32_t weight_oram_offset,
                             uint32_t kernel_size);  /* kH * kW * C */

/* ============================================================================
 * NN Layer Operations (Option C)
 * ============================================================================ */

/* ReLU activation: dst[i] = max(0, src[i]) */
int mars_nn_hw_relu(mars_nn_hw_ctx_t *ctx,
                    uint32_t dst_oram_offset,
                    uint32_t src_oram_offset,
                    uint32_t count_floats);

/* LeakyReLU activation: dst[i] = src[i] > 0 ? src[i] : alpha * src[i] */
int mars_nn_hw_leaky_relu(mars_nn_hw_ctx_t *ctx,
                          uint32_t dst_oram_offset,
                          uint32_t src_oram_offset,
                          uint32_t count_floats,
                          float alpha);

/* Batch normalization (fused scale + bias): dst = src * scale + bias */
int mars_nn_hw_batchnorm(mars_nn_hw_ctx_t *ctx,
                         uint32_t dst_oram_offset,
                         uint32_t src_oram_offset,
                         uint32_t scale_oram_offset,
                         uint32_t bias_oram_offset,
                         uint32_t count_floats);

/* Max pooling 2x2 stride 2: reduces HxW by half */
int mars_nn_hw_maxpool2x2(mars_nn_hw_ctx_t *ctx,
                          uint32_t dst_oram_offset,
                          uint32_t src_oram_offset,
                          uint32_t height, uint32_t width, uint32_t channels);

/* Average pooling 2x2 stride 2: reduces HxW by half */
int mars_nn_hw_avgpool2x2(mars_nn_hw_ctx_t *ctx,
                          uint32_t dst_oram_offset,
                          uint32_t src_oram_offset,
                          uint32_t height, uint32_t width, uint32_t channels);

/* ============================================================================
 * Timing Helpers
 * ============================================================================ */

/* Benchmark timing helper */
typedef struct {
    uint64_t start_cycles;
    uint64_t end_cycles;
    double elapsed_us;
} mars_nn_timing_t;

void mars_nn_timing_start(mars_nn_timing_t *t);
void mars_nn_timing_end(mars_nn_timing_t *t);

/* Get CPU cycle counter (MIPS rdhwr) */
static inline uint64_t mars_nn_get_cycles(void) {
#ifdef __mips__
    uint32_t cycles;
    __asm__ __volatile__(
        "rdhwr %0, $2\n"
        : "=r"(cycles)
    );
    return (uint64_t)cycles;
#else
    return 0;
#endif
}

#ifdef __cplusplus
}
#endif

