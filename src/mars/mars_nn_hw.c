/*
 * Mars NN Hardware Implementation
 *
 * Uses ORAM and NNA DMA for accelerated vector operations.
 */

#include "mars_nn_hw.h"
#include "mxuv3.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/time.h>

/* NNA hardware addresses */
#define NNA_ORAM_BASE       0x12600000
#define NNA_ORAM_MAX_SIZE   0xe0000         /* 896KB max */
#define NNA_DMA_IOBASE      0x12502000
#define NNA_DMA_IOSIZE      0x1000
#define NNA_DESRAM_ADDR     0x1250f000
#define NNA_DESRAM_SIZE     0x4000          /* 16KB */
#define L2CACHE_SIZE_REG    0x10010060

/* IOCTL definitions */
#define SOC_NNA_MAGIC       'c'
#define IOCTL_NNA_MALLOC    _IOWR(SOC_NNA_MAGIC, 0, int)
#define IOCTL_NNA_FREE      _IOWR(SOC_NNA_MAGIC, 1, int)
#define IOCTL_NNA_RDCH      _IOWR(SOC_NNA_MAGIC, 4, int)
#define IOCTL_NNA_WRCH      _IOWR(SOC_NNA_MAGIC, 5, int)

/* DMA descriptor bits */
#define DES_FLAG_CNT        0ULL
#define DES_FLAG_LINK       1ULL
#define DES_FLAG_END        2ULL

struct soc_nna_buf {
    void *vaddr;
    void *paddr;
    int size;
};

/* Simple ORAM bump allocator state */
static uint32_t s_oram_alloc_ptr = 0;

static uint32_t get_l2cache_size(int memfd) {
    void *gpio = mmap(NULL, 0x1000, PROT_READ, MAP_SHARED, memfd, 0x10010000);
    if (gpio == MAP_FAILED) return 0x20000;
    uint32_t val = *(volatile uint32_t *)((char *)gpio + 0x60);
    munmap(gpio, 0x1000);
    uint32_t bits = (val >> 10) & 0x7;
    switch (bits) {
        case 1: return 0x20000;   /* 128KB */
        case 2: return 0x40000;   /* 256KB */
        case 3: return 0x80000;   /* 512KB */
        default: return 0x40000;  /* Default 256KB */
    }
}

int mars_nn_hw_init(mars_nn_hw_ctx_t *ctx) {
    memset(ctx, 0, sizeof(*ctx));

    ctx->memfd = open("/dev/mem", O_RDWR | O_SYNC);
    if (ctx->memfd < 0) { perror("open /dev/mem"); return -1; }

    ctx->fd = open("/dev/soc-nna", O_RDWR);
    if (ctx->fd < 0) { perror("open /dev/soc-nna"); close(ctx->memfd); return -1; }

    uint32_t l2_size = get_l2cache_size(ctx->memfd);
    ctx->oram_paddr = NNA_ORAM_BASE + l2_size;
    ctx->oram_size = NNA_ORAM_MAX_SIZE - l2_size;

    ctx->oram_vaddr = mmap(NULL, ctx->oram_size, PROT_READ | PROT_WRITE,
                           MAP_SHARED, ctx->memfd, ctx->oram_paddr);
    if (ctx->oram_vaddr == MAP_FAILED) { perror("mmap oram"); goto fail; }

    ctx->desram_vaddr = mmap(NULL, NNA_DESRAM_SIZE, PROT_READ | PROT_WRITE,
                             MAP_SHARED, ctx->memfd, NNA_DESRAM_ADDR);
    if (ctx->desram_vaddr == MAP_FAILED) { perror("mmap desram"); goto fail; }

    ctx->dma_io_vaddr = mmap(NULL, NNA_DMA_IOSIZE, PROT_READ | PROT_WRITE,
                             MAP_SHARED, ctx->memfd, NNA_DMA_IOBASE);
    if (ctx->dma_io_vaddr == MAP_FAILED) { perror("mmap dma_io"); goto fail; }

    /* Allocate DDR buffer (1MB for benchmarks) */
    struct soc_nna_buf ddr_buf = { .size = 1024 * 1024 };
    if (ioctl(ctx->fd, IOCTL_NNA_MALLOC, &ddr_buf) < 0) {
        perror("ioctl malloc"); goto fail;
    }
    ctx->ddr_paddr = (uint32_t)(uintptr_t)ddr_buf.paddr;
    ctx->ddr_size = ddr_buf.size;
    ctx->ddr_vaddr = mmap(NULL, ctx->ddr_size, PROT_READ | PROT_WRITE,
                          MAP_SHARED, ctx->memfd, ctx->ddr_paddr);
    if (ctx->ddr_vaddr == MAP_FAILED) { perror("mmap ddr"); goto fail; }

    s_oram_alloc_ptr = 0;
    ctx->initialized = 1;

    printf("Mars NNA HW: ORAM @0x%08x (%uKB), DDR @0x%08x (%uKB)\n",
           ctx->oram_paddr, ctx->oram_size/1024, ctx->ddr_paddr, ctx->ddr_size/1024);
    return 0;

fail:
    mars_nn_hw_cleanup(ctx);
    return -1;
}

void mars_nn_hw_cleanup(mars_nn_hw_ctx_t *ctx) {
    if (ctx->ddr_vaddr && ctx->ddr_vaddr != MAP_FAILED)
        munmap(ctx->ddr_vaddr, ctx->ddr_size);
    if (ctx->dma_io_vaddr && ctx->dma_io_vaddr != MAP_FAILED)
        munmap(ctx->dma_io_vaddr, NNA_DMA_IOSIZE);
    if (ctx->desram_vaddr && ctx->desram_vaddr != MAP_FAILED)
        munmap(ctx->desram_vaddr, NNA_DESRAM_SIZE);
    if (ctx->oram_vaddr && ctx->oram_vaddr != MAP_FAILED)
        munmap(ctx->oram_vaddr, ctx->oram_size);
    if (ctx->fd >= 0) close(ctx->fd);
    if (ctx->memfd >= 0) close(ctx->memfd);
    memset(ctx, 0, sizeof(*ctx));
    ctx->fd = -1; ctx->memfd = -1;
}

uint32_t mars_nn_hw_oram_alloc(mars_nn_hw_ctx_t *ctx, uint32_t size, uint32_t align) {
    uint32_t ptr = (s_oram_alloc_ptr + align - 1) & ~(align - 1);
    if (ptr + size > ctx->oram_size) return (uint32_t)-1;
    s_oram_alloc_ptr = ptr + size;
    return ptr;
}

void mars_nn_hw_oram_reset(mars_nn_hw_ctx_t *ctx) {
    (void)ctx;
    s_oram_alloc_ptr = 0;
}

int mars_nn_hw_to_oram(mars_nn_hw_ctx_t *ctx, const void *src,
                       uint32_t oram_offset, uint32_t size) {
    if (oram_offset + size > ctx->oram_size) return -1;
    memcpy((char *)ctx->oram_vaddr + oram_offset, src, size);
    __sync_synchronize();
    return 0;
}

int mars_nn_hw_from_oram(mars_nn_hw_ctx_t *ctx, uint32_t oram_offset,
                         void *dst, uint32_t size) {
    if (oram_offset + size > ctx->oram_size) return -1;
    __sync_synchronize();
    memcpy(dst, (char *)ctx->oram_vaddr + oram_offset, size);
    return 0;
}

/* MXU vector operation using ORAM data */
int mars_nn_hw_mxu_vec_op(mars_nn_hw_ctx_t *ctx,
                          uint32_t dst_oram_offset,
                          uint32_t src_a_oram_offset,
                          uint32_t src_b_oram_offset,
                          uint32_t count_floats,
                          int op) {
#ifdef __mips__
    float *dst = (float *)((char *)ctx->oram_vaddr + dst_oram_offset);
    float *src_a = (float *)((char *)ctx->oram_vaddr + src_a_oram_offset);
    float *src_b = (float *)((char *)ctx->oram_vaddr + src_b_oram_offset);

    /* Process 16 floats (64 bytes = 1 VPR) at a time */
    size_t i = 0;
    for (; i + 16 <= count_floats; i += 16) {
        /* Load src_a to VPR0, src_b to VPR1 */
        LA0_VPR(0, src_a + i);
        LA0_VPR(1, src_b + i);

        /* Perform operation: VPR0 = VPR0 op VPR1 */
        switch (op) {
            case 0: VPR_ADD(0, 1); break;  /* add */
            case 1: VPR_MUL(0, 1); break;  /* mul */
            case 2: VPR_SUB(0, 1); break;  /* sub */
        }

        /* Store VPR0 to dst */
        SA0_VPR(0, dst + i);
    }

    /* Handle remaining elements with scalar */
    for (; i < count_floats; i++) {
        switch (op) {
            case 0: dst[i] = src_a[i] + src_b[i]; break;
            case 1: dst[i] = src_a[i] * src_b[i]; break;
            case 2: dst[i] = src_a[i] - src_b[i]; break;
        }
    }
    __sync_synchronize();
    return 0;
#else
    (void)ctx; (void)dst_oram_offset; (void)src_a_oram_offset;
    (void)src_b_oram_offset; (void)count_floats; (void)op;
    return -1;
#endif
}

/* ============================================================================
 * NNDMA Transfers
 * ============================================================================ */

/* DMA descriptor encoding */
#define DES_FLAG_SHIFT      50ULL
#define DES_LEN_SHIFT       40ULL
#define DES_ORAM_SHIFT      26ULL
#define DES_DDR_MASK        0x3FFFFFFULL

static inline uint64_t build_des_cnt(uint32_t total_bytes) {
    return (DES_FLAG_CNT << DES_FLAG_SHIFT) | (total_bytes & 0xFFFFF);
}

static inline uint64_t build_des_xfer(uint32_t ddr_addr, uint32_t oram_addr,
                                      uint32_t len, int is_last) {
    uint64_t flag = is_last ? DES_FLAG_END : DES_FLAG_LINK;
    uint64_t len_field = ((len >> 6) - 1) & 0x3FF;
    uint64_t oram_field = (oram_addr >> 6) & 0x3FFF;
    uint64_t ddr_field = (ddr_addr >> 6) & DES_DDR_MASK;
    return (flag << DES_FLAG_SHIFT) | (len_field << DES_LEN_SHIFT) |
           (oram_field << DES_ORAM_SHIFT) | ddr_field;
}

int mars_nn_hw_dma_to_oram(mars_nn_hw_ctx_t *ctx, uint32_t ddr_offset,
                           uint32_t oram_offset, uint32_t size) {
    if (size == 0 || (size & 63)) return -1;  /* Must be 64-byte aligned */

    volatile uint64_t *des = (volatile uint64_t *)ctx->desram_vaddr;
    uint32_t ddr_paddr = ctx->ddr_paddr + ddr_offset;
    uint32_t oram_paddr = ctx->oram_paddr + oram_offset;

    /* Build descriptor chain */
    des[0] = build_des_cnt(size);
    des[1] = build_des_xfer(ddr_paddr, oram_paddr, size, 1);
    __sync_synchronize();

    /* Start DMA read (DDR -> ORAM) */
    uint32_t des_idx = 0;
    if (ioctl(ctx->fd, IOCTL_NNA_RDCH, &des_idx) < 0) {
        perror("RDCH ioctl");
        return -1;
    }
    return 0;
}

int mars_nn_hw_dma_from_oram(mars_nn_hw_ctx_t *ctx, uint32_t oram_offset,
                             uint32_t ddr_offset, uint32_t size) {
    if (size == 0 || (size & 63)) return -1;

    volatile uint64_t *des = (volatile uint64_t *)ctx->desram_vaddr;
    uint32_t base_idx = NNA_DESRAM_SIZE / 16;  /* Use second half for writes */
    uint32_t ddr_paddr = ctx->ddr_paddr + ddr_offset;
    uint32_t oram_paddr = ctx->oram_paddr + oram_offset;

    des[base_idx] = build_des_cnt(size);
    des[base_idx + 1] = build_des_xfer(ddr_paddr, oram_paddr, size, 1);
    __sync_synchronize();

    uint32_t des_idx = base_idx;
    if (ioctl(ctx->fd, IOCTL_NNA_WRCH, &des_idx) < 0) {
        perror("WRCH ioctl");
        return -1;
    }
    return 0;
}

int mars_nn_hw_dma_wait(mars_nn_hw_ctx_t *ctx) {
    volatile uint32_t *io = (volatile uint32_t *)ctx->dma_io_vaddr;
    for (int i = 0; i < 1000000; i++) {
        __sync_synchronize();
        uint32_t rcfg = io[0];  /* RCFG */
        uint32_t wcfg = io[1];  /* WCFG */
        if ((rcfg & 1) == 0 && (wcfg & 1) == 0) return 0;
        for (volatile int j = 0; j < 10; j++);
    }
    return -1;  /* Timeout */
}

/* ============================================================================
 * MXU Extended Operations
 * ============================================================================ */

/* MXU dot product: sum(a[i] * b[i])
 * Optimized version: accumulates products in VPR2, reduces once at end */
float mars_nn_hw_mxu_dot(mars_nn_hw_ctx_t *ctx,
                         uint32_t src_a_oram_offset,
                         uint32_t src_b_oram_offset,
                         uint32_t count_floats) {
#ifdef __mips__
    float *src_a = (float *)((char *)ctx->oram_vaddr + src_a_oram_offset);
    float *src_b = (float *)((char *)ctx->oram_vaddr + src_b_oram_offset);

    /* Use VPR2 as accumulator - initialize to zero */
    float zeros[16] __attribute__((aligned(64))) = {0};
    LA0_VPR(2, zeros);  /* VPR2 = 0 (accumulator) */

    /* Process 16 floats at a time, accumulate in VPR2 */
    size_t i = 0;
    for (; i + 16 <= count_floats; i += 16) {
        LA0_VPR(0, src_a + i);  /* VPR0 = a[i:i+16] */
        LA0_VPR(1, src_b + i);  /* VPR1 = b[i:i+16] */
        VPR_MUL(0, 1);          /* VPR0 = VPR0 * VPR1 */
        VPR_ADD(2, 0);          /* VPR2 += VPR0 (accumulate) */
    }

    /* Single reduction at end: store VPR2 and sum 16 elements */
    float acc[16] __attribute__((aligned(64)));
    SA0_VPR(2, acc);
    float sum = acc[0] + acc[1] + acc[2] + acc[3] +
                acc[4] + acc[5] + acc[6] + acc[7] +
                acc[8] + acc[9] + acc[10] + acc[11] +
                acc[12] + acc[13] + acc[14] + acc[15];

    /* Scalar tail for remaining elements */
    for (; i < count_floats; i++) {
        sum += src_a[i] * src_b[i];
    }
    return sum;
#else
    (void)ctx; (void)src_a_oram_offset; (void)src_b_oram_offset; (void)count_floats;
    return 0.0f;
#endif
}

/* MXU fused multiply-add: dst = a * b + c */
int mars_nn_hw_mxu_fma(mars_nn_hw_ctx_t *ctx,
                       uint32_t dst_oram_offset,
                       uint32_t src_a_oram_offset,
                       uint32_t src_b_oram_offset,
                       uint32_t src_c_oram_offset,
                       uint32_t count_floats) {
#ifdef __mips__
    float *dst = (float *)((char *)ctx->oram_vaddr + dst_oram_offset);
    float *src_a = (float *)((char *)ctx->oram_vaddr + src_a_oram_offset);
    float *src_b = (float *)((char *)ctx->oram_vaddr + src_b_oram_offset);
    float *src_c = (float *)((char *)ctx->oram_vaddr + src_c_oram_offset);

    size_t i = 0;
    for (; i + 16 <= count_floats; i += 16) {
        LA0_VPR(0, src_a + i);
        LA0_VPR(1, src_b + i);
        LA0_VPR(2, src_c + i);
        VPR_MUL(0, 1);   /* VPR0 = a * b */
        VPR_ADD(0, 2);   /* VPR0 = (a * b) + c */
        SA0_VPR(0, dst + i);
    }

    /* Scalar tail */
    for (; i < count_floats; i++) {
        dst[i] = src_a[i] * src_b[i] + src_c[i];
    }
    __sync_synchronize();
    return 0;
#else
    (void)ctx; (void)dst_oram_offset; (void)src_a_oram_offset;
    (void)src_b_oram_offset; (void)src_c_oram_offset; (void)count_floats;
    return -1;
#endif
}

/* Convolution kernel - single output using dot product */
float mars_nn_hw_conv_kernel(mars_nn_hw_ctx_t *ctx,
                             uint32_t input_oram_offset,
                             uint32_t weight_oram_offset,
                             uint32_t kernel_size) {
    return mars_nn_hw_mxu_dot(ctx, input_oram_offset, weight_oram_offset, kernel_size);
}

/* ============================================================================
 * NN Layer Operations (Option C)
 * ============================================================================ */

/* ReLU: dst[i] = max(0, src[i]) - MXU accelerated with MAXSW */
int mars_nn_hw_relu(mars_nn_hw_ctx_t *ctx,
                    uint32_t dst_oram_offset,
                    uint32_t src_oram_offset,
                    uint32_t count_floats) {
#ifdef __mips__
    float *dst = (float *)((char *)ctx->oram_vaddr + dst_oram_offset);
    float *src = (float *)((char *)ctx->oram_vaddr + src_oram_offset);

    /* Initialize VPR1 with zeros for max(x, 0) */
    float zeros[16] __attribute__((aligned(64))) = {0};
    LA0_VPR(1, zeros);

    /* Process 16 floats at a time with MXU MAXSW */
    uint32_t i = 0;
    for (; i + 16 <= count_floats; i += 16) {
        LA0_VPR(0, src + i);        /* VPR0 = input */
        VPR_MAXSW(2, 0, 1);         /* VPR2 = max(VPR0, zeros) */
        SA0_VPR(2, dst + i);
    }

    /* Scalar tail */
    for (; i < count_floats; i++) {
        dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
    }
    __sync_synchronize();
    return 0;
#else
    (void)ctx; (void)dst_oram_offset; (void)src_oram_offset; (void)count_floats;
    return -1;
#endif
}

/* LeakyReLU: dst[i] = src[i] > 0 ? src[i] : alpha * src[i] */
int mars_nn_hw_leaky_relu(mars_nn_hw_ctx_t *ctx,
                          uint32_t dst_oram_offset,
                          uint32_t src_oram_offset,
                          uint32_t count_floats,
                          float alpha) {
    float *dst = (float *)((char *)ctx->oram_vaddr + dst_oram_offset);
    float *src = (float *)((char *)ctx->oram_vaddr + src_oram_offset);

    for (uint32_t i = 0; i < count_floats; i++) {
        float x = src[i];
        dst[i] = x > 0.0f ? x : alpha * x;
    }
    __sync_synchronize();
    return 0;
}

/* Batch normalization: dst = src * scale + bias (uses MXU FMA) */
int mars_nn_hw_batchnorm(mars_nn_hw_ctx_t *ctx,
                         uint32_t dst_oram_offset,
                         uint32_t src_oram_offset,
                         uint32_t scale_oram_offset,
                         uint32_t bias_oram_offset,
                         uint32_t count_floats) {
#ifdef __mips__
    float *dst = (float *)((char *)ctx->oram_vaddr + dst_oram_offset);
    float *src = (float *)((char *)ctx->oram_vaddr + src_oram_offset);
    float *scale = (float *)((char *)ctx->oram_vaddr + scale_oram_offset);
    float *bias = (float *)((char *)ctx->oram_vaddr + bias_oram_offset);

    /* Process 16 floats at a time with MXU */
    uint32_t i = 0;
    for (; i + 16 <= count_floats; i += 16) {
        LA0_VPR(0, src + i);    /* VPR0 = src */
        LA0_VPR(1, scale + i);  /* VPR1 = scale */
        LA0_VPR(2, bias + i);   /* VPR2 = bias */
        VPR_MUL(0, 1);          /* VPR0 = src * scale */
        VPR_ADD(0, 2);          /* VPR0 = src * scale + bias */
        SA0_VPR(0, dst + i);
    }

    /* Scalar tail */
    for (; i < count_floats; i++) {
        dst[i] = src[i] * scale[i] + bias[i];
    }
    __sync_synchronize();
    return 0;
#else
    (void)ctx; (void)dst_oram_offset; (void)src_oram_offset;
    (void)scale_oram_offset; (void)bias_oram_offset; (void)count_floats;
    return -1;
#endif
}

/* Max pooling 2x2 stride 2 - reduces HxW by half
 * Input layout: HWC (height, width, channels)
 * MXU accelerated when channels is multiple of 16 */
int mars_nn_hw_maxpool2x2(mars_nn_hw_ctx_t *ctx,
                          uint32_t dst_oram_offset,
                          uint32_t src_oram_offset,
                          uint32_t height, uint32_t width, uint32_t channels) {
#ifdef __mips__
    float *dst = (float *)((char *)ctx->oram_vaddr + dst_oram_offset);
    float *src = (float *)((char *)ctx->oram_vaddr + src_oram_offset);

    uint32_t out_h = height / 2;
    uint32_t out_w = width / 2;

    /* MXU path: when channels is multiple of 16, vectorize across channels */
    if ((channels % 16) == 0) {
        for (uint32_t oh = 0; oh < out_h; oh++) {
            for (uint32_t ow = 0; ow < out_w; ow++) {
                uint32_t ih = oh * 2;
                uint32_t iw = ow * 2;

                /* Process channels in chunks of 16 */
                for (uint32_t c = 0; c < channels; c += 16) {
                    float *p00 = &src[(ih * width + iw) * channels + c];
                    float *p01 = &src[(ih * width + iw + 1) * channels + c];
                    float *p10 = &src[((ih + 1) * width + iw) * channels + c];
                    float *p11 = &src[((ih + 1) * width + iw + 1) * channels + c];
                    float *pout = &dst[(oh * out_w + ow) * channels + c];

                    LA0_VPR(0, p00);       /* VPR0 = v00 */
                    LA0_VPR(1, p01);       /* VPR1 = v01 */
                    VPR_MAXSW(2, 0, 1);    /* VPR2 = max(v00, v01) */
                    LA0_VPR(0, p10);       /* VPR0 = v10 */
                    VPR_MAXSW(2, 2, 0);    /* VPR2 = max(VPR2, v10) */
                    LA0_VPR(0, p11);       /* VPR0 = v11 */
                    VPR_MAXSW(2, 2, 0);    /* VPR2 = max(VPR2, v11) */
                    SA0_VPR(2, pout);
                }
            }
        }
    } else {
        /* Scalar fallback for non-aligned channels */
        for (uint32_t oh = 0; oh < out_h; oh++) {
            for (uint32_t ow = 0; ow < out_w; ow++) {
                uint32_t ih = oh * 2;
                uint32_t iw = ow * 2;
                for (uint32_t c = 0; c < channels; c++) {
                    float v00 = src[(ih * width + iw) * channels + c];
                    float v01 = src[(ih * width + iw + 1) * channels + c];
                    float v10 = src[((ih + 1) * width + iw) * channels + c];
                    float v11 = src[((ih + 1) * width + iw + 1) * channels + c];

                    float max = v00;
                    if (v01 > max) max = v01;
                    if (v10 > max) max = v10;
                    if (v11 > max) max = v11;

                    dst[(oh * out_w + ow) * channels + c] = max;
                }
            }
        }
    }
    __sync_synchronize();
    return 0;
#else
    (void)ctx; (void)dst_oram_offset; (void)src_oram_offset;
    (void)height; (void)width; (void)channels;
    return -1;
#endif
}

/* Average pooling 2x2 stride 2 - uses MXU for averaging */
int mars_nn_hw_avgpool2x2(mars_nn_hw_ctx_t *ctx,
                          uint32_t dst_oram_offset,
                          uint32_t src_oram_offset,
                          uint32_t height, uint32_t width, uint32_t channels) {
    float *dst = (float *)((char *)ctx->oram_vaddr + dst_oram_offset);
    float *src = (float *)((char *)ctx->oram_vaddr + src_oram_offset);

    uint32_t out_h = height / 2;
    uint32_t out_w = width / 2;

    for (uint32_t oh = 0; oh < out_h; oh++) {
        for (uint32_t ow = 0; ow < out_w; ow++) {
            uint32_t ih = oh * 2;
            uint32_t iw = ow * 2;
            for (uint32_t c = 0; c < channels; c++) {
                float v00 = src[(ih * width + iw) * channels + c];
                float v01 = src[(ih * width + iw + 1) * channels + c];
                float v10 = src[((ih + 1) * width + iw) * channels + c];
                float v11 = src[((ih + 1) * width + iw + 1) * channels + c];

                dst[(oh * out_w + ow) * channels + c] = (v00 + v01 + v10 + v11) * 0.25f;
            }
        }
    }
    __sync_synchronize();
    return 0;
}

/* ============================================================================
 * Timing Helpers
 * ============================================================================ */

void mars_nn_timing_start(mars_nn_timing_t *t) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    t->start_cycles = mars_nn_get_cycles();
    t->elapsed_us = -(tv.tv_sec * 1000000.0 + tv.tv_usec);
}

void mars_nn_timing_end(mars_nn_timing_t *t) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    t->end_cycles = mars_nn_get_cycles();
    t->elapsed_us += (tv.tv_sec * 1000000.0 + tv.tv_usec);
}

