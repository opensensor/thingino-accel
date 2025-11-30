/*
 * AIP (AI Processor) user-level wrapper
 *
 * Thin helper around /dev/jzaip_* devices and AIP-F registers.
 * Intended as a minimal, single-node convolution interface that
 * mirrors the behavior observed in tools/aip_lib_test.c and
 * libaip.so HLIL.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <errno.h>

#include "aip.h"

static inline void aip_reg_write(volatile uint32_t *io, uint32_t offset, uint32_t val) {
    io[offset / 4] = val;
}

static inline uint32_t aip_reg_read(volatile uint32_t *io, uint32_t offset) {
    return io[offset / 4];
}

int aip_init(aip_ctx_t *ctx) {
    if (!ctx) {
        errno = EINVAL;
        return -1;
    }

    memset(ctx, 0, sizeof(*ctx));
    ctx->fd_f = ctx->fd_p = ctx->fd_t = ctx->fd_mem = -1;

    ctx->fd_f = open(AIP_DEV_F, O_RDWR);
    if (ctx->fd_f < 0) {
        perror("aip_init: open " AIP_DEV_F);
        goto fail;
    }

    ctx->fd_p = open(AIP_DEV_P, O_RDWR);
    if (ctx->fd_p < 0) {
        perror("aip_init: open " AIP_DEV_P);
        goto fail;
    }

    ctx->fd_t = open(AIP_DEV_T, O_RDWR);
    if (ctx->fd_t < 0) {
        perror("aip_init: open " AIP_DEV_T);
        goto fail;
    }

    ctx->fd_mem = open("/dev/mem", O_RDWR | O_SYNC);
    if (ctx->fd_mem < 0) {
        perror("aip_init: open /dev/mem");
        goto fail;
    }

    ctx->io = mmap(NULL, AIP_IOSIZE, PROT_READ | PROT_WRITE,
                   MAP_SHARED, ctx->fd_mem, AIP_IOBASE);
    if (ctx->io == MAP_FAILED) {
        perror("aip_init: mmap AIP_IOBASE");
        ctx->io = NULL;
        goto fail;
    }

    /* Optional: allocate a chain buffer. Not strictly required for
     * single-node register-driven mode, but useful for future
     * experimentation and closer to libaip behavior. */
    aip_buf_t buf = { .vaddr = NULL, .paddr = NULL, .size = 0x36000 };
    if (ioctl(ctx->fd_f, IOCTL_AIP_MALLOC, &buf) >= 0) {
        ctx->chainbuf_f_paddr = (uint32_t)(uintptr_t)buf.paddr;
        ctx->chainbuf_f_size = (uint32_t)buf.size;
        ctx->chainbuf_f = mmap(NULL, buf.size, PROT_READ | PROT_WRITE,
                               MAP_SHARED, ctx->fd_f,
                               (off_t)(uintptr_t)buf.paddr);
        if (ctx->chainbuf_f == MAP_FAILED) {
            perror("aip_init: mmap chain buffer");
            ctx->chainbuf_f = NULL;
        }
    } else {
        perror("aip_init: IOCTL_AIP_MALLOC");
    }

    return 0;

fail:
    aip_cleanup(ctx);
    return -1;
}

void aip_cleanup(aip_ctx_t *ctx) {
    if (!ctx) return;

    if (ctx->chainbuf_f && ctx->chainbuf_f_size) {
        munmap(ctx->chainbuf_f, ctx->chainbuf_f_size);
        ctx->chainbuf_f = NULL;
    }

    if (ctx->fd_f >= 0 && ctx->chainbuf_f_size) {
        /* Driver ignores the argument; tests pass NULL here. */
        ioctl(ctx->fd_f, IOCTL_AIP_FREE, 0);
    }

    if (ctx->io) {
        munmap((void *)ctx->io, AIP_IOSIZE);
        ctx->io = NULL;
    }

    if (ctx->fd_mem >= 0) {
        close(ctx->fd_mem);
        ctx->fd_mem = -1;
    }
    if (ctx->fd_f >= 0) {
        close(ctx->fd_f);
        ctx->fd_f = -1;
    }
    if (ctx->fd_t >= 0) {
        close(ctx->fd_t);
        ctx->fd_t = -1;
    }
    if (ctx->fd_p >= 0) {
        close(ctx->fd_p);
        ctx->fd_p = -1;
    }
}

int aip_f_wait(aip_ctx_t *ctx) {
    if (!ctx || ctx->fd_f < 0) {
        errno = EINVAL;
        return -1;
    }

    int status = 0;
    int ret = ioctl(ctx->fd_f, IOCTL_AIP_IRQ_WAIT_CMP, &status);
    if (ret < 0) {
        perror("aip_f_wait: IOCTL_AIP_IRQ_WAIT_CMP");
        return ret;
    }

    return status;
}

int aip_conv2d(aip_ctx_t *ctx,
               uint32_t in_addr, uint32_t out_addr,
               uint32_t kernel_addr, uint32_t bias_addr,
               uint32_t in_w, uint32_t in_h, uint32_t in_c,
               uint32_t out_w, uint32_t out_h, uint32_t out_c,
               uint32_t kernel_w, uint32_t kernel_h,
               uint32_t stride, uint32_t pad) {
    if (!ctx || !ctx->io) {
        errno = EINVAL;
        return -1;
    }

    volatile uint32_t *io = ctx->io;

    /* Reset AIP-F: write bit 1, wait for it to clear (aip_f_reset). */
    aip_reg_write(io, AIP_F_CTRL, 0x02);
    for (int i = 0; i < 1000; ++i) {
        uint32_t ctrl = aip_reg_read(io, AIP_F_CTRL);
        if ((ctrl & 0x02) == 0)
            break;
        usleep(10);
    }

    uint32_t in_size = (in_h << 16) | in_w;
    uint32_t out_size = (out_h << 16) | out_w;
    uint32_t kern_size = (kernel_h << 16) | kernel_w;
    uint32_t stride_hw = (stride << 16) | stride;
    uint32_t ch = (out_c << 16) | in_c;
    uint32_t pad_pool = pad & 0xFFFF;       /* no pooling */
    uint32_t scale_shift = (0 << 16) | 1;   /* shift=0, scale=1 */

    aip_reg_write(io, AIP_F_NODECFG, 0xFFFFFFFF);
    aip_reg_write(io, AIP_F_IN_ADDR, in_addr);
    aip_reg_write(io, AIP_F_OUT_ADDR, out_addr);
    aip_reg_write(io, AIP_F_IN_SIZE, in_size + 0x10);
    aip_reg_write(io, AIP_F_OUT_SIZE, out_size + 0x10);
    aip_reg_write(io, AIP_F_KERNEL_ADDR, kernel_addr);
    aip_reg_write(io, AIP_F_BIAS_ADDR, bias_addr);
    aip_reg_write(io, AIP_F_KERNEL_SIZE, kern_size);
    aip_reg_write(io, AIP_F_STRIDE, stride_hw);
    aip_reg_write(io, AIP_F_IN_CH_OUT_CH, ch);
    aip_reg_write(io, AIP_F_PAD_POOL, pad_pool);
    aip_reg_write(io, AIP_F_SCALE_SHIFT, scale_shift);

    /* Single-node mode: CFG = 0x14 (from decompiled aip_f_init). */
    aip_reg_write(io, AIP_F_CFG, 0x14);

    __sync_synchronize();

    uint32_t ctrl_before = aip_reg_read(io, AIP_F_CTRL);
    aip_reg_write(io, AIP_F_CTRL, ctrl_before | 0x01);  /* start */
    __sync_synchronize();

    return aip_f_wait(ctx);
}

