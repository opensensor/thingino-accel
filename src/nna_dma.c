/*
 * NNA DMA Implementation
 * Based on reverse-engineering of soc-nna kernel driver
 */

#include "nna_dma.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

/* IOCTL definitions (from soc_nna.h) */
#define SOC_NNA_MAGIC               'c'
#define IOCTL_SOC_NNA_MALLOC        _IOWR(SOC_NNA_MAGIC, 0, int)
#define IOCTL_SOC_NNA_FREE          _IOWR(SOC_NNA_MAGIC, 1, int)
#define IOCTL_SOC_NNA_FLUSHCACHE    _IOWR(SOC_NNA_MAGIC, 2, int)
#define IOCTL_SOC_NNA_SETUP_DES     _IOWR(SOC_NNA_MAGIC, 3, int)
#define IOCTL_SOC_NNA_RDCH_START    _IOWR(SOC_NNA_MAGIC, 4, int)
#define IOCTL_SOC_NNA_WRCH_START    _IOWR(SOC_NNA_MAGIC, 5, int)
#define IOCTL_SOC_NNA_VERSION       _IOWR(SOC_NNA_MAGIC, 6, int)

/* L2 cache offset for ORAM (from device.c) */
#define L2CACHE_SIZE_OFFSET         0x60
#define GPIO_BASE                   0x10010000

static uint32_t get_l2cache_offset(void) {
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) return 0x20000;  /* Default 128KB */
    
    void *gpio = mmap(NULL, 0x1000, PROT_READ, MAP_SHARED, fd, GPIO_BASE);
    if (gpio == MAP_FAILED) {
        close(fd);
        return 0x20000;
    }
    
    uint32_t val = *(volatile uint32_t *)((char *)gpio + L2CACHE_SIZE_OFFSET);
    munmap(gpio, 0x1000);
    close(fd);
    
    /* L2 cache size encoding: 0=128KB, 1=256KB, etc. */
    return (val & 0x3) * 0x20000 + 0x20000;
}

int nna_dma_init(nna_dma_ctx_t *ctx) {
    memset(ctx, 0, sizeof(*ctx));
    
    /* Open NNA device */
    ctx->fd = open("/dev/soc-nna", O_RDWR);
    if (ctx->fd < 0) {
        perror("nna_dma_init: open /dev/soc-nna");
        return -1;
    }
    
    /* Open /dev/mem for mapping */
    int memfd = open("/dev/mem", O_RDWR | O_SYNC);
    if (memfd < 0) {
        perror("nna_dma_init: open /dev/mem");
        close(ctx->fd);
        return -1;
    }
    
    /* Calculate ORAM physical address (base + L2 cache offset) */
    uint32_t l2_offset = get_l2cache_offset();
    ctx->oram_paddr = NNA_ORAM_BASE_ADDR + l2_offset;
    printf("NNA DMA: ORAM at 0x%08x (L2 offset: 0x%x)\n", ctx->oram_paddr, l2_offset);
    
    /* Map descriptor RAM */
    ctx->desram_vaddr = mmap(NULL, NNA_DMA_DESRAM_SIZE, PROT_READ | PROT_WRITE,
                             MAP_SHARED, memfd, NNA_DMA_DESRAM_ADDR);
    if (ctx->desram_vaddr == MAP_FAILED) {
        perror("nna_dma_init: mmap desram");
        close(memfd);
        close(ctx->fd);
        return -1;
    }
    
    /* Map DMA I/O registers */
    ctx->dma_io_vaddr = mmap(NULL, NNA_DMA_IOSIZE, PROT_READ | PROT_WRITE,
                             MAP_SHARED, memfd, NNA_DMA_IOBASE);
    if (ctx->dma_io_vaddr == MAP_FAILED) {
        perror("nna_dma_init: mmap dma_io");
        munmap(ctx->desram_vaddr, NNA_DMA_DESRAM_SIZE);
        close(memfd);
        close(ctx->fd);
        return -1;
    }
    
    /* Map ORAM */
    uint32_t oram_size = NNA_ORAM_BASE_SIZE - l2_offset;
    ctx->oram_vaddr = mmap(NULL, oram_size, PROT_READ | PROT_WRITE,
                           MAP_SHARED, memfd, ctx->oram_paddr);
    if (ctx->oram_vaddr == MAP_FAILED) {
        perror("nna_dma_init: mmap oram");
        munmap(ctx->dma_io_vaddr, NNA_DMA_IOSIZE);
        munmap(ctx->desram_vaddr, NNA_DMA_DESRAM_SIZE);
        close(memfd);
        close(ctx->fd);
        return -1;
    }
    
    close(memfd);
    
    printf("NNA DMA initialized:\n");
    printf("  DESRAM: %p (phys 0x%08x)\n", ctx->desram_vaddr, NNA_DMA_DESRAM_ADDR);
    printf("  DMA IO: %p (phys 0x%08x)\n", ctx->dma_io_vaddr, NNA_DMA_IOBASE);
    printf("  ORAM:   %p (phys 0x%08x, size %u KB)\n", 
           ctx->oram_vaddr, ctx->oram_paddr, oram_size / 1024);
    
    return 0;
}

void nna_dma_cleanup(nna_dma_ctx_t *ctx) {
    if (ctx->oram_vaddr && ctx->oram_vaddr != MAP_FAILED) {
        uint32_t l2_offset = ctx->oram_paddr - NNA_ORAM_BASE_ADDR;
        munmap(ctx->oram_vaddr, NNA_ORAM_BASE_SIZE - l2_offset);
    }
    if (ctx->dma_io_vaddr && ctx->dma_io_vaddr != MAP_FAILED)
        munmap(ctx->dma_io_vaddr, NNA_DMA_IOSIZE);
    if (ctx->desram_vaddr && ctx->desram_vaddr != MAP_FAILED)
        munmap(ctx->desram_vaddr, NNA_DMA_DESRAM_SIZE);
    if (ctx->fd >= 0)
        close(ctx->fd);
    memset(ctx, 0, sizeof(*ctx));
}

/* Write descriptor chain and start DMA read (DDR -> ORAM) using kernel ioctl */
int nna_dma_ddr_to_oram(nna_dma_ctx_t *ctx, uint32_t ddr_paddr,
                        uint32_t oram_offset, uint32_t size) {
    volatile uint64_t *des = (volatile uint64_t *)ctx->desram_vaddr;
    uint32_t oram_paddr = ctx->oram_paddr + oram_offset;

    printf("Building DMA descriptors:\n");
    printf("  DDR paddr: 0x%08x\n", ddr_paddr);
    printf("  ORAM paddr: 0x%08x (offset 0x%x)\n", oram_paddr, oram_offset);
    printf("  Size: %u bytes\n", size);

    /* Build descriptor chain in DESRAM */
    int idx = 0;
    uint64_t cnt_des = nna_des_cnt(size);
    des[idx++] = cnt_des;
    printf("  [0] CNT descriptor: 0x%016llx (total_bytes=%u)\n",
           (unsigned long long)cnt_des, size);

    uint32_t remaining = size;
    uint32_t ddr_off = 0;
    uint32_t oram_off = 0;

    while (remaining > 0) {
        uint32_t chunk = (remaining > NNA_MAX_TRANSFER) ? NNA_MAX_TRANSFER : remaining;
        int is_last = (remaining <= NNA_MAX_TRANSFER);
        uint64_t xfer_des = nna_des_transfer(ddr_paddr + ddr_off, oram_paddr + oram_off,
                                              chunk, is_last);
        des[idx] = xfer_des;
        printf("  [%d] %s descriptor: 0x%016llx (ddr=0x%08x, oram=0x%08x, len=%u)\n",
               idx, is_last ? "END" : "LINK", (unsigned long long)xfer_des,
               ddr_paddr + ddr_off, oram_paddr + oram_off, chunk);
        idx++;
        ddr_off += chunk;
        oram_off += chunk;
        remaining -= chunk;
    }

    /* Memory barrier to ensure descriptors are written */
    __sync_synchronize();

    /* Start DMA read via kernel ioctl */
    uint32_t des_idx = 0;  /* Descriptor index in DESRAM (in 8-byte units) */
    printf("Starting DMA read at descriptor index %u\n", des_idx);
    int ret = ioctl(ctx->fd, IOCTL_SOC_NNA_RDCH_START, &des_idx);
    if (ret < 0) {
        perror("ioctl RDCH_START");
        return -1;
    }

    return 0;
}

/* Write descriptor chain and start DMA write (ORAM -> DDR) using kernel ioctl */
int nna_dma_oram_to_ddr(nna_dma_ctx_t *ctx, uint32_t oram_offset,
                        uint32_t ddr_paddr, uint32_t size) {
    volatile uint64_t *des = (volatile uint64_t *)ctx->desram_vaddr;
    uint32_t oram_paddr = ctx->oram_paddr + oram_offset;

    /* Use second half of DESRAM for write descriptors */
    int base_idx = NNA_DMA_DESRAM_SIZE / 16;  /* 8 bytes per descriptor */
    int idx = base_idx;

    des[idx++] = nna_des_cnt(size);  /* Count descriptor */

    uint32_t remaining = size;
    uint32_t ddr_off = 0;
    uint32_t oram_off = 0;

    while (remaining > 0) {
        uint32_t chunk = (remaining > NNA_MAX_TRANSFER) ? NNA_MAX_TRANSFER : remaining;
        int is_last = (remaining <= NNA_MAX_TRANSFER);
        des[idx++] = nna_des_transfer(ddr_paddr + ddr_off, oram_paddr + oram_off,
                                       chunk, is_last);
        ddr_off += chunk;
        oram_off += chunk;
        remaining -= chunk;
    }

    /* Memory barrier to ensure descriptors are written */
    __sync_synchronize();

    /* Start DMA write via kernel ioctl */
    uint32_t des_idx = base_idx;  /* Descriptor index in DESRAM (in 8-byte units) */
    int ret = ioctl(ctx->fd, IOCTL_SOC_NNA_WRCH_START, &des_idx);
    if (ret < 0) {
        perror("ioctl WRCH_START");
        return -1;
    }

    return 0;
}

/* Wait for DMA completion by polling count registers */
int nna_dma_wait(nna_dma_ctx_t *ctx) {
    volatile uint32_t *io = (volatile uint32_t *)ctx->dma_io_vaddr;
    int timeout = 10000000;  /* 10 seconds at ~1us per iteration */

    /* Memory barrier before reading */
    __sync_synchronize();

    while (timeout-- > 0) {
        uint32_t rcfg = io[NNA_DMA_RCFG / 4];
        uint32_t wcfg = io[NNA_DMA_WCFG / 4];
        uint32_t rcnt = io[NNA_DMA_RCNT / 4];
        uint32_t wcnt = io[NNA_DMA_WCNT / 4];

        /* Debug: print register values periodically */
        if (timeout % 1000000 == 0) {
            printf("DMA wait: RCFG=0x%08x WCFG=0x%08x RCNT=0x%08x WCNT=0x%08x\n",
                   rcfg, wcfg, rcnt, wcnt);
        }

        /* Check if DMA is not running (START bit cleared) and count is 0 */
        if ((rcfg & 1) == 0 && (wcfg & 1) == 0) {
            return 0;
        }

        /* Small delay */
        for (volatile int i = 0; i < 10; i++);
    }

    fprintf(stderr, "nna_dma_wait: timeout\n");
    return -1;
}

