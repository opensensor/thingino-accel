/*
 * NNA DMA Test
 * Tests DDR <-> ORAM transfers using the NNA DMA engine
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#include "nna_dma.h"

/* Allocate DMA-capable memory via kernel */
#define SOC_NNA_MAGIC               'c'
#define IOCTL_SOC_NNA_MALLOC        _IOWR(SOC_NNA_MAGIC, 0, int)
#define IOCTL_SOC_NNA_FREE          _IOWR(SOC_NNA_MAGIC, 1, int)
#define IOCTL_SOC_NNA_FLUSHCACHE    _IOWR(SOC_NNA_MAGIC, 2, int)

struct soc_nna_buf {
    void *vaddr;
    void *paddr;
    int size;
};

struct flush_cache_info {
    unsigned int addr;
    unsigned int len;
    unsigned int dir;
};

int main(int argc, char **argv) {
    nna_dma_ctx_t ctx;
    struct soc_nna_buf buf;
    int ret;
    
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  NNA DMA Test                                            ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");
    
    /* Initialize NNA DMA */
    ret = nna_dma_init(&ctx);
    if (ret < 0) {
        fprintf(stderr, "Failed to initialize NNA DMA\n");
        return 1;
    }
    
    /* Allocate DMA buffer in DDR */
    buf.size = 4096;  /* 4KB test buffer */
    buf.vaddr = NULL;
    buf.paddr = NULL;
    
    ret = ioctl(ctx.fd, IOCTL_SOC_NNA_MALLOC, &buf);
    if (ret < 0) {
        perror("ioctl MALLOC");
        nna_dma_cleanup(&ctx);
        return 1;
    }
    
    printf("Allocated DDR buffer:\n");
    printf("  vaddr: %p\n", buf.vaddr);
    printf("  paddr: %p\n", buf.paddr);
    printf("  size:  %d bytes\n\n", buf.size);
    
    /* Map the DDR buffer to userspace */
    int memfd = open("/dev/mem", O_RDWR | O_SYNC);
    if (memfd < 0) {
        perror("open /dev/mem");
        ioctl(ctx.fd, IOCTL_SOC_NNA_FREE, &buf);
        nna_dma_cleanup(&ctx);
        return 1;
    }
    
    void *ddr_vaddr = mmap(NULL, buf.size, PROT_READ | PROT_WRITE,
                           MAP_SHARED, memfd, (uint32_t)buf.paddr);
    close(memfd);
    
    if (ddr_vaddr == MAP_FAILED) {
        perror("mmap DDR buffer");
        ioctl(ctx.fd, IOCTL_SOC_NNA_FREE, &buf);
        nna_dma_cleanup(&ctx);
        return 1;
    }
    
    printf("Mapped DDR buffer to %p\n\n", ddr_vaddr);
    
    /* Fill DDR buffer with test pattern */
    printf("Filling DDR buffer with test pattern...\n");
    uint32_t *ddr_data = (uint32_t *)ddr_vaddr;
    for (int i = 0; i < buf.size / 4; i++) {
        ddr_data[i] = 0xDEAD0000 | i;
    }
    
    /* Flush cache */
    struct flush_cache_info flush = {
        .addr = (uint32_t)ddr_vaddr,
        .len = buf.size,
        .dir = 1  /* DMA_TO_DEVICE */
    };
    ioctl(ctx.fd, IOCTL_SOC_NNA_FLUSHCACHE, &flush);
    
    /* Clear ORAM */
    printf("Clearing ORAM...\n");
    memset(ctx.oram_vaddr, 0, buf.size);
    
    /* Transfer DDR -> ORAM */
    printf("Transferring DDR -> ORAM (%d bytes)...\n", buf.size);
    ret = nna_dma_ddr_to_oram(&ctx, (uint32_t)buf.paddr, 0, buf.size);
    if (ret < 0) {
        fprintf(stderr, "DMA DDR->ORAM failed\n");
        goto cleanup;
    }
    
    /* Wait for completion */
    ret = nna_dma_wait(&ctx);
    if (ret < 0) {
        fprintf(stderr, "DMA wait failed\n");
        goto cleanup;
    }
    
    printf("DMA complete!\n\n");
    
    /* Verify ORAM contents */
    printf("Verifying ORAM contents...\n");
    uint32_t *oram_data = (uint32_t *)ctx.oram_vaddr;
    int errors = 0;
    for (int i = 0; i < buf.size / 4; i++) {
        if (oram_data[i] != ddr_data[i]) {
            if (errors < 10) {
                printf("  Mismatch at %d: expected 0x%08x, got 0x%08x\n",
                       i, ddr_data[i], oram_data[i]);
            }
            errors++;
        }
    }
    
    if (errors == 0) {
        printf("  ✓ All %d words match!\n", buf.size / 4);
    } else {
        printf("  ✗ %d errors out of %d words\n", errors, buf.size / 4);
    }
    
cleanup:
    munmap(ddr_vaddr, buf.size);
    ioctl(ctx.fd, IOCTL_SOC_NNA_FREE, &buf);
    nna_dma_cleanup(&ctx);
    
    return errors ? 1 : 0;
}

