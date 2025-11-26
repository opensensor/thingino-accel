/*
 * thingino-accel - Memory management implementation
 * 
 * DDR and ORAM allocation based on reverse-engineered libdrivers.so
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "nna.h"
#include "nna_memory.h"
#include "device_internal.h"

/* Alignment for DMA operations */
#define NNA_ALIGNMENT 64

/* IOCTL commands from soc-nna.ko */
#define SOC_NNA_MAGIC 'c'
#define IOCTL_SOC_NNA_MALLOC     _IOWR(SOC_NNA_MAGIC, 0, int)
#define IOCTL_SOC_NNA_FREE       _IOWR(SOC_NNA_MAGIC, 1, int)
#define IOCTL_SOC_NNA_FLUSHCACHE _IOWR(SOC_NNA_MAGIC, 2, int)

/* Memory buffer structure (from soc_nna.h) */
struct soc_nna_buf {
    void *vaddr;
    void *paddr;
    int size;
};

/* Track allocated buffers for proper cleanup */
struct mem_block {
    void *user_vaddr;      /* Userspace virtual address (from mmap) */
    void *kernel_vaddr;    /* Kernel virtual address (from ioctl) */
    void *paddr;           /* Physical address */
    size_t size;           /* Allocation size */
    struct mem_block *next;
};

static struct mem_block *g_mem_list = NULL;

/* Simple ORAM allocator state - ORAM not directly accessible */
static struct {
    size_t total;      /* Total ORAM size */
    size_t used;       /* Used ORAM size (simulated) */
    int initialized;   /* Initialization flag */
} g_oram = {
    .total = 0,
    .used = 0,
    .initialized = 0,
};

/* Initialize ORAM allocator */
static int oram_init(void) {
    if (g_oram.initialized) {
        return 0;
    }

    /* Get ORAM size from hardware info */
    nna_hw_info_t hw_info;
    if (nna_get_hw_info(&hw_info) != NNA_SUCCESS) {
        return -1;
    }

    g_oram.total = hw_info.oram_size;
    g_oram.used = 0;
    g_oram.initialized = 1;

    return 0;
}

void* nna_malloc(size_t size) {
    int fd = nna_device_get_fd();
    if (fd < 0) {
        fprintf(stderr, "nna_malloc: NNA not initialized\n");
        return NULL;
    }

    /* Allocate DMA memory through kernel driver */
    struct soc_nna_buf buf = {0};
    buf.size = size;

    if (ioctl(fd, IOCTL_SOC_NNA_MALLOC, &buf) < 0) {
        fprintf(stderr, "nna_malloc: IOCTL_SOC_NNA_MALLOC failed: %s\n",
                strerror(errno));
        return NULL;
    }

    /*
     * The kernel returns:
     * - buf.vaddr: kernel virtual address (not usable from userspace)
     * - buf.paddr: physical address
     * - buf.size: allocated size (page-aligned)
     *
     * We need to mmap the physical address to get a userspace virtual address.
     * Use /dev/mem for mapping nmem allocations (regular RAM).
     */
    int memfd = nna_device_get_memfd();
    if (memfd < 0) {
        fprintf(stderr, "nna_malloc: /dev/mem not available\n");
        ioctl(fd, IOCTL_SOC_NNA_FREE, &buf);
        return NULL;
    }

    void *mapped = mmap(NULL, buf.size, PROT_READ | PROT_WRITE,
                        MAP_SHARED, memfd, (off_t)(uintptr_t)buf.paddr);
    if (mapped == MAP_FAILED) {
        fprintf(stderr, "nna_malloc: mmap failed for paddr=%p size=%d: %s\n",
                buf.paddr, buf.size, strerror(errno));
        /* Free the kernel allocation */
        ioctl(fd, IOCTL_SOC_NNA_FREE, &buf);
        return NULL;
    }

    /* Track this allocation */
    struct mem_block *block = malloc(sizeof(struct mem_block));
    if (block == NULL) {
        fprintf(stderr, "nna_malloc: Failed to allocate tracking block\n");
        munmap(mapped, buf.size);
        ioctl(fd, IOCTL_SOC_NNA_FREE, &buf);
        return NULL;
    }

    block->user_vaddr = mapped;
    block->kernel_vaddr = buf.vaddr;
    block->paddr = buf.paddr;
    block->size = buf.size;
    block->next = g_mem_list;
    g_mem_list = block;

    return mapped;
}

void* nna_memalign(size_t alignment, size_t size) {
    /* The kernel driver always returns aligned memory, so just use nna_malloc */
    (void)alignment;  /* Unused */
    return nna_malloc(size);
}

void* nna_calloc(size_t nmemb, size_t size) {
    size_t total = nmemb * size;
    void *ptr = nna_malloc(total);

    if (ptr != NULL) {
        memset(ptr, 0, total);
    }

    return ptr;
}

void nna_free(void *ptr) {
    if (ptr == NULL) {
        return;
    }

    int fd = nna_device_get_fd();
    if (fd < 0) {
        fprintf(stderr, "nna_free: NNA not initialized\n");
        return;
    }

    /* Find the allocation in our tracking list */
    struct mem_block **prev = &g_mem_list;
    struct mem_block *block = g_mem_list;

    while (block != NULL) {
        if (block->user_vaddr == ptr) {
            /* Found it - unmap and free */
            munmap(block->user_vaddr, block->size);

            /* Free through kernel driver using kernel vaddr */
            struct soc_nna_buf buf = {0};
            buf.vaddr = block->kernel_vaddr;
            buf.paddr = block->paddr;
            buf.size = block->size;

            if (ioctl(fd, IOCTL_SOC_NNA_FREE, &buf) < 0) {
                fprintf(stderr, "nna_free: IOCTL_SOC_NNA_FREE failed: %s\n",
                        strerror(errno));
            }

            /* Remove from list */
            *prev = block->next;
            free(block);
            return;
        }
        prev = &block->next;
        block = block->next;
    }

    fprintf(stderr, "nna_free: Pointer %p not found in allocation list\n", ptr);
}

void* nna_oram_malloc(size_t size) {
    if (oram_init() != 0) {
        return NULL;
    }

    /* Align size to 64 bytes */
    size_t aligned_size = (size + NNA_ALIGNMENT - 1) & ~(NNA_ALIGNMENT - 1);

    /* Check if we have enough space */
    if (g_oram.used + aligned_size > g_oram.total) {
        fprintf(stderr, "nna_oram_malloc: Out of ORAM memory "
                "(requested %zu, available %zu)\n",
                aligned_size, g_oram.total - g_oram.used);
        return NULL;
    }

    /*
     * ORAM is not directly accessible from userspace.
     * This is a placeholder that tracks ORAM usage for statistics.
     * Actual ORAM management is done by the kernel driver and hardware.
     */
    g_oram.used += aligned_size;

    /* Return a dummy pointer to indicate success */
    return (void*)0x1;
}

void nna_oram_free(void *ptr) {
    /* 
     * Simple bump allocator doesn't support individual frees.
     * In a real implementation, we'd need a proper allocator
     * (e.g., buddy allocator or free list).
     * 
     * For now, ORAM is reset when NNA is deinitialized.
     */
    (void)ptr;
}

int nna_oram_get_stats(size_t *total, size_t *used, size_t *free) {
    if (oram_init() != 0) {
        return NNA_ERROR_INIT;
    }

    if (total) *total = g_oram.total;
    if (used) *used = g_oram.used;
    if (free) *free = g_oram.total - g_oram.used;

    return NNA_SUCCESS;
}

void nna_cache_flush(void *ptr, size_t size) {
    /*
     * On MIPS, we need to flush the cache before DMA reads.
     * This is typically done via a syscall or special instruction.
     * 
     * For now, this is a placeholder. Real implementation would use:
     * - cacheflush() syscall on MIPS
     * - Or direct cache instructions
     */
    (void)ptr;
    (void)size;
    
    /* TODO: Implement MIPS cache flush */
}

void nna_cache_invalidate(void *ptr, size_t size) {
    /*
     * On MIPS, we need to invalidate the cache before CPU reads
     * data written by DMA.
     * 
     * For now, this is a placeholder.
     */
    (void)ptr;
    (void)size;
    
    /* TODO: Implement MIPS cache invalidate */
}

