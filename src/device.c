/*
 * thingino-accel - Open Source NNA Library for Ingenic T41/T31
 * 
 * NNA device interface - /dev/soc-nna interaction
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <errno.h>

#include "nna.h"
#include "nna_runtime.h"
#include "device_internal.h"

/* Device path */
#define NNA_DEVICE_PATH "/dev/soc-nna"

/* IOCTL commands (from soc-nna.ko kernel driver) */
#define SOC_NNA_MAGIC 'c'
#define IOCTL_SOC_NNA_MALLOC     _IOWR(SOC_NNA_MAGIC, 0, int)
#define IOCTL_SOC_NNA_FREE       _IOWR(SOC_NNA_MAGIC, 1, int)
#define IOCTL_SOC_NNA_FLUSHCACHE _IOWR(SOC_NNA_MAGIC, 2, int)
#define IOCTL_SOC_NNA_SETUP_DES  _IOWR(SOC_NNA_MAGIC, 3, int)
#define IOCTL_SOC_NNA_RDCH_START _IOWR(SOC_NNA_MAGIC, 4, int)
#define IOCTL_SOC_NNA_WRCH_START _IOWR(SOC_NNA_MAGIC, 5, int)
#define IOCTL_SOC_NNA_VERSION    _IOWR(SOC_NNA_MAGIC, 6, int)

/* Hardware register addresses (from OEM libdrivers.so __aie_mmap) */
#define L2CACHE_SIZE_PADDR   0x12200000  /* L2 cache size register */
#define L2CACHE_SIZE_OFFSET  0x60        /* GPIO register offset */
#define ORAM_BASE_ADDR       0x12600000  /* Base ORAM address (before L2 offset) */
#define NNDMA_IO_PADDR       0x12508000  /* NNA DMA I/O registers */
#define NNDMA_IO_SIZE        0x20        /* 32 bytes */
#define NNDMA_DESRAM_PADDR   0x12500000  /* NNA DMA descriptor RAM */
#define NNDMA_DESRAM_SIZE    0x8000      /* 32KB (from OEM strerror constant) */

/* Memory buffer structure */
struct soc_nna_buf {
    void *vaddr;
    void *paddr;
    int size;
};

/* Parse nmem info from /proc/cmdline (from OEM libdrivers.so) */
static int get_nmem_info(uint32_t *size, uint32_t *paddr) {
    char cmdline[4096];
    FILE *fp = fopen("/proc/cmdline", "r");

    if (fp == NULL) {
        fprintf(stderr, "get_nmem_info: open file (/proc/cmdline) error\n");
        return -1;
    }

    size_t read_size = fread(cmdline, 1, sizeof(cmdline), fp);
    if (read_size == 0) {
        fprintf(stderr, "get_nmem_info: fread (/proc/cmdline) error\n");
        fclose(fp);
        return -1;
    }
    cmdline[read_size] = '\0';

    /* Find "nmem=" in cmdline */
    char *nmem_str = strstr(cmdline, "nmem");
    if (nmem_str == NULL) {
        fprintf(stderr, "get_nmem_info: nmem not found in cmdline\n");
        fclose(fp);
        return -1;
    }

    /* Parse nmem=<size>M@<addr> or nmem=<size>K@<addr> */
    uint32_t nmem_size = 0;
    uint32_t nmem_addr = 0;
    char *at_sign = strchr(nmem_str, '@');

    if (at_sign != NULL && *(at_sign - 1) == 'M') {
        /* Size in MB */
        if (sscanf(nmem_str, "nmem=%uM@%x", &nmem_size, &nmem_addr) == 2) {
            nmem_size = nmem_size << 20;  /* Convert MB to bytes */
        }
    } else if (at_sign != NULL && *(at_sign - 1) == 'K') {
        /* Size in KB */
        if (sscanf(nmem_str, "nmem=%uK@%x", &nmem_size, &nmem_addr) == 2) {
            nmem_size = nmem_size << 10;  /* Convert KB to bytes */
        }
    }

    fclose(fp);

    if (nmem_addr == 0 || nmem_size == 0) {
        fprintf(stderr, "CMD Line Nmem Size:%u, Addr:0x%08x is invalid\n", nmem_size, nmem_addr);
        return -1;
    }

    *size = nmem_size;
    *paddr = nmem_addr;
    return 0;
}

/* Global device state */
static struct {
    int fd;                     /* NNA device file descriptor */
    int memfd;                  /* /dev/mem file descriptor */
    int initialized;            /* Initialization flag */
    void *oram_mapped;          /* Mapped ORAM pointer */
    uint32_t oram_pbase;        /* ORAM physical address */
    uint32_t oram_size;         /* ORAM size */
    void *l2cache_size_vaddr;   /* L2 cache size register mapping */
    void *nndma_io_mapped;      /* NNA DMA I/O registers */
    void *nndma_desram_mapped;  /* NNA DMA descriptor RAM */
    void *ddr_mapped;           /* DDR DMA memory */
    uint32_t ddr_pbase;         /* DDR physical address */
    uint32_t ddr_size;          /* DDR size */
} g_nna_dev = {
    .fd = -1,
    .memfd = -1,
    .initialized = 0,
    .oram_mapped = NULL,
    .oram_pbase = 0,
    .oram_size = 0,
    .l2cache_size_vaddr = NULL,
    .nndma_io_mapped = NULL,
    .nndma_desram_mapped = NULL,
    .ddr_mapped = NULL,
    .ddr_pbase = 0,
    .ddr_size = 0,
};

int nna_init(void) {
    if (g_nna_dev.initialized) {
        return NNA_SUCCESS;
    }

    /* Open /dev/mem for physical memory mapping */
    g_nna_dev.memfd = open("/dev/mem", O_RDWR | O_SYNC);
    if (g_nna_dev.memfd < 0) {
        fprintf(stderr, "Error: /dev/mem open failed: %s\n", strerror(errno));
        return NNA_ERROR_DEVICE;
    }

    /* Open NNA device */
    g_nna_dev.fd = open(NNA_DEVICE_PATH, O_RDWR);
    if (g_nna_dev.fd < 0) {
        fprintf(stderr, "Error: /dev/soc-nna open failed: %s\n", strerror(errno));
        close(g_nna_dev.memfd);
        g_nna_dev.memfd = -1;
        return NNA_ERROR_DEVICE;
    }

    /* Step 1: Map L2 cache size register to determine ORAM base */
    g_nna_dev.l2cache_size_vaddr = mmap(NULL, 0x1000, PROT_READ | PROT_WRITE,
                                         MAP_SHARED, g_nna_dev.memfd, L2CACHE_SIZE_PADDR);
    if (g_nna_dev.l2cache_size_vaddr == MAP_FAILED) {
        fprintf(stderr, "Error: get l2cache_size_vaddr failed\n");
        goto cleanup_devices;
    }

    /* Read L2 cache size from GPIO register at offset 0x60 */
    uint32_t l2cache_gpio = *(volatile uint32_t*)((char*)g_nna_dev.l2cache_size_vaddr + L2CACHE_SIZE_OFFSET);
    uint32_t l2cache_bits = (l2cache_gpio >> 10) & 0x7;
    uint32_t l2cache_size_val;

    /* Decode L2 cache size (from OEM logic) */
    switch (l2cache_bits) {
        case 1: l2cache_size_val = 0x20000; break;   /* 128KB */
        case 2: l2cache_size_val = 0x40000; break;   /* 256KB */
        case 3: l2cache_size_val = 0x80000; break;   /* 512KB */
        case 4: l2cache_size_val = 0x100000; break;  /* 1MB */
        default: l2cache_size_val = 0x40000; break;  /* Default 256KB */
    }

    /* Calculate ORAM base and size */
    g_nna_dev.oram_pbase = ORAM_BASE_ADDR + l2cache_size_val;
    g_nna_dev.oram_size = 0x80000 - l2cache_size_val;  /* Total 512KB - L2 size */

    printf("L2 cache: %u KB, ORAM base: 0x%08x, ORAM size: %u KB\n",
           l2cache_size_val / 1024, g_nna_dev.oram_pbase, g_nna_dev.oram_size / 1024);

    /* Step 2: Map NNA DMA I/O registers */
    g_nna_dev.nndma_io_mapped = mmap(NULL, NNDMA_IO_SIZE, PROT_READ | PROT_WRITE,
                                      MAP_SHARED, g_nna_dev.memfd, NNDMA_IO_PADDR);
    if (g_nna_dev.nndma_io_mapped == MAP_FAILED) {
        fprintf(stderr, "Error: NNDMA IO mmap paddr=0x%x size=0x%x failed: %s\n",
                NNDMA_IO_PADDR, NNDMA_IO_SIZE, strerror(errno));
        goto cleanup_l2cache;
    }

    /* Step 3: Map NNA DMA descriptor RAM */
    g_nna_dev.nndma_desram_mapped = mmap(NULL, NNDMA_DESRAM_SIZE, PROT_READ | PROT_WRITE,
                                          MAP_SHARED, g_nna_dev.memfd, NNDMA_DESRAM_PADDR);
    if (g_nna_dev.nndma_desram_mapped == MAP_FAILED) {
        fprintf(stderr, "Error: NNDMA DESRAM mmap paddr=0x%x size=0x%x failed: %s\n",
                NNDMA_DESRAM_PADDR, NNDMA_DESRAM_SIZE, strerror(errno));
        goto cleanup_nndma_io;
    }

    /* Step 4: Map ORAM */
    g_nna_dev.oram_mapped = mmap(NULL, g_nna_dev.oram_size, PROT_READ | PROT_WRITE,
                                  MAP_SHARED, g_nna_dev.memfd, g_nna_dev.oram_pbase);
    if (g_nna_dev.oram_mapped == MAP_FAILED) {
        fprintf(stderr, "Error: ORAM mmap paddr=0x%08x size=0x%08x failed: %s\n",
                g_nna_dev.oram_pbase, g_nna_dev.oram_size, strerror(errno));
        goto cleanup_nndma_desram;
    }

    /* Step 5: Get nmem info from /proc/cmdline */
    uint32_t nmem_size = 0;
    uint32_t nmem_paddr = 0;

    if (get_nmem_info(&nmem_size, &nmem_paddr) < 0) {
        fprintf(stderr, "Error: Failed to get nmem info\n");
        goto cleanup_oram;
    }

    printf("nmem available: %u MB at 0x%08x\n", nmem_size / (1024*1024), nmem_paddr);

    /* Get NNA version */
    struct {
        uint32_t version_buf;
        uint32_t nmem_extension_buf;
        uint32_t nmem_paddr;
        uint32_t nmem_size;
    } version_info = {0};

    if (ioctl(g_nna_dev.fd, IOCTL_SOC_NNA_VERSION, &version_info) < 0) {
        fprintf(stderr, "Warning: The version number is not obtained. Please upgrade the soc-nna!\n");
        version_info.version_buf = 0x41;  /* Default T41 */
    }

    printf("NNA version: 0x%08x\n", version_info.version_buf);

    /* Allocate DDR DMA memory - use available nmem size or 4MB, whichever is smaller */
    g_nna_dev.ddr_size = nmem_size;
    if (g_nna_dev.ddr_size > 4 * 1024 * 1024) {
        g_nna_dev.ddr_size = 4 * 1024 * 1024;  /* Cap at 4MB for now */
    }

    struct soc_nna_buf ddr_buf;
    ddr_buf.size = g_nna_dev.ddr_size;

    if (ioctl(g_nna_dev.fd, IOCTL_SOC_NNA_MALLOC, &ddr_buf) < 0) {
        fprintf(stderr, "Error: DDR Malloc size=%d failed: %s\n",
                g_nna_dev.ddr_size, strerror(errno));
        goto cleanup_oram;
    }

    g_nna_dev.ddr_pbase = (uint32_t)(uintptr_t)ddr_buf.paddr;

    /* Step 6: Map DDR memory to userspace */
    g_nna_dev.ddr_mapped = mmap(NULL, g_nna_dev.ddr_size, PROT_READ | PROT_WRITE,
                                 MAP_SHARED, g_nna_dev.memfd, g_nna_dev.ddr_pbase);
    if (g_nna_dev.ddr_mapped == MAP_FAILED) {
        fprintf(stderr, "Error: DDR mmap paddr=0x%08x size=0x%08x failed: %s\n",
                g_nna_dev.ddr_pbase, g_nna_dev.ddr_size, strerror(errno));
        /* Free the allocated DMA memory */
        ioctl(g_nna_dev.fd, IOCTL_SOC_NNA_FREE, &ddr_buf);
        goto cleanup_oram;
    }

    g_nna_dev.initialized = 1;

    /* Initialize runtime environment for .mgk models */
    if (nna_runtime_init() < 0) {
        fprintf(stderr, "nna_init: Runtime initialization failed\n");
        goto cleanup_ddr;
    }

    printf("NNA initialized successfully\n");
    printf("  Device: %s\n", NNA_DEVICE_PATH);
    printf("  ORAM: 0x%08x (%u KB)\n", g_nna_dev.oram_pbase, g_nna_dev.oram_size / 1024);
    printf("  DDR:  0x%08x (%u MB)\n", g_nna_dev.ddr_pbase, g_nna_dev.ddr_size / (1024*1024));
    printf("  NNDMA IO: %p\n", g_nna_dev.nndma_io_mapped);
    printf("  NNDMA DESRAM: %p\n", g_nna_dev.nndma_desram_mapped);

    return NNA_SUCCESS;

cleanup_ddr:
    munmap(g_nna_dev.ddr_mapped, g_nna_dev.ddr_size);
    struct soc_nna_buf cleanup_buf = { .paddr = (void*)(uintptr_t)g_nna_dev.ddr_pbase, .size = g_nna_dev.ddr_size };
    ioctl(g_nna_dev.fd, IOCTL_SOC_NNA_FREE, &cleanup_buf);
cleanup_oram:
    munmap(g_nna_dev.oram_mapped, g_nna_dev.oram_size);
cleanup_nndma_desram:
    munmap(g_nna_dev.nndma_desram_mapped, NNDMA_DESRAM_SIZE);
cleanup_nndma_io:
    munmap(g_nna_dev.nndma_io_mapped, NNDMA_IO_SIZE);
cleanup_l2cache:
    munmap(g_nna_dev.l2cache_size_vaddr, 0x1000);
cleanup_devices:
    close(g_nna_dev.fd);
    close(g_nna_dev.memfd);
    g_nna_dev.fd = -1;
    g_nna_dev.memfd = -1;
    return NNA_ERROR_INIT;
}

void nna_deinit(void) {
    if (!g_nna_dev.initialized) {
        return;
    }

    /* Unmap DDR memory */
    if (g_nna_dev.ddr_mapped != NULL) {
        munmap(g_nna_dev.ddr_mapped, g_nna_dev.ddr_size);
        g_nna_dev.ddr_mapped = NULL;

        /* Free DDR DMA memory */
        if (g_nna_dev.fd >= 0) {
            struct soc_nna_buf ddr_buf = {
                .paddr = (void*)(uintptr_t)g_nna_dev.ddr_pbase,
                .size = g_nna_dev.ddr_size
            };
            ioctl(g_nna_dev.fd, IOCTL_SOC_NNA_FREE, &ddr_buf);
        }
    }

    /* Unmap ORAM */
    if (g_nna_dev.oram_mapped != NULL) {
        munmap(g_nna_dev.oram_mapped, g_nna_dev.oram_size);
        g_nna_dev.oram_mapped = NULL;
    }

    /* Unmap NNA DMA descriptor RAM */
    if (g_nna_dev.nndma_desram_mapped != NULL) {
        munmap(g_nna_dev.nndma_desram_mapped, NNDMA_DESRAM_SIZE);
        g_nna_dev.nndma_desram_mapped = NULL;
    }

    /* Unmap NNA DMA I/O registers */
    if (g_nna_dev.nndma_io_mapped != NULL) {
        munmap(g_nna_dev.nndma_io_mapped, NNDMA_IO_SIZE);
        g_nna_dev.nndma_io_mapped = NULL;
    }

    /* Unmap L2 cache size register */
    if (g_nna_dev.l2cache_size_vaddr != NULL) {
        munmap(g_nna_dev.l2cache_size_vaddr, 0x1000);
        g_nna_dev.l2cache_size_vaddr = NULL;
    }

    /* Close devices */
    if (g_nna_dev.fd >= 0) {
        close(g_nna_dev.fd);
        g_nna_dev.fd = -1;
    }

    if (g_nna_dev.memfd >= 0) {
        close(g_nna_dev.memfd);
        g_nna_dev.memfd = -1;
    }

    g_nna_dev.initialized = 0;
}

int nna_get_hw_info(nna_hw_info_t *info) {
    if (info == NULL) {
        return NNA_ERROR_INVALID;
    }

    if (!g_nna_dev.initialized) {
        return NNA_ERROR_INIT;
    }

    info->oram_vbase = (uint32_t)(uintptr_t)g_nna_dev.oram_mapped;
    info->oram_pbase = g_nna_dev.oram_pbase;
    info->oram_size = g_nna_dev.oram_size;

    /* Get NNA version from kernel driver */
    struct {
        uint32_t version_buf;
        uint32_t nmem_extension_buf;
        uint32_t nmem_paddr;
        uint32_t nmem_size;
    } version_info = {0};

    if (ioctl(g_nna_dev.fd, IOCTL_SOC_NNA_VERSION, &version_info) == 0) {
        info->version = version_info.version_buf;
    } else {
        info->version = 0x41;  /* Default T41 */
    }

    return NNA_SUCCESS;
}

int nna_is_ready(void) {
    return g_nna_dev.initialized;
}

const char* nna_get_version(void) {
    return "0.1.0-dev";
}

/* Internal function to get device FD */
int nna_device_get_fd(void) {
    return g_nna_dev.fd;
}

/* Internal function to get /dev/mem FD */
int nna_device_get_memfd(void) {
    return g_nna_dev.memfd;
}

/* Internal function to get ORAM pointer */
void* nna_device_get_oram(void) {
    return g_nna_dev.oram_mapped;
}

/* Internal function to get NNA DMA I/O registers */
void* nna_device_get_nndma_io(void) {
    return g_nna_dev.nndma_io_mapped;
}

/* Internal function to get NNA DMA descriptor RAM */
void* nna_device_get_nndma_desram(void) {
    return g_nna_dev.nndma_desram_mapped;
}

/* Internal function to get DDR virtual address */
void* nna_device_get_ddr(void) {
    return g_nna_dev.ddr_mapped;
}

/* Internal function to get DDR physical address */
uint32_t nna_device_get_ddr_pbase(void) {
    return g_nna_dev.ddr_pbase;
}

int nna_lock(void) {
    /* TODO: Implement multi-process locking via /dev/nna_lock */
    return NNA_SUCCESS;
}

int nna_unlock(void) {
    /* TODO: Implement multi-process unlocking */
    return NNA_SUCCESS;
}

