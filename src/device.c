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

/* ORAM constants (from soc_nna.ko analysis) */
#define NNA_ORAM_BASE_ADDR  0x12620000  /* Actual ORAM base from kernel */
#define NNA_ORAM_BASE_SIZE  0x60000     /* 384 KB for T41 */

/* Memory buffer structure */
struct soc_nna_buf {
    void *vaddr;
    void *paddr;
    int size;
};

/* Global device state */
static struct {
    int fd;                /* NNA device file descriptor */
    int memfd;             /* /dev/mem file descriptor */
    int initialized;       /* Initialization flag */
    void *oram_mapped;     /* Mapped ORAM pointer */
    uint32_t oram_pbase;   /* ORAM physical address */
    uint32_t oram_size;    /* ORAM size */
} g_nna_dev = {
    .fd = -1,
    .memfd = -1,
    .initialized = 0,
    .oram_mapped = NULL,
    .oram_pbase = NNA_ORAM_BASE_ADDR,
    .oram_size = NNA_ORAM_BASE_SIZE,
};

int nna_init(void) {
    if (g_nna_dev.initialized) {
        return NNA_SUCCESS;
    }

    /* Open /dev/mem for physical memory mapping */
    g_nna_dev.memfd = open("/dev/mem", O_RDWR | O_SYNC);
    if (g_nna_dev.memfd < 0) {
        fprintf(stderr, "nna_init: Failed to open /dev/mem: %s\n",
                strerror(errno));
        return NNA_ERROR_DEVICE;
    }

    /* Open NNA device */
    g_nna_dev.fd = open(NNA_DEVICE_PATH, O_RDWR);
    if (g_nna_dev.fd < 0) {
        fprintf(stderr, "nna_init: Failed to open %s: %s\n",
                NNA_DEVICE_PATH, strerror(errno));
        close(g_nna_dev.memfd);
        g_nna_dev.memfd = -1;
        return NNA_ERROR_DEVICE;
    }

    /*
     * Note: ORAM is not directly mapped to userspace in the standard flow.
     * The kernel driver manages ORAM internally. User applications allocate
     * DMA memory from the nmem region using IOCTL_SOC_NNA_MALLOC.
     *
     * ORAM info is stored for reference but not mapped.
     */
    g_nna_dev.oram_mapped = NULL;

    g_nna_dev.initialized = 1;

    /* Initialize runtime environment for .mgk models */
    if (nna_runtime_init() < 0) {
        fprintf(stderr, "nna_init: Runtime initialization failed\n");
        close(g_nna_dev.memfd);
        close(g_nna_dev.fd);
        g_nna_dev.memfd = -1;
        g_nna_dev.fd = -1;
        g_nna_dev.initialized = 0;
        return NNA_ERROR_INIT;
    }

    printf("NNA initialized successfully\n");
    printf("  Device: %s\n", NNA_DEVICE_PATH);
    printf("  ORAM: 0x%08x (%u KB) - managed by kernel\n",
           g_nna_dev.oram_pbase,
           g_nna_dev.oram_size / 1024);

    return NNA_SUCCESS;
}

void nna_deinit(void) {
    if (!g_nna_dev.initialized) {
        return;
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
    g_nna_dev.oram_mapped = NULL;
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

int nna_lock(void) {
    /* TODO: Implement multi-process locking via /dev/nna_lock */
    return NNA_SUCCESS;
}

int nna_unlock(void) {
    /* TODO: Implement multi-process unlocking */
    return NNA_SUCCESS;
}

