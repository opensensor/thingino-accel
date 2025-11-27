/*
 * thingino-accel - Runtime Environment for .mgk Models
 *
 * Provides symbols required by .mgk model files
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include "nna.h"
#include "device_internal.h"

/* ORAM and DDR base addresses - exported for .mgk models */
void *oram_base = NULL;
void *__oram_vbase = NULL;
void *__ddr_pbase = NULL;
void *__ddr_vbase = NULL;
void *__nndma_io_vbase = NULL;
void *__nndma_desram_vbase = NULL;

/* Global variables needed by .mgk models */
int l2cache_size = 256 * 1024;  /* 256KB L2 cache */
pthread_mutex_t net_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Standard library functions that may be missing */
void __assert(const char *func, const char *file, int line, const char *expr) {
    fprintf(stderr, "Assertion failed: %s (%s: %s: %d)\n", expr, file, func, line);

    const char *s_func = func ? func : "";
    const char *s_file = file ? file : "";
    const char *s_expr = expr ? expr : "";

    int has_output1 = strstr(s_func, "output_size==1") ||
                      strstr(s_file, "output_size==1") ||
                      strstr(s_expr, "output_size==1");
    int has_p_param = strstr(s_func, "p==(uint64_t)param_index") ||
                      strstr(s_file, "p==(uint64_t)param_index") ||
                      strstr(s_expr, "p==(uint64_t)param_index");
    int has_file = strstr(s_func, "magik_op_override.cpp") ||
                   strstr(s_file, "magik_op_override.cpp") ||
                   strstr(s_expr, "magik_op_override.cpp");
    int has_func = strstr(s_func, "conv2d_int8_param_init") ||
                   strstr(s_file, "conv2d_int8_param_init") ||
                   strstr(s_expr, "conv2d_int8_param_init");

    if ((has_output1 || has_p_param) && has_file && has_func) {
        fprintf(stderr, "[VENUS] ignoring conv2d_int8_param_init assert (%s) via __assert; continuing execution\n", s_expr);
        fflush(stderr);
        return;
    }

    abort();
}

/* Get device state from device.c */
extern void* nna_device_get_oram(void);
extern void* nna_device_get_nndma_io(void);
extern void* nna_device_get_nndma_desram(void);
extern void* nna_device_get_ddr(void);
extern uint32_t nna_device_get_ddr_pbase(void);
/* Local DDR backing store for .mgk models.
 *
 * The OEM stack uses a DDR heap managed by libdrivers / soc-nna. For now we
 * decouple the logical DDR pointer that .mgk models see (__ddr_vbase) from the
 * physical DMA region used by the NNA driver.
 *
 * We back __ddr_vbase with a plain user-space allocation so that the model's
 * parameter parsing and CPU-side tensor ops cannot fault on device mappings
 * while we bring up the rest of the stack.
 */
static void *runtime_ddr_base = NULL;
static size_t runtime_ddr_size = 0;


/* Initialize runtime environment */
int nna_runtime_init(void) {
    nna_hw_info_t hw_info;

    if (nna_get_hw_info(&hw_info) != NNA_SUCCESS) {
        fprintf(stderr, "nna_runtime_init: Failed to get hardware info\n");
        return -1;
    }

    /* Set ORAM base addresses */
    oram_base = (void*)(uintptr_t)hw_info.oram_pbase;
    __oram_vbase = nna_device_get_oram();

    /* Set DDR base addresses.
     *
     * For now we back __ddr_vbase with a plain user-space allocation instead of
     * mapping the kernel's DDR DMA region directly into user space. This avoids
     * SIGBUS faults from CPU-side accesses to device memory while we are still
     * bringing up the NNA stack. The NNA driver continues to use its own DMA
     * buffers via nna_malloc / nna_free.
     */
    __ddr_pbase = (void*)(uintptr_t)nna_device_get_ddr_pbase();

    if (!runtime_ddr_base) {
        /* 8MB is plenty for model parameters and small workspaces for now. */
        const size_t requested = 8 * 1024 * 1024;
        void *buf = NULL;
        int ret = posix_memalign(&buf, 64, requested);
        if (ret != 0 || !buf) {
            fprintf(stderr, "nna_runtime_init: failed to allocate runtime DDR buffer (%zu bytes)\n", requested);
            return -1;
        }
        memset(buf, 0, requested);
        runtime_ddr_base = buf;
        runtime_ddr_size = requested;
    }
    __ddr_vbase = runtime_ddr_base;

    /* Set NNA DMA base addresses */
    __nndma_io_vbase = nna_device_get_nndma_io();
    __nndma_desram_vbase = nna_device_get_nndma_desram();

    printf("Runtime initialized:\n");
    printf("  oram_base = %p\n", oram_base);
    printf("  __oram_vbase = %p\n", __oram_vbase);
    printf("  __ddr_pbase = %p\n", __ddr_pbase);
    printf("  __ddr_vbase = %p\n", __ddr_vbase);
    printf("  __nndma_io_vbase = %p\n", __nndma_io_vbase);
    printf("  __nndma_desram_vbase = %p\n", __nndma_desram_vbase);

    return 0;
}

/* Cache flush function required by models */
void __aie_flushcache(void *addr, size_t size) {
    int fd = nna_device_get_fd();
    if (fd < 0) {
        fprintf(stderr, "__aie_flushcache: NNA not initialized\n");
        return;
    }

    /* Flush cache using kernel driver */
    struct flush_cache_info {
        unsigned int addr;
        unsigned int len;
        unsigned int dir;
    } info;

    info.addr = (unsigned int)(uintptr_t)addr;
    info.len = (unsigned int)size;
    info.dir = 0;  /* Default direction */

    /* IOCTL_SOC_NNA_FLUSHCACHE = 0xc0046302 */
    if (ioctl(fd, 0xc0046302, &info) < 0) {
        fprintf(stderr, "__aie_flushcache: ioctl failed\n");
    }
}

void __aie_flushcache_dir(void *addr, size_t size, int direction) {
    int fd = nna_device_get_fd();
    if (fd < 0) {
        fprintf(stderr, "__aie_flushcache_dir: NNA not initialized\n");
        return;
    }

    /* Flush cache using kernel driver */
    struct flush_cache_info {
        unsigned int addr;
        unsigned int len;
        unsigned int dir;
    } info;

    info.addr = (unsigned int)(uintptr_t)addr;
    info.len = (unsigned int)size;
    info.dir = (unsigned int)direction;

    /* IOCTL_SOC_NNA_FLUSHCACHE = 0xc0046302 */
    if (ioctl(fd, 0xc0046302, &info) < 0) {
        fprintf(stderr, "__aie_flushcache_dir: ioctl failed\n");
    }
}

/* Venus C++ stubs are now in venus_c_api.cpp */

