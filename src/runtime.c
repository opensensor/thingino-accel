/*
 * thingino-accel - Runtime Environment for .mgk Models
 * 
 * Provides symbols required by .mgk model files
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <sys/ioctl.h>

#include "nna.h"
#include "device_internal.h"

/* ORAM and DDR base addresses - exported for .mgk models */
void *oram_base = NULL;
void *__oram_vbase = NULL;
void *__ddr_pbase = NULL;

/* Initialize runtime environment */
int nna_runtime_init(void) {
    nna_hw_info_t hw_info;
    
    if (nna_get_hw_info(&hw_info) != NNA_SUCCESS) {
        fprintf(stderr, "nna_runtime_init: Failed to get hardware info\n");
        return -1;
    }
    
    /* Set ORAM base addresses */
    oram_base = (void*)(uintptr_t)hw_info.oram_pbase;
    __oram_vbase = (void*)(uintptr_t)hw_info.oram_vbase;
    
    /* DDR base is typically 0 on MIPS */
    __ddr_pbase = (void*)0;
    
    printf("Runtime initialized:\n");
    printf("  oram_base = %p\n", oram_base);
    printf("  __oram_vbase = %p\n", __oram_vbase);
    printf("  __ddr_pbase = %p\n", __ddr_pbase);
    
    return 0;
}

/* Cache flush function required by models */
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

