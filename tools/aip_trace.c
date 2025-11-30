/*
 * AIP Trace - LD_PRELOAD library to trace libaip.so calls
 * Usage: LD_PRELOAD=/opt/aip_trace.so LD_LIBRARY_PATH=/opt /opt/some_aip_app
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#define AIP_IOBASE 0x12b00000

/* Original function pointers */
static int (*real_ingenic_aip_init)(void) = NULL;
static int (*real_aip_f_init)(void *nodes, int count, int cfg) = NULL;
static int (*real_aip_f_start)(int mode) = NULL;
static int (*real_aip_f_wait)(void) = NULL;
static void (*real_aip_f_reset)(void) = NULL;

static volatile uint32_t *aip_io = NULL;
static int fd_mem = -1;

static void init_trace(void) {
    static int initialized = 0;
    if (initialized) return;
    initialized = 1;
    
    fd_mem = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd_mem >= 0) {
        aip_io = mmap(NULL, 0x1000, PROT_READ | PROT_WRITE, MAP_SHARED, fd_mem, AIP_IOBASE);
        if (aip_io == MAP_FAILED) aip_io = NULL;
    }
    fprintf(stderr, "[TRACE] AIP trace initialized, io=%p\n", (void*)aip_io);
}

static void dump_aip_f_regs(const char *label) {
    if (!aip_io) return;
    fprintf(stderr, "[TRACE] === %s ===\n", label);
    fprintf(stderr, "  CTRL:      0x%08x\n", aip_io[0x200/4]);
    fprintf(stderr, "  CFG:       0x%08x\n", aip_io[0x208/4]);
    fprintf(stderr, "  NODECFG:   0x%08x\n", aip_io[0x20C/4]);
    fprintf(stderr, "  IN_ADDR:   0x%08x\n", aip_io[0x210/4]);
    fprintf(stderr, "  OUT_ADDR:  0x%08x\n", aip_io[0x214/4]);
    fprintf(stderr, "  IN_SIZE:   0x%08x\n", aip_io[0x218/4]);
    fprintf(stderr, "  OUT_SIZE:  0x%08x\n", aip_io[0x21C/4]);
    fprintf(stderr, "  KERN_ADDR: 0x%08x\n", aip_io[0x220/4]);
    fprintf(stderr, "  BIAS_ADDR: 0x%08x\n", aip_io[0x224/4]);
    fprintf(stderr, "  KERN_SIZE: 0x%08x\n", aip_io[0x228/4]);
    fprintf(stderr, "  STRIDE:    0x%08x\n", aip_io[0x22C/4]);
    fprintf(stderr, "  IN/OUT_CH: 0x%08x\n", aip_io[0x230/4]);
    fprintf(stderr, "  PAD_POOL:  0x%08x\n", aip_io[0x234/4]);
    fprintf(stderr, "  SCALE_SH:  0x%08x\n", aip_io[0x238/4]);
    fprintf(stderr, "  CHAIN_ADR: 0x%08x\n", aip_io[0x23C/4]);
    fprintf(stderr, "  CHAIN_SZ:  0x%08x\n", aip_io[0x240/4]);
}

/* Hook ingenic_aip_init */
int ingenic_aip_init(void) {
    init_trace();
    if (!real_ingenic_aip_init) {
        real_ingenic_aip_init = dlsym(RTLD_NEXT, "ingenic_aip_init");
    }
    fprintf(stderr, "[TRACE] ingenic_aip_init() called\n");
    dump_aip_f_regs("Before init");
    int ret = real_ingenic_aip_init();
    dump_aip_f_regs("After init");
    fprintf(stderr, "[TRACE] ingenic_aip_init() returned %d\n", ret);
    return ret;
}

/* Hook aip_f_init */
int aip_f_init(void *nodes, int count, int cfg) {
    init_trace();
    if (!real_aip_f_init) {
        real_aip_f_init = dlsym(RTLD_NEXT, "aip_f_init");
    }
    fprintf(stderr, "[TRACE] aip_f_init(nodes=%p, count=%d, cfg=0x%x) called\n", nodes, count, cfg);
    
    /* Dump node parameters */
    if (nodes && count > 0) {
        uint32_t *n = (uint32_t*)nodes;
        fprintf(stderr, "[TRACE] Node params (12 x uint32):\n");
        for (int i = 0; i < 12 && i < count * 12; i++) {
            fprintf(stderr, "  [%2d] 0x%08x\n", i, n[i]);
        }
    }
    
    dump_aip_f_regs("Before aip_f_init");
    int ret = real_aip_f_init(nodes, count, cfg);
    dump_aip_f_regs("After aip_f_init");
    fprintf(stderr, "[TRACE] aip_f_init() returned %d\n", ret);
    return ret;
}

/* Hook aip_f_start */
int aip_f_start(int mode) {
    init_trace();
    if (!real_aip_f_start) {
        real_aip_f_start = dlsym(RTLD_NEXT, "aip_f_start");
    }
    fprintf(stderr, "[TRACE] aip_f_start(mode=%d) called\n", mode);
    dump_aip_f_regs("Before start");
    int ret = real_aip_f_start(mode);
    fprintf(stderr, "[TRACE] aip_f_start() returned %d, CTRL=0x%08x\n", ret, aip_io ? aip_io[0x200/4] : 0);
    return ret;
}

/* Hook aip_f_wait */
int aip_f_wait(void) {
    init_trace();
    if (!real_aip_f_wait) {
        real_aip_f_wait = dlsym(RTLD_NEXT, "aip_f_wait");
    }
    fprintf(stderr, "[TRACE] aip_f_wait() called, CTRL=0x%08x\n", aip_io ? aip_io[0x200/4] : 0);
    int ret = real_aip_f_wait();
    dump_aip_f_regs("After wait");
    fprintf(stderr, "[TRACE] aip_f_wait() returned %d\n", ret);
    return ret;
}

/* Hook aip_f_reset */
void aip_f_reset(void) {
    init_trace();
    if (!real_aip_f_reset) {
        real_aip_f_reset = dlsym(RTLD_NEXT, "aip_f_reset");
    }
    fprintf(stderr, "[TRACE] aip_f_reset() called\n");
    real_aip_f_reset();
    dump_aip_f_regs("After reset");
}

/* Also hook ioctl to see what's happening */
int ioctl(int fd, unsigned long request, ...) {
    static int (*real_ioctl)(int, unsigned long, ...) = NULL;
    if (!real_ioctl) {
        real_ioctl = dlsym(RTLD_NEXT, "ioctl");
    }
    
    void *arg;
    __builtin_va_list ap;
    __builtin_va_start(ap, request);
    arg = __builtin_va_arg(ap, void*);
    __builtin_va_end(ap);
    
    /* Log AIP-related ioctls */
    if ((request & 0xFFFF0000) == 0xC0040000) {
        fprintf(stderr, "[TRACE] ioctl(fd=%d, req=0x%lx, arg=%p)\n", fd, request, arg);
    }
    
    int ret = real_ioctl(fd, request, arg);
    
    if ((request & 0xFFFF0000) == 0xC0040000) {
        fprintf(stderr, "[TRACE] ioctl returned %d\n", ret);
    }
    
    return ret;
}

