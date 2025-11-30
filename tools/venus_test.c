/*
 * venus_test.c - Test Ingenic Venus library
 *
 * This program uses dlopen to load the Ingenic libraries and test
 * if they can initialize the NNA hardware correctly.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <errno.h>

/* IOCTL codes from kernel driver */
#define IOCTL_AIP_MALLOC    0xc0045001
#define IOCTL_AIP_FREE      0xc0045002

struct jz_aip_chainbuf {
    void *vaddr;
    uint32_t paddr;
    uint32_t size;
};

/* Function pointer types based on decompiled code */
typedef int (*aie_mmap_fn)(void);
typedef int (*aie_munmap_fn)(void);
typedef void* (*get_aip_ioaddr_fn)(void);
typedef int (*aip_open_clk_fn)(void);
typedef int (*aip_close_clk_fn)(void);
typedef int (*aip_t_reset_fn)(void*);
typedef int (*aip_f_reset_fn)(void*);

/* Test jzaip mmap directly */
static int test_jzaip_mmap(const char *devname)
{
    int fd;
    struct jz_aip_chainbuf buf;
    void *mapped;
    int ret;

    printf("\n=== Testing %s ===\n", devname);

    fd = open(devname, O_RDWR);
    if (fd < 0) {
        printf("  Failed to open: %s\n", strerror(errno));
        return -1;
    }
    printf("  Opened fd=%d\n", fd);

    /* Get chain buffer info via ioctl */
    memset(&buf, 0, sizeof(buf));
    buf.size = 0x36000;  /* Request size like library does */

    ret = ioctl(fd, IOCTL_AIP_MALLOC, &buf);
    printf("  ioctl MALLOC returned: %d (errno=%d: %s)\n", ret, errno, strerror(errno));
    printf("  buf: vaddr=%p paddr=0x%08x size=0x%x\n", buf.vaddr, buf.paddr, buf.size);

    if (ret >= 0 && buf.paddr != 0) {
        /* Try to mmap */
        mapped = mmap(NULL, buf.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.paddr);
        if (mapped == MAP_FAILED) {
            printf("  mmap failed: %s\n", strerror(errno));
        } else {
            printf("  mmap succeeded: %p\n", mapped);
            munmap(mapped, buf.size);
        }
    }

    close(fd);
    return 0;
}

int main(int argc, char *argv[])
{
    void *libdrivers = NULL;
    void *libaip = NULL;
    int ret;
    
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Venus Library Test                                          ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    fflush(stdout);

    /* First test jzaip devices directly */
    test_jzaip_mmap("/dev/jzaip_t");
    test_jzaip_mmap("/dev/jzaip_f");
    test_jzaip_mmap("/dev/jzaip_p");

    printf("\n");
    fflush(stdout);

    /* Load libdrivers.so only - skip libaip.so due to constructor issues */
    printf("Loading libdrivers.so...\n");
    fflush(stdout);
    libdrivers = dlopen("/opt/libdrivers.so", RTLD_NOW | RTLD_GLOBAL);
    if (!libdrivers) {
        printf("  Failed: %s\n", dlerror());
        return 1;
    }
    printf("  Loaded successfully\n");
    fflush(stdout);

    /* Skip libaip.so - it has constructor issues with chain buffer sizes */
    libaip = NULL;
    printf("Skipping libaip.so (constructor has chain buffer size mismatch)\n");
    fflush(stdout);
    
    /* Get function pointers from libdrivers */
    aie_mmap_fn aie_mmap = (aie_mmap_fn)dlsym(libdrivers, "__aie_mmap");
    aie_munmap_fn aie_munmap = (aie_munmap_fn)dlsym(libdrivers, "__aie_munmap");

    printf("\nFunction pointers:\n");
    printf("  __aie_mmap:     %p\n", (void*)aie_mmap);
    printf("  __aie_munmap:   %p\n", (void*)aie_munmap);
    fflush(stdout);

    if (!aie_mmap || !aie_munmap) {
        printf("\nError: Required functions not found\n");
        dlclose(libdrivers);
        return 1;
    }

    /* First, let's check if show_drivers_version works */
    typedef void (*show_version_fn)(void);
    show_version_fn show_version = (show_version_fn)dlsym(libdrivers, "show_drivers_version");
    printf("\n  show_drivers_version: %p\n", (void*)show_version);
    fflush(stdout);

    if (show_version) {
        printf("  Calling show_drivers_version()...\n");
        fflush(stdout);
        show_version();
        printf("  Done\n");
        fflush(stdout);
    }

    /* Skip __aie_mmap for now - it crashes */
    printf("\nSkipping __aie_mmap (crashes - needs investigation)\n");
    fflush(stdout);
    ret = -1;

    if (ret == 0) {
        printf("  __aie_mmap succeeded!\n");

        /* Try to get some info from the library's global variables */
        uint32_t *oram_base = (uint32_t*)dlsym(libdrivers, "oram_base");
        uint32_t *oram_real_size = (uint32_t*)dlsym(libdrivers, "oram_real_size");
        uint32_t *l2cache_size = (uint32_t*)dlsym(libdrivers, "l2cache_size");
        void **oram_vbase = (void**)dlsym(libdrivers, "__oram_vbase");
        void **ddr_vbase = (void**)dlsym(libdrivers, "__ddr_vbase");
        uint32_t *ddr_pbase = (uint32_t*)dlsym(libdrivers, "__ddr_pbase");

        printf("\n  Global variables:\n");
        if (oram_base) printf("    oram_base:      0x%08x\n", *oram_base);
        if (oram_real_size) printf("    oram_real_size: 0x%08x\n", *oram_real_size);
        if (l2cache_size) printf("    l2cache_size:   0x%08x\n", *l2cache_size);
        if (oram_vbase) printf("    __oram_vbase:   %p\n", *oram_vbase);
        if (ddr_vbase) printf("    __ddr_vbase:    %p\n", *ddr_vbase);
        if (ddr_pbase) printf("    __ddr_pbase:    0x%08x\n", *ddr_pbase);
        fflush(stdout);
    }

    /* Cleanup */
    printf("\nCalling __aie_munmap()...\n");
    fflush(stdout);
    ret = aie_munmap();
    printf("  Returned: %d\n", ret);

    dlclose(libdrivers);

    printf("\n✓ Test complete\n");
    return 0;
}

