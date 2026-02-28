#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "mxuv3.h"

static void *xaligned_alloc(size_t align, size_t size) {
    void *p = NULL;
    if (posix_memalign(&p, align, size) != 0)
        return NULL;
    return p;
}

#ifdef __mips__
static inline void sa0_raw_half(int vpr, void *ptr, int offset, int n) {
    register void *_base __asm__("t0") = ptr;
    uint32_t word = 0x710000d5u | ((uint32_t)offset << 16) | ((uint32_t)vpr << 11) | ((uint32_t)n << 9);
    __asm__ __volatile__(
        ".set push\n"
        ".set noreorder\n"
        ".word %1\n"
        ".set pop\n"
        :: "r"(_base), "i"(word)
        : "memory");
}
#endif

static void dump64(const char *label, const uint8_t *p) {
    printf("%s:", label);
    for (int i = 0; i < 64; i++) {
        if ((i % 16) == 0)
            printf("\n  %02x:", i);
        printf(" %02x", p[i]);
    }
    printf("\n");
}

int main(void) {
#ifndef __mips__
    printf("mxuv3_sa0_offset_test: built for non-MIPS host, nothing to do.\n");
    return 0;
#else
    uint8_t *in = xaligned_alloc(64, 64);
    uint8_t *out = xaligned_alloc(64, 64);
    uint8_t *big = xaligned_alloc(64, 256);
    if (!in || !out || !big) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    for (int i = 0; i < 64; i++)
        in[i] = (uint8_t)(i + 1);
    memset(out, 0, 64);
    memset(big, 0, 256);

    printf("MXUv3 SA0 offset validation\n");
    printf("- Expectation: offset 0/1 are correct; offset 2/3 may be broken on XBurst2.\n\n");

    /* Baseline: normal store via SA0_VPR (offsets 0 and 1) */
    mxuv3_load_vpr0(in);
    SA0_VPR(0, out);

    int ok01 = (memcmp(in, out, 64) == 0);
    printf("Check 1 (SA0 offsets 0/1 round-trip): %s\n", ok01 ? "PASS" : "FAIL");
    if (!ok01) {
        dump64("in", in);
        dump64("out", out);
        return 1;
    }

    /* Probe: attempt SA0 with offsets 2/3 (writing to big+64..127) */
    sa0_raw_half(0, big, 2, 0);
    sa0_raw_half(0, big, 3, 1);

    int ok23 = (memcmp(in, big + 64, 64) == 0);
    printf("Check 2 (SA0 offsets 2/3 behave like +64 bytes): %s\n", ok23 ? "PASS" : "FAIL");

    if (!ok23) {
        printf("Note: FAIL here is expected on some XBurst2 silicon/kernels; FFmpeg assumes only offsets 0/1 are reliable.\n");
    } else {
        printf("Note: offsets 2/3 appear to work on this device; rerun multiple times and with different buffers to confirm reliability.\n");
    }

    free(in);
    free(out);
    free(big);
    return 0;
#endif
}
