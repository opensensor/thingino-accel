/*
 * MXUv3 VPR MAX/MIN instruction test for Ingenic XBurst2 (A1/T41)
 *
 * Validates MAXSW, MINSW, MAXUB, MINUB on real hardware.
 * These are critical for FFmpeg pixel clamping (clip_uint8) and
 * neural network activation (ReLU = max(x, 0)).
 *
 * Each VPR is 512 bits = 64 bytes = 16 x int32 = 64 x uint8.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "mxuv3.h"

#ifdef __mips__

static int test_maxsw(void)
{
    int32_t a[16] __attribute__((aligned(64)));
    int32_t b[16] __attribute__((aligned(64)));
    int32_t out[16] __attribute__((aligned(64)));
    int pass = 1;

    for (int i = 0; i < 16; i++) {
        a[i] = (i % 2 == 0) ? (i * 100) : -(i * 100);
        b[i] = 0;
    }

    LA0_VPR(0, a);
    LA0_VPR(1, b);
    VPR_MAXSW(2, 0, 1);
    SA0_VPR(2, out);

    printf("MAXSW (ReLU: max(x, 0)) test:\n");
    for (int i = 0; i < 16; i++) {
        int32_t expect = a[i] > 0 ? a[i] : 0;
        if (out[i] != expect) {
            printf("  FAIL [%d]: a=%d expect=%d got=%d\n", i, a[i], expect, out[i]);
            pass = 0;
        }
    }
    if (pass) printf("  all 16 lanes correct: PASS\n");
    return pass;
}

static int test_minsw(void)
{
    int32_t a[16] __attribute__((aligned(64)));
    int32_t b[16] __attribute__((aligned(64)));
    int32_t out[16] __attribute__((aligned(64)));
    int pass = 1;

    for (int i = 0; i < 16; i++) {
        a[i] = i * 50 + 10;
        b[i] = 255;
    }

    LA0_VPR(0, a);
    LA0_VPR(1, b);
    VPR_MINSW(2, 0, 1);
    SA0_VPR(2, out);

    printf("MINSW (clamp upper: min(x, 255)) test:\n");
    for (int i = 0; i < 16; i++) {
        int32_t expect = a[i] < 255 ? a[i] : 255;
        if (out[i] != expect) {
            printf("  FAIL [%d]: a=%d expect=%d got=%d\n", i, a[i], expect, out[i]);
            pass = 0;
        }
    }
    if (pass) printf("  all 16 lanes correct: PASS\n");
    return pass;
}

static int test_maxub(void)
{
    uint8_t a[64] __attribute__((aligned(64)));
    uint8_t b[64] __attribute__((aligned(64)));
    uint8_t out[64] __attribute__((aligned(64)));
    int pass = 1, errs = 0;

    for (int i = 0; i < 64; i++) { a[i] = (uint8_t)(i * 3); b[i] = 128; }

    LA0_VPR(0, a);
    LA0_VPR(1, b);
    VPR_MAXUB(2, 0, 1);
    SA0_VPR(2, out);

    printf("MAXUB (64 x uint8 max) test:\n");
    for (int i = 0; i < 64; i++) {
        uint8_t expect = a[i] > b[i] ? a[i] : b[i];
        if (out[i] != expect) {
            if (errs < 4) printf("  FAIL [%d]: a=%u b=%u expect=%u got=%u\n", i, a[i], b[i], expect, out[i]);
            errs++; pass = 0;
        }
    }
    if (errs > 4) printf("  ... and %d more errors\n", errs - 4);
    if (pass) printf("  all 64 lanes correct: PASS\n");
    return pass;
}

static int test_minub(void)
{
    uint8_t a[64] __attribute__((aligned(64)));
    uint8_t b[64] __attribute__((aligned(64)));
    uint8_t out[64] __attribute__((aligned(64)));
    int pass = 1, errs = 0;

    for (int i = 0; i < 64; i++) { a[i] = (uint8_t)(i * 4); b[i] = 100; }

    LA0_VPR(0, a);
    LA0_VPR(1, b);
    VPR_MINUB(2, 0, 1);
    SA0_VPR(2, out);

    printf("MINUB (64 x uint8 min) test:\n");
    for (int i = 0; i < 64; i++) {
        uint8_t expect = a[i] < b[i] ? a[i] : b[i];
        if (out[i] != expect) {
            if (errs < 4) printf("  FAIL [%d]: a=%u b=%u expect=%u got=%u\n", i, a[i], b[i], expect, out[i]);
            errs++; pass = 0;
        }
    }
    if (errs > 4) printf("  ... and %d more errors\n", errs - 4);
    if (pass) printf("  all 64 lanes correct: PASS\n");
    return pass;
}

int main(void)
{
    printf("MXUv3 VPR MAX/MIN test (A1/T41)\n");
    printf("================================\n\n");
    __asm__ __volatile__(".word 0x4a80000b" ::: "memory");

    int pass = 1;
    pass &= test_maxsw(); printf("\n");
    pass &= test_minsw(); printf("\n");
    pass &= test_maxub(); printf("\n");
    pass &= test_minub();
    printf("\nOverall: %s\n", pass ? "ALL PASS" : "SOME FAILED");
    return pass ? 0 : 1;
}
#else
int main(void) { printf("Non-MIPS host, nothing to do.\n"); return 0; }
#endif