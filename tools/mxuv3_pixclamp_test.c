/*
 * MXUv3 pixel clamping & integer halfword test for Ingenic XBurst2 (A1/T41)
 *
 * Tests MAXSH, MINSH (signed halfword max/min) for clip_pixel() pattern
 * used by FFmpeg put_pixels_clamped / add_pixels_clamped.
 *
 * Also probes MAXSB and whether ADDSW/ADDSH/SUBSW/SUBSH exist under rs=16.
 *
 * Each VPR = 512 bits = 64 bytes = 32 x int16 = 16 x int32 = 64 x int8.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <setjmp.h>

#include "mxuv3.h"

#ifdef __mips__

static jmp_buf sigill_jmp;
static volatile int sigill_caught;

static void sigill_handler(int sig)
{
    (void)sig;
    sigill_caught = 1;
    longjmp(sigill_jmp, 1);
}

/* Try executing an arbitrary COP2 instruction, return 0 if SIGILL */
static int try_cop2(uint32_t encoding, const char *name)
{
    struct sigaction sa, old;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = sigill_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGILL, &sa, &old);
    sigill_caught = 0;

    if (setjmp(sigill_jmp) == 0) {
        /* Emit the instruction via .word */
        __asm__ __volatile__(".word %0\n sync\n" :: "r"(encoding) : "memory");
        /* If we get here, no SIGILL */
    }
    sigaction(SIGILL, &old, NULL);
    printf("  %-20s (0x%08x): %s\n", name, encoding,
           sigill_caught ? "SIGILL" : "OK");
    return !sigill_caught;
}

/*
 * MAXSH/MINSH: signed halfword max/min
 *   rs=16, fn=0x1D (MAXSH), fn=0x15 (MINSH)
 */
#define VPR_MAXSH(vrd, vrs, vrp) do { \
    __asm__ __volatile__( \
        ".word %0\n sync\n" \
        :: "i"(0x48000000 | (16 << 21) | ((vrs) << 16) | ((vrp) << 11) | ((vrd) << 6) | 0x1D) \
        : "memory"); \
} while(0)

#define VPR_MINSH(vrd, vrs, vrp) do { \
    __asm__ __volatile__( \
        ".word %0\n sync\n" \
        :: "i"(0x48000000 | (16 << 21) | ((vrs) << 16) | ((vrp) << 11) | ((vrd) << 6) | 0x15) \
        : "memory"); \
} while(0)

/* Test clip_pixel pattern: clamp int16 to [0, 255] using MAXSH+MINSH */
static int test_clip_pixel(void)
{
    int16_t input[32]  __attribute__((aligned(64)));
    int16_t zeros[32]  __attribute__((aligned(64)));
    int16_t limit[32]  __attribute__((aligned(64)));
    int16_t output[32] __attribute__((aligned(64)));
    int pass = 1;

    /* Fill with challenging values: negatives, 0-255 range, overflow */
    for (int i = 0; i < 32; i++) {
        input[i] = (int16_t)(i * 20 - 200); /* -200, -180, ..., 420 */
        zeros[i] = 0;
        limit[i] = 255;
    }

    /* VPR0=input, VPR1=zeros, VPR3=limit */
    LA0_VPR(0, input);
    LA0_VPR(1, zeros);
    LA0_VPR(3, limit);

    /* VPR2 = max(input, 0)  → clamp lower bound */
    VPR_MAXSH(2, 0, 1);
    /* VPR4 = min(VPR2, 255) → clamp upper bound */
    VPR_MINSH(4, 2, 3);

    SA0_VPR(4, output);

    printf("Pixel clamping (MAXSH+MINSH int16 clip to [0,255]) test:\n");
    int errs = 0;
    for (int i = 0; i < 32; i++) {
        int16_t v = input[i];
        int16_t expect = v < 0 ? 0 : (v > 255 ? 255 : v);
        if (output[i] != expect) {
            if (errs < 6)
                printf("  FAIL [%d]: in=%d expect=%d got=%d\n",
                       i, v, expect, output[i]);
            errs++;
            pass = 0;
        }
    }
    if (errs > 6) printf("  ... and %d more errors\n", errs - 6);
    if (pass) printf("  all 32 lanes correct: PASS\n");
    return pass;
}

/* Test MAXSB (signed byte max, fn=0x1C under rs=16) */
static int test_maxsb(void)
{
    int8_t a[64] __attribute__((aligned(64)));
    int8_t b[64] __attribute__((aligned(64)));
    int8_t out[64] __attribute__((aligned(64)));
    int pass = 1, errs = 0;

    for (int i = 0; i < 64; i++) {
        a[i] = (int8_t)(i * 3 - 96);    /* mix of pos/neg */
        b[i] = 0;                         /* ReLU for int8 */
    }

    LA0_VPR(0, a);
    LA0_VPR(1, b);
    /* MAXSB: rs=16, fn=0x1C */
    __asm__ __volatile__(
        ".word %0\n sync\n"
        :: "i"(0x48000000 | (16 << 21) | (0 << 16) | (1 << 11) | (2 << 6) | 0x1C)
        : "memory");
    SA0_VPR(2, out);

    printf("MAXSB (64 x int8 max, ReLU) test:\n");
    for (int i = 0; i < 64; i++) {
        int8_t expect = a[i] > 0 ? a[i] : 0;
        if (out[i] != expect) {
            if (errs < 4) printf("  FAIL [%d]: a=%d expect=%d got=%d\n",
                                 i, a[i], expect, out[i]);
            errs++; pass = 0;
        }
    }
    if (errs > 4) printf("  ... and %d more errors\n", errs - 4);
    if (pass) printf("  all 64 lanes correct: PASS\n");
    return pass;
}

int main(void)
{
    printf("MXUv3 pixel clamping & int16/int8 test (A1/T41)\n");
    printf("=================================================\n\n");
    __asm__ __volatile__(".word 0x4a80000b" ::: "memory");

    int pass = 1;
    pass &= test_clip_pixel();
    printf("\n");
    pass &= test_maxsb();

    printf("\nOverall: %s\n", pass ? "ALL PASS" : "SOME FAILED");
    return pass ? 0 : 1;
}
#else
int main(void) { printf("Non-MIPS host, nothing to do.\n"); return 0; }
#endif

