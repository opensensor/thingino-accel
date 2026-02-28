#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "mxuv3.h"

static int32_t dot16_ssb(const int8_t *a, const int8_t *b) {
    int32_t s = 0;
    for (int i = 0; i < 16; i++)
        s += (int32_t)a[i] * (int32_t)b[i];
    return s;
}

static void scalar_s4macssb(int32_t out[4], const int8_t a[64], const int8_t b[64]) {
    out[0] = dot16_ssb(a + 0,  b + 0);
    out[1] = dot16_ssb(a + 16, b + 16);
    out[2] = dot16_ssb(a + 32, b + 32);
    out[3] = dot16_ssb(a + 48, b + 48);
}

static uint64_t now_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000ull + (uint64_t)tv.tv_usec;
}

int main(int argc, char **argv) {
#ifndef __mips__
    (void)argc; (void)argv;
    printf("mxuv3_s4mac_test: built for non-MIPS host, nothing to do.\n");
    return 0;
#else
    int iters = 100000;
    if (argc > 1) {
        iters = atoi(argv[1]);
        if (iters <= 0) iters = 100000;
    }

    static int8_t a[64] __attribute__((aligned(64)));
    static int8_t b[64] __attribute__((aligned(64)));
    static int32_t out_words[16] __attribute__((aligned(64)));

    /* Deterministic pattern (mix of negative/positive) */
    for (int i = 0; i < 64; i++) {
        a[i] = (int8_t)((i * 3) - 96);
        b[i] = (int8_t)((i * 5) - 80);
    }
    memset(out_words, 0, sizeof(out_words));

    int32_t ref[4];
    scalar_s4macssb(ref, a, b);

    VSR_ZERO(0);
    LA0_VPR(0, a);
    LA0_VPR(1, b);
    S4MACSSB(0, 0, 1);
    MFSUMZ(2, 0);
    SA0_VPR(2, out_words);

    /* Per mxuv3.h: results are at int32 indices 0,4,8,12 */
    int32_t got[4] = { out_words[0], out_words[4], out_words[8], out_words[12] };

    int ok = (got[0] == ref[0] && got[1] == ref[1] && got[2] == ref[2] && got[3] == ref[3]);
    printf("MXUv3 S4MACSSB correctness: %s\n", ok ? "PASS" : "FAIL");
    if (!ok) {
        printf("  ref: %d %d %d %d\n", ref[0], ref[1], ref[2], ref[3]);
        printf("  got: %d %d %d %d\n", got[0], got[1], got[2], got[3]);
        return 1;
    }

    /* Benchmark */
    volatile int32_t sink = 0;
    uint64_t t0 = now_us();
    for (int i = 0; i < iters; i++) {
        VSR_ZERO(0);
        LA0_VPR(0, a);
        LA0_VPR(1, b);
        S4MACSSB(0, 0, 1);
        MFSUMZ(2, 0);
        SA0_VPR(2, out_words);
        sink ^= out_words[0];
    }
    uint64_t t1 = now_us();
    double us = (double)(t1 - t0);
    printf("Benchmark: %d iters in %.0f us (%.2f ns/iter) [sink=%d]\n",
           iters, us, (us * 1000.0) / (double)iters, sink);

    return 0;
#endif
}
