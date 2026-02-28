#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "mxuv3.h"

static uint64_t now_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000ull + (uint64_t)tv.tv_usec;
}

#ifdef __mips__
static void mxuv3_clear_block_128(int16_t *block) {
    if (((uintptr_t)block & 63u) != 0) {
        memset(block, 0, 128);
        return;
    }
    /*
     * Use SUMZ + MFSUM to reliably zero VPR0.
     *
     * VPR_ZERO (float self-subtract) fails when VPR0 contains NaN bit
     * patterns because NaN - NaN = NaN.  SUMZ is a hardware zero of the
     * sum register, and MFSUM copies it into a VPR — always producing
     * all-zero regardless of prior contents.
     *
     * The first COP2 instruction also primes CU2 if the kernel enables
     * it lazily on exception.
     */
    MXUV3_ZERO_VPR(0);
    SA0_VPR(0, block);
    SA0_VPR(0, (uint8_t *)block + 64);
}
#endif

static void clear_block_memset(int16_t *block) {
    memset(block, 0, 128);
}

static int check_zero_128(const int16_t *block) {
    for (int i = 0; i < 64; i++)
        if (block[i] != 0)
            return 0;
    return 1;
}

static void dump_i16_8(const char *label, const int16_t *block, int base) {
    printf("%s[%d..%d]:", label, base, base + 7);
    for (int i = 0; i < 8; i++)
        printf(" %d", block[base + i]);
    printf("\n");
}

int main(int argc, char **argv) {
#ifndef __mips__
    (void)argc; (void)argv;
    printf("mxuv3_blockclear_bench: built for non-MIPS host, nothing to do.\n");
    return 0;
#else
    int blocks = 4096;
    int iters = 200;
    if (argc > 1) blocks = atoi(argv[1]);
    if (argc > 2) iters  = atoi(argv[2]);
    if (blocks <= 0) blocks = 4096;
    if (iters <= 0) iters = 200;

    size_t bytes = (size_t)blocks * 128;
    int16_t *buf = NULL;
    if (posix_memalign((void **)&buf, 64, bytes) != 0 || !buf) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    /* Seed with non-zero values */
    for (size_t i = 0; i < (size_t)blocks * 64; i++)
        buf[i] = (int16_t)(i | 1);

    printf("buf=%p (buf&63=%lu)\n", (void *)buf, (unsigned long)((uintptr_t)buf & 63u));

    /* Correctness (aligned) */
    dump_i16_8("before clear", buf, 0);
    mxuv3_clear_block_128(buf);
    dump_i16_8("after clear", buf, 0);
    dump_i16_8("after clear", buf, 32);
    dump_i16_8("after clear", buf, 56);
    printf("MXUv3 clear_block_128 correctness (aligned): %s\n", check_zero_128(buf) ? "PASS" : "FAIL");
    if (!check_zero_128(buf))
        return 1;

    /* Correctness (unaligned fallback) */
    int16_t *unaligned = (int16_t *)((uint8_t *)buf + 2);
    clear_block_memset(unaligned);
    printf("memset clear correctness (unaligned): %s\n", check_zero_128(unaligned) ? "PASS" : "FAIL");
    if (!check_zero_128(unaligned))
        return 1;

    /* Benchmark memset */
    uint64_t t0 = now_us();
    for (int it = 0; it < iters; it++) {
        for (int b = 0; b < blocks; b++) {
            int16_t *blk = (int16_t *)((uint8_t *)buf + (size_t)b * 128);
            clear_block_memset(blk);
        }
    }
    uint64_t t1 = now_us();

    /* Re-seed for MXUv3 bench */
    for (size_t i = 0; i < (size_t)blocks * 64; i++)
        buf[i] = (int16_t)(i | 1);

    uint64_t t2 = now_us();
    for (int it = 0; it < iters; it++) {
        for (int b = 0; b < blocks; b++) {
            int16_t *blk = (int16_t *)((uint8_t *)buf + (size_t)b * 128);
            mxuv3_clear_block_128(blk);
        }
    }
    uint64_t t3 = now_us();

    double us_memset = (double)(t1 - t0);
    double us_mxu    = (double)(t3 - t2);
    double mb = ((double)bytes * (double)iters) / (1024.0 * 1024.0);
    printf("\nBench: %d blocks x %d iters (%.1f MiB total writes)\n", blocks, iters, mb);
    printf("  memset : %.0f us (%.1f MiB/s)\n", us_memset, mb / (us_memset / 1e6));
    printf("  MXUv3  : %.0f us (%.1f MiB/s)\n", us_mxu,    mb / (us_mxu / 1e6));
    printf("  speedup: %.2fx\n", us_memset / us_mxu);

    free(buf);
    return 0;
#endif
}
