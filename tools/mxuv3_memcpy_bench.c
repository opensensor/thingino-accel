/*
 * MXUv3 VPR memcpy benchmark for Ingenic XBurst2 (A1/T41)
 *
 * Compares VPR-based 64-byte bulk copy vs scalar word copy for
 * FFmpeg put_pixels patterns: contiguous, stride-16, stride-8.
 *
 * Usage: mxuv3_memcpy_bench [iters]
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "mxuv3.h"

static uint64_t now_us(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_usec;
}

#ifdef __mips__

/* Scalar 16-byte row copy (like put_pixels16 does per row) */
static void scalar_copy_16(uint8_t *dst, const uint8_t *src, int stride, int rows)
{
    for (int r = 0; r < rows; r++) {
        uint32_t *d = (uint32_t *)dst;
        const uint32_t *s = (const uint32_t *)src;
        d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3];
        dst += stride;
        src += stride;
    }
}

/* VPR-based 64-byte copy (copy 4 rows of 16 bytes at once if stride=16) */
static void vpr_copy_64(uint8_t *dst, const uint8_t *src)
{
    LA0_VPR(0, src);
    SA0_VPR(0, dst);
}

/* Correctness test: copy 64 bytes via VPR and compare */
static int test_vpr_copy_correctness(void)
{
    uint8_t src[64] __attribute__((aligned(64)));
    uint8_t dst[64] __attribute__((aligned(64)));

    for (int i = 0; i < 64; i++) src[i] = (uint8_t)(i + 1);
    memset(dst, 0xAA, 64);

    vpr_copy_64(dst, src);

    int pass = 1;
    for (int i = 0; i < 64; i++) {
        if (dst[i] != src[i]) {
            printf("  FAIL at [%d]: expect=%u got=%u\n", i, src[i], dst[i]);
            pass = 0;
        }
    }
    return pass;
}

/* Benchmark: contiguous 1 MB copy */
static void bench_contiguous(int iters, size_t size)
{
    uint8_t *src = NULL, *dst = NULL;
    posix_memalign((void **)&src, 64, size);
    posix_memalign((void **)&dst, 64, size);
    if (!src || !dst) { printf("alloc fail\n"); return; }
    memset(src, 0x42, size);

    /* Scalar: word-at-a-time */
    uint64_t t0 = now_us();
    for (int it = 0; it < iters; it++) {
        uint32_t *s = (uint32_t *)src;
        uint32_t *d = (uint32_t *)dst;
        for (size_t i = 0; i < size / 4; i++) d[i] = s[i];
    }
    uint64_t t1 = now_us();

    /* VPR: 64 bytes at a time */
    uint64_t t2 = now_us();
    for (int it = 0; it < iters; it++) {
        mxuv3_memcpy_64(dst, src, size);
    }
    uint64_t t3 = now_us();

    double mb = ((double)size * iters) / (1024.0 * 1024.0);
    double us_scalar = (double)(t1 - t0);
    double us_vpr    = (double)(t3 - t2);
    printf("  Contiguous %zu KB x %d iters (%.1f MiB):\n", size/1024, iters, mb);
    printf("    scalar: %.0f us (%.1f MiB/s)\n", us_scalar, mb / (us_scalar / 1e6));
    printf("    VPR:    %.0f us (%.1f MiB/s)\n", us_vpr,    mb / (us_vpr / 1e6));
    printf("    speedup: %.2fx\n", us_scalar / us_vpr);

    free(src); free(dst);
}

/* Benchmark: stride-16 rows (like put_pixels16, h=16) */
static void bench_stride16(int iters)
{
    /* Simulate frame buffer: stride=1920, 16x16 block copies */
    int stride = 1920;
    int rows = 16;
    size_t frame_size = (size_t)stride * rows + 64;
    uint8_t *src = NULL, *dst = NULL;
    posix_memalign((void **)&src, 64, frame_size);
    posix_memalign((void **)&dst, 64, frame_size);
    if (!src || !dst) { printf("alloc fail\n"); return; }
    memset(src, 0x55, frame_size);

    uint64_t t0 = now_us();
    for (int it = 0; it < iters; it++)
        scalar_copy_16(dst, src, stride, rows);
    uint64_t t1 = now_us();

    /* VPR can't do strided copies (SA0/LA0 require contiguous 64-byte blocks).
     * But for stride==16, 4 rows are contiguous 64 bytes if we copy 4 at a time.
     * This only works when stride == width being copied. */
    uint64_t t2 = now_us();
    if (stride == 16) {
        for (int it = 0; it < iters; it++) {
            vpr_copy_64(dst, src);
            /* second 64 bytes for rows 4-7, etc. not contiguous with stride=1920 */
        }
    } else {
        /* For non-contiguous strides, VPR can't help directly.
         * We still benchmark the scalar path for comparison. */
        for (int it = 0; it < iters; it++)
            scalar_copy_16(dst, src, stride, rows);
    }
    uint64_t t3 = now_us();

    double ns_scalar = (double)(t1 - t0) * 1000.0 / iters;
    double ns_vpr    = (double)(t3 - t2) * 1000.0 / iters;
    printf("  Stride-16 (stride=%d, %d rows) x %d iters:\n", stride, rows, iters);
    printf("    scalar: %.0f ns/block\n", ns_scalar);
    printf("    VPR/scalar: %.0f ns/block (stride != 16, VPR N/A)\n", ns_vpr);

    free(src); free(dst);
}

int main(int argc, char **argv)
{
    int iters = 10000;
    if (argc > 1) iters = atoi(argv[1]);
    if (iters <= 0) iters = 10000;

    printf("MXUv3 VPR memcpy benchmark (A1/T41)\n");
    printf("====================================\n\n");
    __asm__ __volatile__(".word 0x4a80000b" ::: "memory");

    printf("VPR copy correctness: %s\n\n", test_vpr_copy_correctness() ? "PASS" : "FAIL");

    bench_contiguous(10, 1024 * 1024);
    printf("\n");
    bench_stride16(iters);

    return 0;
}
#else
int main(void) { printf("Non-MIPS host, nothing to do.\n"); return 0; }
#endif

