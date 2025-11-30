#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "mxuv3.h"

/*
 * Simple MXUv3 sum-register smoke test for T41.
 *
 * Flow:
 *   1. Fill a 64-byte buffer with a known pattern and load into VPR0.
 *   2. MTSUM VSR0 <- VPR0.
 *   3. Zero VPR0.
 *   4. MFSUM VPR1 <- VSR0 and store to out[].
 *   5. SUMZ VSR0; MFSUM VPR2 <- VSR0 and store to out_zero[].
 */

static uint8_t in[64]       __attribute__((aligned(64)));
static uint8_t out[64]      __attribute__((aligned(64)));
static uint8_t out_zero[64] __attribute__((aligned(64)));

static void dump_buf(const char *label, const uint8_t *buf, size_t n) {
    printf("%s:", label);
    for (size_t i = 0; i < n; ++i) {
        if ((i % 16) == 0) {
            printf("\n  %2zu:", i);
        }
        printf(" %02x", buf[i]);
    }
    printf("\n");
}

int main(void) {
#ifndef __mips__
    printf("mxuv3_sum_test: built for non-MIPS host, nothing to do.\n");
    return 0;
#else
    printf("MXUv3 sum-register smoke test\n");

    /* Prepare input pattern */
    for (int i = 0; i < 64; ++i) {
        in[i] = (uint8_t)(i + 1);
    }
    memset(out, 0, sizeof(out));
    memset(out_zero, 0, sizeof(out_zero));

    /* Load VPR0 from in[] */
    mxuv3_load_vpr0(in);

    /* Copy VPR0 -> VSR0 */
    MXUV3_MTSUM(0, 0);  /* vsd=0 (VSR0), vrs=0 (VPR0) */

    /* Zero VPR0 so we can see the effect of reading from VSR0 later */
    mxuv3_zero_vpr0();

    /* Move VSR0 -> VPR1, store to out[] */
    MXUV3_MFSUM(1, 0);  /* vrd=1 (VPR1), vss=0 (VSR0) */
    mxuv3_store_vpr1(out);

    /* Clear VSR0 and confirm we read back zeros */
    MXUV3_SUMZ(0);      /* vsd=0 (VSR0) */
    MXUV3_MFSUM(2, 0);  /* vrd=2 (VPR2), vss=0 (VSR0) */
    mxuv3_store_vpr2(out_zero);

    dump_buf("in", in, sizeof(in));
    dump_buf("out (MFSUM VSR0)", out, sizeof(out));
    dump_buf("out_zero (SUMZ+MFSUM)", out_zero, sizeof(out_zero));

    int ok1 = (memcmp(in, out, sizeof(in)) == 0);
    int ok2 = 1;
    for (size_t i = 0; i < sizeof(out_zero); ++i) {
        if (out_zero[i] != 0) {
            ok2 = 0;
            break;
        }
    }

    printf("\nCheck 1 (VSR0 round-trip) : %s\n", ok1 ? "PASS" : "FAIL");
    printf("Check 2 (SUMZ behaviour)  : %s\n", ok2 ? "PASS" : "FAIL");

    return (ok1 && ok2) ? 0 : 1;
#endif
}

