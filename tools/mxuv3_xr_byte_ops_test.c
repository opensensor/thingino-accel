/*
 * MXUv3 XR byte-parallel instruction test for Ingenic XBurst2 (A1/T41)
 *
 * Validates Q8AVG, Q8AVGR, Q8ABD, Q8ADD_AA, Q8ADD_SS on real hardware.
 * These are used in FFmpeg hpeldsp_mxu.c for half-pel interpolation.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __mips__

/* ---- Minimal XR intrinsics (same encodings as FFmpeg mxu.h) ---- */

#define _MXU_XR5(n) (((n) == 0) ? 16 : (n))
#define _MXU_GPR_T0 8

#define S32I2M(xr, val) do {                                               \
    register uint32_t __mxu_v __asm__("t0") = (uint32_t)(val);             \
    __asm__ __volatile__(".word %1"                                         \
        :: "r"(__mxu_v),                                                    \
           "i"(0x7000002F | (_MXU_GPR_T0 << 16) | (_MXU_XR5(xr) << 6)));  \
} while (0)

#define S32M2I(xr) ({                                                       \
    register uint32_t __mxu_v __asm__("t0");                                \
    __asm__ __volatile__(".word %1"                                         \
        : "=r"(__mxu_v)                                                     \
        : "i"(0x7000002E | (_MXU_GPR_T0 << 16) | (_MXU_XR5(xr) << 6)));   \
    __mxu_v; })

/* Q8AVG:  truncating average    byte_n = (a[n] + b[n]) >> 1 */
#define Q8AVG(xra, xrb, xrc) \
    __asm__ __volatile__(".word %0" :: "i"( \
        0x70100006 | ((xra) << 6) | ((xrb) << 10) | ((xrc) << 14)))

/* Q8AVGR: rounding average      byte_n = (a[n] + b[n] + 1) >> 1 */
#define Q8AVGR(xra, xrb, xrc) \
    __asm__ __volatile__(".word %0" :: "i"( \
        0x70140006 | ((xra) << 6) | ((xrb) << 10) | ((xrc) << 14)))

/* Q8ABD:  absolute difference   byte_n = |a[n] - b[n]| */
#define Q8ABD(xra, xrb, xrc) \
    __asm__ __volatile__(".word %0" :: "i"( \
        0x70100007 | ((xra) << 6) | ((xrb) << 10) | ((xrc) << 14)))

/* Q8ADD_AA: add-add with saturation  byte_n = clamp(a[n] + b[n], 0, 255) */
#define Q8ADD_AA(xra, xrb, xrc) \
    __asm__ __volatile__(".word %0" :: "i"( \
        0x701C0006 | ((xra) << 6) | ((xrb) << 10) | ((xrc) << 14)))

/* Q8ADD_SS: sub-sub with saturation  byte_n = clamp(a[n] - b[n], 0, 255) */
#define Q8ADD_SS(xra, xrb, xrc) \
    __asm__ __volatile__(".word %0" :: "i"( \
        0x731C0006 | ((xra) << 6) | ((xrb) << 10) | ((xrc) << 14)))

/* Trigger CU2 enablement (COP2 instruction) */
static void ensure_cu2(void)
{
    __asm__ __volatile__(".word 0x4a80000b" ::: "memory");
}

static int test_q8avg(void)
{
    uint32_t a = 0x80402010;  /* bytes: 0x10=16, 0x20=32, 0x40=64, 0x80=128 */
    uint32_t b = 0x60604020;  /* bytes: 0x20=32, 0x40=64, 0x60=96, 0x60=96  */
    uint32_t expect_trunc = 0x70502818; /* (16+32)/2=24, (32+64)/2=48, (64+96)/2=80, (128+96)/2=112 */
    uint32_t expect_round = 0x70502818; /* same here since all sums are even */

    S32I2M(1, a);
    S32I2M(2, b);
    Q8AVG(3, 1, 2);
    uint32_t got_trunc = S32M2I(3);

    S32I2M(1, a);
    S32I2M(2, b);
    Q8AVGR(4, 1, 2);
    uint32_t got_round = S32M2I(4);

    /* Test with odd sum: 1+2=3 → trunc=1, round=2 */
    uint32_t c = 0x01010101;
    uint32_t d = 0x02020202;
    S32I2M(1, c);
    S32I2M(2, d);
    Q8AVG(5, 1, 2);
    uint32_t got_odd_trunc = S32M2I(5);
    Q8AVGR(6, 1, 2);
    uint32_t got_odd_round = S32M2I(6);

    printf("Q8AVG  test:\n");
    printf("  a=0x%08x b=0x%08x\n", a, b);
    printf("  trunc: expect=0x%08x got=0x%08x %s\n", expect_trunc, got_trunc,
           got_trunc == expect_trunc ? "PASS" : "FAIL");
    printf("  round: expect=0x%08x got=0x%08x %s\n", expect_round, got_round,
           got_round == expect_round ? "PASS" : "FAIL");
    printf("  odd-sum trunc: expect=0x01010101 got=0x%08x %s\n", got_odd_trunc,
           got_odd_trunc == 0x01010101 ? "PASS" : "FAIL");
    printf("  odd-sum round: expect=0x02020202 got=0x%08x %s\n", got_odd_round,
           got_odd_round == 0x02020202 ? "PASS" : "FAIL");

    return (got_trunc == expect_trunc && got_round == expect_round &&
            got_odd_trunc == 0x01010101 && got_odd_round == 0x02020202);
}

static int test_q8abd(void)
{
    uint32_t a = 0xFF804020;  /* 32, 64, 128, 255 */
    uint32_t b = 0x60402010;  /* 16, 32, 64,  96  */
    /* |32-16|=16, |64-32|=32, |128-64|=64, |255-96|=159 */
    uint32_t expect = 0x9F402010;

    S32I2M(1, a);
    S32I2M(2, b);
    Q8ABD(3, 1, 2);
    uint32_t got = S32M2I(3);

    printf("Q8ABD  test:\n");
    printf("  a=0x%08x b=0x%08x\n", a, b);
    printf("  expect=0x%08x got=0x%08x %s\n", expect, got,
           got == expect ? "PASS" : "FAIL");
    return got == expect;
}

static int test_q8add(void)
{
    uint32_t a = 0xF0804020;  /* 32, 64, 128, 240 */
    uint32_t b = 0x20202020;  /* 32, 32, 32,  32  */
    /* AA: clamp(32+32,0,255)=64, clamp(64+32)=96, clamp(128+32)=160, clamp(240+32)=255 */
    uint32_t expect_aa = 0xFFA06040;
    /* SS: clamp(32-32,0,255)=0, clamp(64-32)=32, clamp(128-32)=96, clamp(240-32)=208 */
    uint32_t expect_ss = 0xD0602000;

    S32I2M(1, a);
    S32I2M(2, b);
    Q8ADD_AA(3, 1, 2);
    uint32_t got_aa = S32M2I(3);

    S32I2M(1, a);
    S32I2M(2, b);
    Q8ADD_SS(4, 1, 2);
    uint32_t got_ss = S32M2I(4);

    printf("Q8ADD  test:\n");
    printf("  a=0x%08x b=0x%08x\n", a, b);
    printf("  AA: expect=0x%08x got=0x%08x %s\n", expect_aa, got_aa,
           got_aa == expect_aa ? "PASS" : "FAIL");
    printf("  SS: expect=0x%08x got=0x%08x %s\n", expect_ss, got_ss,
           got_ss == expect_ss ? "PASS" : "FAIL");
    return (got_aa == expect_aa && got_ss == expect_ss);
}

int main(void)
{
    printf("MXUv3 XR byte-parallel ops test (A1/T41)\n");
    printf("=========================================\n\n");

    ensure_cu2();

    int pass = 1;
    pass &= test_q8avg();
    printf("\n");
    pass &= test_q8abd();
    printf("\n");
    pass &= test_q8add();

    printf("\nOverall: %s\n", pass ? "ALL PASS" : "SOME FAILED");
    return pass ? 0 : 1;
}

#else
int main(void) {
    printf("Built for non-MIPS host, nothing to do.\n");
    return 0;
}
#endif

