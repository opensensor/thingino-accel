/*
 * MXUv3 Instruction Test
 * Tests MXUv3 opcodes on T41 (XBurst2)
 *
 * T41 has MXUv3 with 32x VPR registers (512-bit each = 64 bytes)
 * NOT the older MXU1/MXU2 with XR registers!
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>

#ifdef __mips__
#include "mxuv3.h"

static inline long long get_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)tv.tv_sec * 1000000LL + tv.tv_usec;
}

/*
 * MXUv3 Instructions from kernel mxuv3.c:
 *
 * SA0 (Store Aligned 0) - Store VPR to memory:
 *   0x710000d5 | offset << 16 | vprn << 11 | n << 9
 *   where offset = byte offset / 32, n = which 256-bit half (0 or 1)
 *   Uses $t0 as base address
 *
 * LA0 (Load Aligned 0) - Load memory to VPR:
 *   Similar encoding (need to find exact opcode)
 *
 * CFCMXU - Read MXU control register:
 *   0x70100203 | reg << 11
 *   reg 0 = MIR (MXU ID Register)
 *   reg 1 = MCSR (MXU Control/Status Register)
 *
 * MFSUM - Move From Sum register:
 *   0x4a60000f | vss << 11 | vrd << 7
 */

/* Read MXU ID Register */
static inline uint32_t cfcmxu_mir(void) {
    uint32_t result;
    __asm__ __volatile__(
        ".set push\n"
        ".set noreorder\n"
        ".word 0x70100203\n"  /* CFCMXU $v0, mir (reg 0) */
        "move %0, $v0\n"
        ".set pop\n"
        : "=r"(result)
        :
        : "v0"
    );
    return result;
}

/* Read MXU Control/Status Register */
static inline uint32_t cfcmxu_mcsr(void) {
    uint32_t result;
    __asm__ __volatile__(
        ".set push\n"
        ".set noreorder\n"
        ".word 0x70100a03\n"  /* CFCMXU $v0, mcsr (reg 1 << 11) */
        "move %0, $v0\n"
        ".set pop\n"
        : "=r"(result)
        :
        : "v0"
    );
    return result;
}

/* SA0 - Store 256 bits from VPR to memory
 * Each VPR is 512-bit, so need 2 stores (n=0 and n=1)
 */
static inline void sa0_vpr0(void *ptr) {
    register void *base __asm__("t0") = ptr;
    __asm__ __volatile__(
        ".set push\n"
        ".set noreorder\n"
        ".word 0x710000d5\n"  /* SA0 vpr0[255:0] -> (t0+0) */
        ".word 0x710102d5\n"  /* SA0 vpr0[511:256] -> (t0+32) */
        ".set pop\n"
        :: "r"(base)
        : "memory"
    );
}

/* LA0 - Load 256 bits from memory to VPR
 * Encoding: 0x71001811 | offset << 16 | n << 14 | vprn << 6
 * where offset = byte offset / 32, n = which 256-bit half (0 or 1)
 * Uses $t0 as base address
 */
static inline void la0_vpr0(void *ptr) {
    register void *base __asm__("t0") = ptr;
    __asm__ __volatile__(
        ".set push\n"
        ".set noreorder\n"
        ".word 0x71001811\n"  /* LA0 vpr0[255:0] <- (t0+0), offset=0, n=0 */
        ".word 0x71015811\n"  /* LA0 vpr0[511:256] <- (t0+32), offset=1, n=1 */
        ".set pop\n"
        :: "r"(base)
        : "memory"
    );
}

/* Load to VPR1 */
static inline void la0_vpr1(void *ptr) {
    register void *base __asm__("t0") = ptr;
    __asm__ __volatile__(
        ".set push\n"
        ".set noreorder\n"
        ".word 0x71001851\n"  /* LA0 vpr1[255:0] <- (t0+0), vpr=1 */
        ".word 0x71015851\n"  /* LA0 vpr1[511:256] <- (t0+32) */
        ".set pop\n"
        :: "r"(base)
        : "memory"
    );
}

/* Store VPR1 */
static inline void sa0_vpr1(void *ptr) {
    register void *base __asm__("t0") = ptr;
    __asm__ __volatile__(
        ".set push\n"
        ".set noreorder\n"
        ".word 0x710008d5\n"  /* SA0 vpr1[255:0] -> (t0+0), vpr=1 */
        ".word 0x71010ad5\n"  /* SA0 vpr1[511:256] -> (t0+32) */
        ".set pop\n"
        :: "r"(base)
        : "memory"
    );
}

int main(void) {
    printf("MXUv3 Instruction Test (T41 XBurst2)\n");
    printf("====================================\n\n");

    /* Test 1: Read MXU identification registers */
    printf("Test 1: Read MXU control registers\n");

    uint32_t mir = cfcmxu_mir();
    uint32_t mcsr = cfcmxu_mcsr();

    printf("  MIR (ID Register):     0x%08x\n", mir);
    printf("  MCSR (Control/Status): 0x%08x\n", mcsr);

    /* Test 2: VPR register store (SA0) */
    printf("\nTest 2: VPR register store (SA0 instruction)\n");

    /* Allocate aligned buffer for VPR (64 bytes = 512 bits) */
    uint8_t *vpr_buf = aligned_alloc(64, 64);
    if (!vpr_buf) {
        printf("  Failed to allocate buffer\n");
        return 1;
    }

    memset(vpr_buf, 0xAA, 64);  /* Fill with pattern */

    printf("  Before SA0: first 8 bytes = %02x %02x %02x %02x %02x %02x %02x %02x\n",
           vpr_buf[0], vpr_buf[1], vpr_buf[2], vpr_buf[3],
           vpr_buf[4], vpr_buf[5], vpr_buf[6], vpr_buf[7]);

    /* Store VPR0 to buffer */
    sa0_vpr0(vpr_buf);

    printf("  After SA0:  first 8 bytes = %02x %02x %02x %02x %02x %02x %02x %02x\n",
           vpr_buf[0], vpr_buf[1], vpr_buf[2], vpr_buf[3],
           vpr_buf[4], vpr_buf[5], vpr_buf[6], vpr_buf[7]);

    /* Check if VPR0 was stored (values should change from 0xAA pattern) */
    int changed = 0;
    for (int i = 0; i < 64; i++) {
        if (vpr_buf[i] != 0xAA) {
            changed = 1;
            break;
        }
    }

    if (changed) {
        printf("  VPR0 store successful - data was written\n");
    } else {
        printf("  VPR0 store may have failed - data unchanged\n");
    }

    /* Test 3: Load/Store round-trip */
    printf("\nTest 3: VPR load/store round-trip (LA0/SA0)\n");

    /* Create test pattern */
    uint8_t *input_buf = aligned_alloc(64, 64);
    uint8_t *output_buf = aligned_alloc(64, 64);

    for (int i = 0; i < 64; i++) {
        input_buf[i] = i;  /* Pattern: 0, 1, 2, ... 63 */
        output_buf[i] = 0;
    }

    printf("  Input:  %02x %02x %02x %02x ... %02x %02x %02x %02x\n",
           input_buf[0], input_buf[1], input_buf[2], input_buf[3],
           input_buf[60], input_buf[61], input_buf[62], input_buf[63]);

    /* Load input -> VPR0, then Store VPR0 -> output */
    la0_vpr0(input_buf);
    sa0_vpr0(output_buf);

    printf("  Output: %02x %02x %02x %02x ... %02x %02x %02x %02x\n",
           output_buf[0], output_buf[1], output_buf[2], output_buf[3],
           output_buf[60], output_buf[61], output_buf[62], output_buf[63]);

    /* Verify */
    int match = 1;
    for (int i = 0; i < 64; i++) {
        if (input_buf[i] != output_buf[i]) {
            match = 0;
            printf("  Mismatch at byte %d: input=%02x, output=%02x\n",
                   i, input_buf[i], output_buf[i]);
            break;
        }
    }

    if (match) {
        printf("  Round-trip successful - all 64 bytes match!\n");
    }

    /* Test 4: VPR1 load/store */
    printf("\nTest 4: VPR1 load/store round-trip\n");

    for (int i = 0; i < 64; i++) {
        input_buf[i] = 64 + i;  /* Different pattern */
        output_buf[i] = 0;
    }

    la0_vpr1(input_buf);
    sa0_vpr1(output_buf);

    match = 1;
    for (int i = 0; i < 64; i++) {
        if (input_buf[i] != output_buf[i]) {
            match = 0;
            printf("  Mismatch at byte %d: input=%02x, output=%02x\n",
                   i, input_buf[i], output_buf[i]);
            break;
        }
    }

    if (match) {
        printf("  VPR1 round-trip successful!\n");
    }

    /* Test 5: Try vector compute instructions
     *
     * From libvenus analysis:
     * - func=0x34: Some kind of vector operation (rs=14)
     * - func=0x35: Another vector operation (rs=0)
     * - func=0x3a: Common operation (rs=2)
     *
     * Let's try 0x7041003a which appears frequently after LA0 loads
     */
    printf("\nTest 5: Vector compute (experimental)\n");

    /* Set up test data - use int8 for potential Q8 operations */
    int8_t *a = (int8_t*)input_buf;
    int8_t *b = (int8_t*)output_buf;
    int8_t *c = aligned_alloc(64, 64);
    int8_t *d = aligned_alloc(64, 64);

    /* Use simple values to understand the operation */
    for (int i = 0; i < 64; i++) {
        a[i] = 2;           /* All 2s */
        b[i] = 3;           /* All 3s */
        c[i] = 0;
        d[i] = 0;
    }

    printf("  A[0..7] = %d %d %d %d %d %d %d %d\n",
           a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
    printf("  B[0..7] = %d %d %d %d %d %d %d %d\n",
           b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]);

    /* Load A into VPR0, B into VPR1 */
    la0_vpr0(a);
    la0_vpr1(b);

    /* Test MFSUM/MTSUM instructions from kernel
     *
     * MFSUM: 0x4a60000f | vss << 11 - moves sum register to VPR0
     * MTSUM: 0x4a60001d | vsd << 6 - moves VPR0 to sum register
     *
     * There are 4 sum registers (VS0-VS3)
     */
    printf("  Testing MFSUM (Move From Sum) instruction...\n");

    /* Clear VPR0 first */
    memset(c, 0xAA, 64);
    la0_vpr0(c);

    /* MFSUM: Move VS0 to VPR0 */
    __asm__ __volatile__(
        ".set push\n"
        ".set noreorder\n"
        ".word 0x4a60000f\n"  /* MFSUM: vss=0 -> VPR0 */
        ".set pop\n"
        ::: "memory"
    );

    /* Store VPR0 to see what was in VS0 */
    sa0_vpr0(c);

    printf("  VS0 contents (first 16 bytes): ");
    for (int i = 0; i < 16; i++) {
        printf("%02x ", (uint8_t)c[i]);
    }
    printf("\n");

    /* Now test MTSUM: load data into VPR0, then move to VS0 */
    printf("  Testing MTSUM (Move To Sum) instruction...\n");

    /* Load test pattern into VPR0 */
    for (int i = 0; i < 64; i++) {
        a[i] = i + 1;  /* 1, 2, 3, ... 64 */
    }
    la0_vpr0(a);

    /* MTSUM: Move VPR0 to VS0 */
    __asm__ __volatile__(
        ".set push\n"
        ".set noreorder\n"
        ".word 0x4a60001d\n"  /* MTSUM: VPR0 -> vsd=0 */
        ".set pop\n"
        ::: "memory"
    );

    /* Clear VPR0 */
    memset(c, 0, 64);
    la0_vpr0(c);

    /* MFSUM: Move VS0 back to VPR0 */
    __asm__ __volatile__(
        ".set push\n"
        ".set noreorder\n"
        ".word 0x4a60000f\n"  /* MFSUM: vss=0 -> VPR0 */
        ".set pop\n"
        ::: "memory"
    );

    /* Store VPR0 to see if round-trip worked */
    sa0_vpr0(c);

    printf("  After MTSUM/MFSUM round-trip:\n");
    printf("  C[0..7] = %d %d %d %d %d %d %d %d\n",
           c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);

    int match2 = 1;
    for (int i = 0; i < 64; i++) {
        if (c[i] != a[i]) match2 = 0;
    }
    printf("  Round-trip %s!\n", match2 ? "successful" : "FAILED");

    /* Try classic MXU1 instructions with XR registers using raw opcodes.
     *
     * From kernel mxu-xburst.c:
     * S32M2I (XR -> GPR t0): 0x7008 + (xr << 6) + 0x2e
     * S32I2M (GPR t0 -> XR): 0x7008 + (xr << 6) + 0x2f
     *
     * XR0 uses 0x10 << 6 = 0x400 instead of 0
     */
    printf("\n  Testing classic MXU1 (XR registers)...\n");

    /* Test S32I2M / S32M2I round-trip */
    uint32_t val32 = 0x12345678;
    uint32_t result32 = 0;

    /* S32I2M: t0 -> XR1 = 0x7008006f */
    /* S32M2I: XR1 -> t0 = 0x7008006e */
    register uint32_t t0_reg __asm__("t0");

    t0_reg = val32;
    __asm__ __volatile__(
        ".word 0x7008006f\n"  /* S32I2M: t0 -> XR1 */
        :: "r"(t0_reg) : "memory"
    );

    t0_reg = 0;
    __asm__ __volatile__(
        ".word 0x7008006e\n"  /* S32M2I: XR1 -> t0 */
        : "=r"(t0_reg) :: "memory"
    );
    result32 = t0_reg;

    printf("  S32I2M/S32M2I round-trip: input=0x%08x, output=0x%08x %s\n",
           val32, result32, val32 == result32 ? "OK" : "FAIL");

    /* If XR works, test Q8MUL
     *
     * Q8MUL encoding (verified from Ingenic toolchain):
     * bits 0-5: func = 0x38
     * bits 6-9: xra
     * bits 10-13: xrb
     * bits 14-17: xrc
     * bits 18-21: xrd
     * bits 22-25: 0
     * bits 26-31: opcode = 0x1C (SPECIAL2)
     *
     * Q8MUL xra,xrb,xrc,xrd:
     *   xra = low bytes of (xrb * xrc)
     *   xrd = high bytes of (xrb * xrc)
     */

    /* Test XR2 */
    val32 = 0xAABBCCDD;
    t0_reg = val32;
    __asm__ __volatile__(".word 0x700800af\n" :: "r"(t0_reg) : "memory");
    t0_reg = 0;
    __asm__ __volatile__(".word 0x700800ae\n" : "=r"(t0_reg) :: "memory");
    result32 = t0_reg;
    printf("  XR2 round-trip: input=0x%08x, output=0x%08x %s\n",
           val32, result32, val32 == result32 ? "OK" : "FAIL");

    /* Test Q8MUL: xr1 = low(xr2 * xr3), xr4 = high(xr2 * xr3) */
    printf("\n  Testing Q8MUL (Quad 8-bit multiply)...\n");

    /* Load test values into XR2 and XR3 */
    /* XR2 = 0x02020202 (all 2s) */
    /* XR3 = 0x03030303 (all 3s) */
    /* Expected: XR1 = 0x06060606 (2*3=6 for each byte) */
    /* Expected: XR4 = 0x00000000 (no overflow for small values) */

    /* S32I2M xr2, $t0 = 0x700800af */
    t0_reg = 0x02020202;
    __asm__ __volatile__(".word 0x700800af\n" :: "r"(t0_reg) : "memory");

    /* S32I2M xr3, $t0 = 0x700800ef */
    t0_reg = 0x03030303;
    __asm__ __volatile__(".word 0x700800ef\n" :: "r"(t0_reg) : "memory");

    /* Q8MUL xr1,xr2,xr3,xr4 = 0x7010c878 */
    __asm__ __volatile__(".word 0x7010c878\n" ::: "memory");

    /* S32M2I xr1, $t0 = 0x7008006e */
    __asm__ __volatile__(".word 0x7008006e\n" : "=r"(t0_reg) :: "memory");
    uint32_t xr1_result = t0_reg;

    /* S32M2I xr4, $t0 = 0x7008010e */
    __asm__ __volatile__(".word 0x7008010e\n" : "=r"(t0_reg) :: "memory");
    uint32_t xr4_result = t0_reg;

    printf("  Q8MUL xr1,xr2,xr3,xr4:\n");
    printf("    XR2 = 0x02020202, XR3 = 0x03030303\n");
    printf("    XR1 (low)  = 0x%08x (expected 0x06060606)\n", xr1_result);
    printf("    XR4 (high) = 0x%08x (expected 0x00000000)\n", xr4_result);

    /* Test with larger values that will overflow */
    /* XR2 = 0x10101010 (16 each) */
    /* XR3 = 0x10101010 (16 each) */
    /* Expected: 16*16 = 256 = 0x100, so low=0x00, high=0x01 */
    t0_reg = 0x10101010;
    __asm__ __volatile__(".word 0x700800af\n" :: "r"(t0_reg) : "memory");
    __asm__ __volatile__(".word 0x700800ef\n" :: "r"(t0_reg) : "memory");
    __asm__ __volatile__(".word 0x7010c878\n" ::: "memory");
    __asm__ __volatile__(".word 0x7008006e\n" : "=r"(t0_reg) :: "memory");
    xr1_result = t0_reg;
    __asm__ __volatile__(".word 0x7008010e\n" : "=r"(t0_reg) :: "memory");
    xr4_result = t0_reg;

    printf("  Q8MUL with overflow:\n");
    printf("    XR2 = 0x10101010, XR3 = 0x10101010\n");
    printf("    XR1 (low)  = 0x%08x (expected 0x00000000)\n", xr1_result);
    printf("    XR4 (high) = 0x%08x (expected 0x01010101)\n", xr4_result);

    /* Test the full compute sequence from libvenus
     *
     * The struct loaded at 0x20720:
     * t4 = struct[0]   - base ptr for udi1 load
     * t3 = struct[4]   - base ptr for udi1 load
     * t2 = struct[8]
     * t1 = struct[12]
     * t0 = struct[16]  - config for func=0x39
     * a3 = struct[20]  - config for func=0x38
     * a2 = struct[24]
     * v1 = struct[28]  - offset for udi1
     * v0 = struct[32]
     */
    printf("\n  Testing compute with config values...\n");

    /* Load test data */
    for (int i = 0; i < 64; i++) { a[i] = 2; b[i] = 3; }
    LA0_VPR(29, a);  /* VPR29 = 2 */
    LA0_VPR(30, b);  /* VPR30 = 3 */

    /* Clear sum registers */
    memset(c, 0, 64);
    la0_vpr0(c);
    __asm__ __volatile__(".word 0x4a60001d\n" ::: "memory");  /* VS0 = 0 */

    /* Try different config values for t0 and a3 */
    register uint32_t t0_cfg __asm__("t0");
    register uint32_t a3_cfg __asm__("a3");

    /* Test 1: t0=0, a3=0 */
    t0_cfg = 0;
    a3_cfg = 0;
    LA0_VPR(29, a);
    LA0_VPR(30, b);
    __asm__ __volatile__(
        ".word 0x70fdf038\n"  /* func=0x38: setup lane 0 */
        ".word 0x7108ef79\n"  /* func=0x39: compute VPR29 */
        ".word 0x7108f7b9\n"  /* func=0x39: compute VPR30 */
        :: "r"(t0_cfg), "r"(a3_cfg) : "memory"
    );
    SA0_VPR(29, c);
    SA0_VPR(30, d);
    printf("  t0=0, a3=0: VPR29=%d, VPR30=%d\n", c[0], d[0]);

    /* Test 2: t0=1, a3=0 */
    t0_cfg = 1;
    a3_cfg = 0;
    LA0_VPR(29, a);
    LA0_VPR(30, b);
    __asm__ __volatile__(
        ".word 0x70fdf038\n"
        ".word 0x7108ef79\n"
        ".word 0x7108f7b9\n"
        :: "r"(t0_cfg), "r"(a3_cfg) : "memory"
    );
    SA0_VPR(29, c);
    SA0_VPR(30, d);
    printf("  t0=1, a3=0: VPR29=%d, VPR30=%d\n", c[0], d[0]);

    /* Test 3: t0=0x100, a3=0 */
    t0_cfg = 0x100;
    a3_cfg = 0;
    LA0_VPR(29, a);
    LA0_VPR(30, b);
    __asm__ __volatile__(
        ".word 0x70fdf038\n"
        ".word 0x7108ef79\n"
        ".word 0x7108f7b9\n"
        :: "r"(t0_cfg), "r"(a3_cfg) : "memory"
    );
    SA0_VPR(29, c);
    SA0_VPR(30, d);
    printf("  t0=0x100, a3=0: VPR29=%d, VPR30=%d\n", c[0], d[0]);

    /* Check sum register */
    __asm__ __volatile__(".word 0x4a60000f\n" ::: "memory");  /* MFSUM VS0 -> VPR0 */
    sa0_vpr0(c);
    printf("  VS0[0..3]=%d %d %d %d\n", c[0], c[1], c[2], c[3]);

    /* Summary */
    printf("  Pattern: a + b? or a * b? or something else?\n");

    free(c);
    free(d);
    free(input_buf);
    free(output_buf);
    free(vpr_buf);

    /* Test 6: Benchmark VPR load/store vs scalar memcpy */
    printf("\nTest 6: VPR load/store benchmark\n");

    #define BENCH_SIZE (1024 * 1024)  /* 1MB */
    #define BENCH_ITERS 10

    uint8_t *bench_src = aligned_alloc(64, BENCH_SIZE);
    uint8_t *bench_dst = aligned_alloc(64, BENCH_SIZE);

    /* Initialize source */
    for (int i = 0; i < BENCH_SIZE; i++) {
        bench_src[i] = i & 0xFF;
    }

    /* Benchmark scalar memcpy */
    printf("  Scalar memcpy (%d MB x %d iters)...", BENCH_SIZE/(1024*1024), BENCH_ITERS);
    fflush(stdout);
    volatile int dummy = 0;
    long long t0 = get_time_us();
    for (int iter = 0; iter < BENCH_ITERS; iter++) {
        memcpy(bench_dst, bench_src, BENCH_SIZE);
        dummy += bench_dst[0];  /* Prevent optimization */
    }
    long long t1 = get_time_us();
    long long scalar_us = t1 - t0;
    printf(" %lld us (%.1f MB/s)\n", scalar_us,
           (double)(BENCH_SIZE * BENCH_ITERS) / scalar_us);

    /* Clear destination */
    memset(bench_dst, 0, BENCH_SIZE);

    /* Benchmark VPR copy (64 bytes at a time) */
    printf("  VPR copy (%d MB x %d iters)...", BENCH_SIZE/(1024*1024), BENCH_ITERS);
    fflush(stdout);
    t0 = get_time_us();
    for (int iter = 0; iter < BENCH_ITERS; iter++) {
        for (int off = 0; off < BENCH_SIZE; off += 64) {
            LA0_VPR(0, bench_src + off);
            SA0_VPR(0, bench_dst + off);
        }
        dummy += bench_dst[0];
    }
    t1 = get_time_us();
    long long vpr_us = t1 - t0;
    printf(" %lld us (%.1f MB/s)\n", vpr_us,
           (double)(BENCH_SIZE * BENCH_ITERS) / vpr_us);

    /* Verify */
    int errors = 0;
    for (int i = 0; i < BENCH_SIZE; i++) {
        if (bench_src[i] != bench_dst[i]) errors++;
    }
    printf("  Verification: %d errors\n", errors);
    (void)dummy;

    free(bench_src);
    free(bench_dst);

    printf("\nMXUv3 test complete.\n");
    printf("\nNote: T41 uses MXUv3 with VPR registers (512-bit each).\n");
    printf("VPR0-VPR31 available for vector operations.\n");

    return 0;
}

#else
int main(void) {
    printf("MXU test requires MIPS target\n");
    return 1;
}
#endif

