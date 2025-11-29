/*
 * MXU Raw Opcode Intrinsics
 * 
 * Ingenic XBurst MXU (Media Extension Unit) instructions using raw .word encodings.
 * Works with any MIPS assembler without MXU mnemonic support.
 *
 * Decoded by reverse-engineering kernel and OEM library binaries.
 * 
 * MXU instruction format (SPECIAL2 opcode = 0x1C = 28):
 * [31:26] = 011100 (SPECIAL2)
 * [25:21] = rs (base register for load/store, or source)
 * [20:16] = rt (GPR for transfers, or source)  
 * [15:11] = rd (destination field)
 * [10:6]  = sa (XR register number, or shift amount)
 * [5:0]   = function code
 *
 * Function codes:
 * 0x2E (46) = S32M2I - Move from XR to GPR
 * 0x2F (47) = S32I2M - Move from GPR to XR
 * 0x30 (48) = S32LDD related
 * 0x35 (53) = D16MUL/Q8MAC family
 * 0x3A (58) = D16MAC/Q8ACCE family
 *
 * Copyright (c) 2024 OpenSensor Project
 * SPDX-License-Identifier: MIT
 */

#ifndef _MXU_RAW_H_
#define _MXU_RAW_H_

#ifdef __mips__

/* Enable MXU by setting CU2 bit in CP0 Status register */
#define MXU_ENABLE() \
    __asm__ __volatile__( \
        ".set push\n" \
        ".set noreorder\n" \
        "mfc0 $t0, $12\n" \
        "lui $t1, 0x4000\n" \
        "or $t0, $t0, $t1\n" \
        "mtc0 $t0, $12\n" \
        ".set pop\n" \
        ::: "t0", "t1")

/*
 * S32I2M - Move from GPR to XR register
 * Encoding: 0x70000000 | (rt << 16) | (xr << 6) | 0x2F
 * For xr0: add 0x400 (bit 10 set)
 */
#define S32I2M_XR0(val) \
    __asm__ __volatile__(".word 0x7008042f" :: "r"(val))
#define S32I2M_XR1(val) \
    __asm__ __volatile__(".word 0x7008006f" :: "r"(val))
#define S32I2M_XR2(val) \
    __asm__ __volatile__(".word 0x700800af" :: "r"(val))
#define S32I2M_XR3(val) \
    __asm__ __volatile__(".word 0x700800ef" :: "r"(val))
#define S32I2M_XR4(val) \
    __asm__ __volatile__(".word 0x7008012f" :: "r"(val))
#define S32I2M_XR5(val) \
    __asm__ __volatile__(".word 0x7008016f" :: "r"(val))
#define S32I2M_XR6(val) \
    __asm__ __volatile__(".word 0x700801af" :: "r"(val))
#define S32I2M_XR7(val) \
    __asm__ __volatile__(".word 0x700801ef" :: "r"(val))

/*
 * S32M2I - Move from XR register to GPR
 * Encoding: 0x70000000 | (rt << 16) | (xr << 6) | 0x2E
 */
#define S32M2I_XR0() ({ \
    register unsigned int __v __asm__("t0"); \
    __asm__ __volatile__(".word 0x7008042e" : "=r"(__v)); \
    __v; })
#define S32M2I_XR1() ({ \
    register unsigned int __v __asm__("t0"); \
    __asm__ __volatile__(".word 0x7008006e" : "=r"(__v)); \
    __v; })
#define S32M2I_XR2() ({ \
    register unsigned int __v __asm__("t0"); \
    __asm__ __volatile__(".word 0x700800ae" : "=r"(__v)); \
    __v; })
#define S32M2I_XR3() ({ \
    register unsigned int __v __asm__("t0"); \
    __asm__ __volatile__(".word 0x700800ee" : "=r"(__v)); \
    __v; })
#define S32M2I_XR4() ({ \
    register unsigned int __v __asm__("t0"); \
    __asm__ __volatile__(".word 0x7008012e" : "=r"(__v)); \
    __v; })
#define S32M2I_XR5() ({ \
    register unsigned int __v __asm__("t0"); \
    __asm__ __volatile__(".word 0x7008016e" : "=r"(__v)); \
    __v; })
#define S32M2I_XR6() ({ \
    register unsigned int __v __asm__("t0"); \
    __asm__ __volatile__(".word 0x700801ae" : "=r"(__v)); \
    __v; })
#define S32M2I_XR7() ({ \
    register unsigned int __v __asm__("t0"); \
    __asm__ __volatile__(".word 0x700801ee" : "=r"(__v)); \
    __v; })

/*
 * S32LDD - Load 32-bit to XR from memory with offset 0
 * Encoding: 0x70000000 | (base_gpr << 21) | (xr << 16) | (xr << 6) | 0x2E
 *
 * Usage: Load from address in register, xr = *base
 */
#define MXU_S32LDD_OP(base_gpr, xr) \
    (0x70000000 | ((base_gpr) << 21) | ((xr) << 16) | ((xr) << 6) | 0x2E)

/* S32LDD using $v0 as base - common pattern */
#define S32LDD_V0_XR0()  __asm__ __volatile__(".word 0x7040002e")
#define S32LDD_V0_XR1()  __asm__ __volatile__(".word 0x7041006e")
#define S32LDD_V0_XR2()  __asm__ __volatile__(".word 0x704200ae")
#define S32LDD_V0_XR3()  __asm__ __volatile__(".word 0x704300ee")
#define S32LDD_V0_XR4()  __asm__ __volatile__(".word 0x7044012e")
#define S32LDD_V0_XR5()  __asm__ __volatile__(".word 0x7045016e")

/* S32LDD using $v1 as base */
#define S32LDD_V1_XR0()  __asm__ __volatile__(".word 0x7060002e")
#define S32LDD_V1_XR1()  __asm__ __volatile__(".word 0x7061006e")
#define S32LDD_V1_XR2()  __asm__ __volatile__(".word 0x706200ae")
#define S32LDD_V1_XR3()  __asm__ __volatile__(".word 0x706300ee")

/*
 * S32STD - Store 32-bit from XR to memory
 * Encoding appears to be similar but different function code
 * For now use GPR intermediary
 */
#define S32STD_VIA_GPR(xr_idx, ptr) do { \
    unsigned int __tmp = S32M2I_XR##xr_idx(); \
    *(unsigned int*)(ptr) = __tmp; \
} while(0)

/*
 * Q8MUL/Q8MAC - Quad 8-bit Multiply/Multiply-Accumulate
 * func=0x35 for Q8MUL/D16MUL family
 *
 * Encoding: 0x70000000 | (mode << 21) | (xra << 16) | (xrc << 11) | (xrd << 6) | 0x35
 *
 * Q8MAC modes (bits 25:21):
 *   0 = AA (add-add)
 *   Other modes TBD
 *
 * Operands (from mxu_media.h):
 *   xra = result/accumulator low
 *   xrb = source 1 (might be implicit or in different field)
 *   xrc = source 2
 *   xrd = accumulator high
 */
#define MXU_Q8MAC_OP(xra, xrc, xrd) \
    (0x70000000 | (0 << 21) | ((xra) << 16) | ((xrc) << 11) | ((xrd) << 6) | 0x35)

/* Q8MAC xr0, xr?, xr0, xr2, AA - common pattern seen in libvenus */
#define Q8MAC_XR0_XR0_XR2()  __asm__ __volatile__(".word 0x700000b5")
#define Q8MAC_XR1_XR1_XR2()  __asm__ __volatile__(".word 0x700108b5")

/*
 * D8SUM - Dual 8-bit Sum (for pooling)
 * func=0x3A family
 */

/*
 * Helper: Load 4 bytes to XR using pointer in v0
 */
static inline void mxu_load_xr0_from_v0(const void *ptr) {
    register const void *p __asm__("v0") = ptr;
    __asm__ __volatile__(".word 0x7040002e" :: "r"(p) : "memory");
}

static inline void mxu_load_xr1_from_v0(const void *ptr) {
    register const void *p __asm__("v0") = ptr;
    __asm__ __volatile__(".word 0x7041006e" :: "r"(p) : "memory");
}

static inline void mxu_load_xr2_from_v0(const void *ptr) {
    register const void *p __asm__("v0") = ptr;
    __asm__ __volatile__(".word 0x704200ae" :: "r"(p) : "memory");
}

static inline void mxu_load_xr3_from_v0(const void *ptr) {
    register const void *p __asm__("v0") = ptr;
    __asm__ __volatile__(".word 0x704300ee" :: "r"(p) : "memory");
}

#else /* !__mips__ */

/* Non-MIPS stubs */
#define MXU_ENABLE()

#endif /* __mips__ */

#endif /* _MXU_RAW_H_ */

