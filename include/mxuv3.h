/*
 * MXUv3 (Media Extension Unit v3) intrinsics for Ingenic XBurst2 (T41)
 *
 * MXUv3 has 32 VPR registers, each 512 bits (64 bytes).
 * This is different from MXU1/MXU2 which had 16 XR registers (32 bits each).
 *
 * Instruction encodings derived from kernel source and libvenus disassembly.
 */

#ifndef _MXUV3_H_
#define _MXUV3_H_

#include <stdint.h>

#ifdef __mips__

/*
 * Cache prefetch for hiding memory latency
 * PREF hint=0 (load), hint=1 (store)
 */
#define PREFETCH_LOAD(addr) \
    __asm__ __volatile__("pref 0, 0(%0)" :: "r"(addr))
#define PREFETCH_LOAD_OFFSET(addr, offset) \
    __asm__ __volatile__("pref 0, %1(%0)" :: "r"(addr), "i"(offset))
#define PREFETCH_STORE(addr) \
    __asm__ __volatile__("pref 1, 0(%0)" :: "r"(addr))

/*
 * LA0 - Load Aligned 0
 * Loads 256 bits from memory to VPR register (need 2 loads for full 512-bit VPR)
 * Encoding: 0x71001811 | offset << 16 | n << 14 | vprn << 6
 *   offset = byte offset / 32
 *   n = which 256-bit half (0 = low, 1 = high)
 *   vprn = VPR register number (0-31)
 *   Uses $t0 as base address
 */
#define LA0_VPR(vpr, ptr) do { \
    register void *_base __asm__("t0") = (ptr); \
    __asm__ __volatile__( \
        ".set push\n" \
        ".set noreorder\n" \
        ".word %1\n" \
        ".word %2\n" \
        ".set pop\n" \
        :: "r"(_base), \
           "i"(0x71001811 | (0 << 16) | (0 << 14) | ((vpr) << 6)), \
           "i"(0x71001811 | (1 << 16) | (1 << 14) | ((vpr) << 6)) \
        : "memory" \
    ); \
} while(0)

/*
 * SA0 - Store Aligned 0
 * Stores 256 bits from VPR register to memory (need 2 stores for full 512-bit VPR)
 * Encoding: 0x710000d5 | offset << 16 | vprn << 11 | n << 9
 *   offset = byte offset / 32
 *   vprn = VPR register number (0-31)
 *   n = which 256-bit half (0 = low, 1 = high)
 *   Uses $t0 as base address
 */
#define SA0_VPR(vpr, ptr) do { \
    register void *_base __asm__("t0") = (ptr); \
    __asm__ __volatile__( \
        ".set push\n" \
        ".set noreorder\n" \
        ".word %1\n" \
        ".word %2\n" \
        ".set pop\n" \
        :: "r"(_base), \
           "i"(0x710000d5 | (0 << 16) | ((vpr) << 11) | (0 << 9)), \
           "i"(0x710000d5 | (1 << 16) | ((vpr) << 11) | (1 << 9)) \
        : "memory" \
    ); \
} while(0)

/* Convenience macros for specific VPR registers */
static inline void mxuv3_load_vpr0(void *ptr) { LA0_VPR(0, ptr); }
static inline void mxuv3_load_vpr1(void *ptr) { LA0_VPR(1, ptr); }
static inline void mxuv3_load_vpr2(void *ptr) { LA0_VPR(2, ptr); }
static inline void mxuv3_load_vpr3(void *ptr) { LA0_VPR(3, ptr); }

static inline void mxuv3_store_vpr0(void *ptr) { SA0_VPR(0, ptr); }
static inline void mxuv3_store_vpr1(void *ptr) { SA0_VPR(1, ptr); }
static inline void mxuv3_store_vpr2(void *ptr) { SA0_VPR(2, ptr); }
static inline void mxuv3_store_vpr3(void *ptr) { SA0_VPR(3, ptr); }

/*
 * CFCMXU - Copy From Control MXU register
 * Reads MXU control/status registers
 * Encoding: 0x70100203 | reg << 11
 *   reg 0 = MIR (MXU ID Register)
 *   reg 1 = MCSR (MXU Control/Status Register)
 */
static inline uint32_t mxuv3_read_mir(void) {
    uint32_t result;
    __asm__ __volatile__(
        ".set push\n"
        ".set noreorder\n"
        ".word 0x70100203\n"
        "move %0, $v0\n"
        ".set pop\n"
        : "=r"(result) :: "v0"
    );
    return result;
}

/*
 * VPR register size and alignment requirements
 */
#define MXUV3_VPR_SIZE      64   /* 512 bits = 64 bytes */
#define MXUV3_VPR_ALIGN     64   /* Must be 64-byte aligned */
#define MXUV3_VPR_COUNT     32   /* 32 VPR registers available */

/* Number of elements per VPR for different data types */
#define MXUV3_VPR_INT8_COUNT   64   /* 64 x 8-bit */
#define MXUV3_VPR_INT16_COUNT  32   /* 32 x 16-bit */
#define MXUV3_VPR_INT32_COUNT  16   /* 16 x 32-bit */
#define MXUV3_VPR_FLOAT_COUNT  16   /* 16 x 32-bit float */

/*
 * VPR_CONCAT - Concatenate two VPR registers
 * func=0x38: VPR[sa] = {VPR[rt], VPR[rd]}
 * First 32 bytes from VPR[rt], second 32 bytes from VPR[rd]
 * Encoding: 0x70000038 | (7 << 21) | (rt << 16) | (rd << 11) | (sa << 6)
 */
#define VPR_CONCAT(dst, src1, src2) do { \
    __asm__ __volatile__( \
        ".word %0\n" \
        :: "i"(0x70000038 | (7 << 21) | ((src1) << 16) | ((src2) << 11) | ((dst) << 6)) \
        : "memory" \
    ); \
} while(0)

/*
 * VPR_COPY - Copy VPR[src] to VPR0
 * c2 func=0x08: VPR0 = VPR[rd]
 * Encoding: 0x4a000008 | (rd << 11)
 */
#define VPR_COPY_TO_VPR0(src) do { \
    __asm__ __volatile__( \
        ".word %0\n" \
        :: "i"(0x4a000008 | ((src) << 11)) \
        : "memory" \
    ); \
} while(0)

/*
 * VPR_SETUP - Setup/initialization for MAC operations
 * func=0x30: Setup VPR29 and VPR30 for MAC
 */
#define VPR_SETUP_MAC() do { \
    __asm__ __volatile__( \
        ".word 0x70f0ef70\n" \
        ".word 0x70f0f7b0\n" \
        ::: "memory" \
    ); \
} while(0)

/*
 * Load VPR29 and VPR30 (commonly used for weights in MAC)
 */
static inline void mxuv3_load_vpr29(void *ptr) { LA0_VPR(29, ptr); }
static inline void mxuv3_load_vpr30(void *ptr) { LA0_VPR(30, ptr); }
static inline void mxuv3_store_vpr29(void *ptr) { SA0_VPR(29, ptr); }
static inline void mxuv3_store_vpr30(void *ptr) { SA0_VPR(30, ptr); }

/*
 * Fast memory copy using VPR registers (64 bytes at a time)
 * Both src and dst must be 64-byte aligned
 * Size must be multiple of 64 bytes
 */
static inline void mxuv3_memcpy_64(void *dst, const void *src, size_t size) {
    register void *_src __asm__("t0");
    register void *_dst __asm__("t0");
    uint8_t *s = (uint8_t *)src;
    uint8_t *d = (uint8_t *)dst;

    for (size_t i = 0; i < size; i += 64) {
        /* Load 64 bytes to VPR0 */
        _src = s + i;
        __asm__ __volatile__(
            ".word 0x71001811\n"  /* LA0 VPR0 low half */
            ".word 0x71015811\n"  /* LA0 VPR0 high half */
            :: "r"(_src) : "memory"
        );

        /* Store 64 bytes from VPR0 */
        _dst = d + i;
        __asm__ __volatile__(
            ".word 0x710000d5\n"  /* SA0 VPR0 low half */
            ".word 0x710102d5\n"  /* SA0 VPR0 high half */
            :: "r"(_dst) : "memory"
        );
    }
}

/*
 * Fast memory copy using 4 VPR registers (256 bytes at a time)
 * For larger blocks - uses VPR0-3 for better throughput
 */
static inline void mxuv3_memcpy_256(void *dst, const void *src, size_t size) {
    register void *_base __asm__("t0");
    uint8_t *s = (uint8_t *)src;
    uint8_t *d = (uint8_t *)dst;
    size_t i;

    /* Process full 256-byte blocks */
    for (i = 0; i + 255 < size; i += 256) {
        /* Load VPR0 (bytes 0-63) */
        _base = s + i;
        __asm__ __volatile__(
            ".word 0x71001811\n .word 0x71015811\n"
            :: "r"(_base) : "memory"
        );
        /* Load VPR1 (bytes 64-127) */
        _base = s + i + 64;
        __asm__ __volatile__(
            ".word 0x71001851\n .word 0x71015851\n"
            :: "r"(_base) : "memory"
        );
        /* Load VPR2 (bytes 128-191) */
        _base = s + i + 128;
        __asm__ __volatile__(
            ".word 0x71001891\n .word 0x71015891\n"
            :: "r"(_base) : "memory"
        );
        /* Load VPR3 (bytes 192-255) */
        _base = s + i + 192;
        __asm__ __volatile__(
            ".word 0x710018d1\n .word 0x710158d1\n"
            :: "r"(_base) : "memory"
        );

        /* Store VPR0 (bytes 0-63) */
        _base = d + i;
        __asm__ __volatile__(
            ".word 0x710000d5\n .word 0x710102d5\n"
            :: "r"(_base) : "memory"
        );
        /* Store VPR1 (bytes 64-127) */
        _base = d + i + 64;
        __asm__ __volatile__(
            ".word 0x710008d5\n .word 0x71010ad5\n"
            :: "r"(_base) : "memory"
        );
        /* Store VPR2 (bytes 128-191) */
        _base = d + i + 128;
        __asm__ __volatile__(
            ".word 0x710010d5\n .word 0x710112d5\n"
            :: "r"(_base) : "memory"
        );
        /* Store VPR3 (bytes 192-255) */
        _base = d + i + 192;
        __asm__ __volatile__(
            ".word 0x710018d5\n .word 0x71011ad5\n"
            :: "r"(_base) : "memory"
        );
    }

    /* Handle remaining 64-byte blocks */
    for (; i + 63 < size; i += 64) {
        _base = s + i;
        __asm__ __volatile__(
            ".word 0x71001811\n .word 0x71015811\n"
            :: "r"(_base) : "memory"
        );
        _base = d + i;
        __asm__ __volatile__(
            ".word 0x710000d5\n .word 0x710102d5\n"
            :: "r"(_base) : "memory"
        );
    }
}

/*
 * Fast memory set using VPR (set 64 bytes at a time)
 * Value is broadcast to all 64 bytes
 */
static inline void mxuv3_memset_64(void *dst, uint8_t value, size_t size) {
    register void *_base __asm__("t0");
    uint8_t *d = (uint8_t *)dst;

    /* Create a pattern buffer with the value repeated */
    uint8_t pattern[64] __attribute__((aligned(64)));
    for (int i = 0; i < 64; i++) pattern[i] = value;

    /* Load pattern to VPR0 */
    _base = pattern;
    __asm__ __volatile__(
        ".word 0x71001811\n .word 0x71015811\n"
        :: "r"(_base) : "memory"
    );

    /* Store to all destination blocks */
    for (size_t i = 0; i < size; i += 64) {
        _base = d + i;
        __asm__ __volatile__(
            ".word 0x710000d5\n .word 0x710102d5\n"
            :: "r"(_base) : "memory"
        );
    }
}

/*
 * ============================================================================
 * MXUv3 COMPUTE INSTRUCTIONS (COP2 opcode = 0x12)
 * ============================================================================
 *
 * COP2 instruction format (32-bit):
 * [31:26] = 010010 (COP2 = 0x12 = 18)
 * [25:21] = rs (operation class selector)
 * [20:16] = rt (source VPR register 2)
 * [15:11] = rd (destination VPR register)
 * [10:6]  = sa (source VPR register 1)
 * [5:0]   = fn (function code within operation class)
 *
 * Known rs values (operation classes):
 *   rs=19: Multiply operations (fn=35 = MUL)
 *   rs=20: Add/Sub operations (fn=3 = ADD, fn=11 = SUB)
 *
 * Binary operation formula: VPR[rd] = VPR[sa] op VPR[rt]
 * Unary operation (same reg): VPR[rd] = op(VPR[rd]) when rd=rt=sa
 *
 * Example encodings:
 *   ADD: 0x4a84__83 = rs=20, fn=3
 *   SUB: 0x4a84__8b = rs=20, fn=11
 *   MUL: 0x4a64__a3 = rs=19, fn=35
 */

    /*
     * Build COP2 instruction encoding
     * op=0x12 (COP2), remaining fields as parameters
     */
    #define MXUV3_COP2_INST(rs, rt, rd, sa, fn) \
        (0x48000000 | ((rs) << 21) | ((rt) << 16) | ((rd) << 11) | ((sa) << 6) | (fn))

    /*
     * Sum register access (VSR0-VSR3)
     *
     * These instructions use COP2 with rs=19 (0b10011) as the operation class.
     * The mapping of fields is reconstructed as:
     *   SUMZ   vsd           -> rs=19, rt=0,   rd=0,   sa=vsd, fn=0x1c
     *   MFSUM  vrd, vss      -> rs=19, rt=0,   rd=vss, sa=vrd, fn=0x0f
     *   MFSUMZ vrd, vsd      -> rs=19, rt=0,   rd=vsd, sa=vrd, fn=0x1e
     *   MTSUM  vsd, vrs      -> rs=19, rt=vrs, rd=0,   sa=vsd, fn=0x1d
     *   MXSUM  vrd, vrs, vsd -> rs=19, rt=vrs, rd=vsd, sa=vrd, fn=0x1f
     *
     * Here vrd/vrs are VPR indices and vss/vsd are VSR indices (0-3).
     */

    #define MXUV3_SUMZ(vsd) do { \
        __asm__ __volatile__( \
            ".word %0\n" \
            :: "i"(MXUV3_COP2_INST(19, 0, 0, (vsd), 0x1c)) \
            : "memory" \
        ); \
    } while (0)

    #define MXUV3_MFSUM(vrd, vss) do { \
        __asm__ __volatile__( \
            ".word %0\n" \
            :: "i"(MXUV3_COP2_INST(19, 0, (vss), (vrd), 0x0f)) \
            : "memory" \
        ); \
    } while (0)

    #define MXUV3_MFSUMZ(vrd, vsd) do { \
        __asm__ __volatile__( \
            ".word %0\n" \
            :: "i"(MXUV3_COP2_INST(19, 0, (vsd), (vrd), 0x1e)) \
            : "memory" \
        ); \
    } while (0)

    #define MXUV3_MTSUM(vsd, vrs) do { \
        __asm__ __volatile__( \
            ".word %0\n" \
            :: "i"(MXUV3_COP2_INST(19, (vrs), 0, (vsd), 0x1d)) \
            : "memory" \
        ); \
    } while (0)

    #define MXUV3_MXSUM(vrd, vrs, vsd) do { \
        __asm__ __volatile__( \
            ".word %0\n" \
            :: "i"(MXUV3_COP2_INST(19, (vrs), (vsd), (vrd), 0x1f)) \
            : "memory" \
        ); \
    } while (0)

/*
 * VPR_ADD - Vector Add (16 floats) - IN-PLACE
 * VPR[dst] = VPR[src] + VPR[dst]
 * Hardware constraint: rd must equal sa
 * Encoding: rs=20, rt=src, rd=dst, sa=dst, fn=3
 */
#define VPR_ADD(dst, src) do { \
    __asm__ __volatile__( \
        ".word %0\n sync\n" \
        :: "i"(MXUV3_COP2_INST(20, src, dst, dst, 3)) \
        : "memory" \
    ); \
} while(0)

/*
 * VPR_SUB - Vector Subtract (16 floats) - IN-PLACE
 * VPR[dst] = VPR[src] - VPR[dst]
 * Hardware constraint: rd must equal sa
 * Encoding: rs=20, rt=src, rd=dst, sa=dst, fn=11
 */
#define VPR_SUB(dst, src) do { \
    __asm__ __volatile__( \
        ".word %0\n sync\n" \
        :: "i"(MXUV3_COP2_INST(20, src, dst, dst, 11)) \
        : "memory" \
    ); \
} while(0)

/*
 * VPR_MUL - Vector Multiply (16 floats) - IN-PLACE
 * VPR[dst] = VPR[src] * VPR[dst]
 * Hardware constraint: rd must equal sa
 * Encoding: rs=19, rt=src, rd=dst, sa=dst, fn=35
 */
#define VPR_MUL(dst, src) do { \
    __asm__ __volatile__( \
        ".word %0\n sync\n" \
        :: "i"(MXUV3_COP2_INST(19, src, dst, dst, 35)) \
        : "memory" \
    ); \
} while(0)

/*
 * VPR_SQR - Vector Square (Unary: same register for all operands)
 * VPR[reg] = VPR[reg] * VPR[reg]
 * This is MUL with the same register for src1, src2, and dst
 * Encoding: rs=19, fn=35, rt=rd=sa=reg
 */
#define VPR_SQR(reg) do { \
    __asm__ __volatile__( \
        ".word %0\n sync\n" \
        :: "i"(MXUV3_COP2_INST(19, reg, reg, reg, 35)) \
        : "memory" \
    ); \
} while(0)

/*
 * VPR_DBL - Vector Double (Unary: x + x = 2x)
 * VPR[reg] = VPR[reg] + VPR[reg]
 * This is ADD with the same register
 * Encoding: rs=20, fn=3, rt=rd=sa=reg
 */
#define VPR_DBL(reg) do { \
    __asm__ __volatile__( \
        ".word %0\n sync\n" \
        :: "i"(MXUV3_COP2_INST(20, reg, reg, reg, 3)) \
        : "memory" \
    ); \
} while(0)

/*
 * VPR_ZERO - Vector Zero (Unary: x - x = 0)
 * VPR[reg] = VPR[reg] - VPR[reg] = 0
 * This is SUB with the same register
 * Encoding: rs=20, fn=11, rt=rd=sa=reg
 */
#define VPR_ZERO(reg) do { \
    __asm__ __volatile__( \
        ".word %0\n sync\n" \
        :: "i"(MXUV3_COP2_INST(20, reg, reg, reg, 11)) \
        : "memory" \
    ); \
} while(0)

/*
 * Inline functions for VPR arithmetic (convenient for C usage)
 * These operate on VPR0-3 which are commonly used for computation
 */

/* VPR0 = VPR1 + VPR0 (in-place add) */
static inline void mxuv3_add_vpr0_vpr1(void) {
    VPR_ADD(0, 1);
}

/* VPR0 = VPR1 - VPR0 (in-place sub) */
static inline void mxuv3_sub_vpr0_vpr1(void) {
    VPR_SUB(0, 1);
}

/* VPR0 = VPR1 * VPR0 (in-place mul) */
static inline void mxuv3_mul_vpr0_vpr1(void) {
    VPR_MUL(0, 1);
}

/* VPR0 = VPR0 * VPR0 (square) */
static inline void mxuv3_sqr_vpr0(void) {
    VPR_SQR(0);
}

/* VPR0 = VPR0 + VPR0 (double) */
static inline void mxuv3_dbl_vpr0(void) {
    VPR_DBL(0);
}

/* VPR0 = 0 */
static inline void mxuv3_zero_vpr0(void) {
    VPR_ZERO(0);
}

/*
 * =============================================================================
 * MAX/MIN Instructions (from MXU3 PDF Section 3.7.30-3.7.35)
 * =============================================================================
 *
 * These are critical for neural network activation functions (ReLU) and pooling.
 *
 * Encoding: COP2 with rs=16 (0x10)
 *   010010 10000 vrs vrp vrd function
 *   bits:  31-26 25-21 20-16 15-11 10-6  5-0
 *
 * Function codes:
 *   MAXSB (signed byte):     0111 00 = 0x1C
 *   MAXSH (signed halfword): 0111 01 = 0x1D
 *   MAXSW (signed word):     0111 10 = 0x1E
 *   MINSB (signed byte):     0101 00 = 0x14
 *   MINSH (signed halfword): 0101 01 = 0x15
 *   MINSW (signed word):     0101 10 = 0x16
 *
 * Operation: VPR[vrd] = max/min(VPR[vrs], VPR[vrp]) element-wise
 *
 * For float32: Use signed word (W) format - IEEE754 float bit patterns
 * compare correctly with signed integer comparison for positive values
 * and for values with same sign.
 */

/* MAXSW - Max Signed Word (16 x int32, works for positive floats) */
#define VPR_MAXSW(vrd, vrs, vrp) do { \
    __asm__ __volatile__( \
        ".word %0\n sync\n" \
        :: "i"(0x48000000 | (16 << 21) | ((vrs) << 16) | ((vrp) << 11) | ((vrd) << 6) | 0x1E) \
        : "memory" \
    ); \
} while(0)

/* MINSW - Min Signed Word (16 x int32, works for positive floats) */
#define VPR_MINSW(vrd, vrs, vrp) do { \
    __asm__ __volatile__( \
        ".word %0\n sync\n" \
        :: "i"(0x48000000 | (16 << 21) | ((vrs) << 16) | ((vrp) << 11) | ((vrd) << 6) | 0x16) \
        : "memory" \
    ); \
} while(0)

/* MAXUB - Max Unsigned Byte (64 x uint8) - for quantized networks */
#define VPR_MAXUB(vrd, vrs, vrp) do { \
    __asm__ __volatile__( \
        ".word %0\n sync\n" \
        :: "i"(0x48000000 | (16 << 21) | ((vrs) << 16) | ((vrp) << 11) | ((vrd) << 6) | 0x08) \
        : "memory" \
    ); \
} while(0)

/* MINUB - Min Unsigned Byte (64 x uint8) - for quantized networks */
#define VPR_MINUB(vrd, vrs, vrp) do { \
    __asm__ __volatile__( \
        ".word %0\n sync\n" \
        :: "i"(0x48000000 | (16 << 21) | ((vrs) << 16) | ((vrp) << 11) | ((vrd) << 6) | 0x00) \
        : "memory" \
    ); \
} while(0)

/*
 * Convenience macros for common patterns:
 *
 * ReLU: y = max(x, 0)
 *   Load zeros to VPR[zero_reg], input to VPR[in_reg]
 *   VPR_MAXSW(out_reg, in_reg, zero_reg)
 *
 * MaxPool 2x2: find max of 4 values
 *   VPR_MAXSW(temp, v0, v1)  // temp = max(v0, v1)
 *   VPR_MAXSW(temp, temp, v2) // temp = max(temp, v2)
 *   VPR_MAXSW(out, temp, v3)  // out = max(temp, v3)
 */

/*
 * Instruction encodings reference table:
 *
 * | Operation | rs | fn | Hex Pattern     | Formula                    |
 * |-----------|----|----|-----------------|----------------------------|
 * | ADD       | 20 |  3 | 0x4a84xxxx + 83 | VPR[rd] = VPR[sa] + VPR[rt]|
 * | SUB       | 20 | 11 | 0x4a84xxxx + 8b | VPR[rd] = VPR[sa] - VPR[rt]|
 * | MUL       | 19 | 35 | 0x4a64xxxx + a3 | VPR[rd] = VPR[sa] * VPR[rt]|
 * | SQR       | 19 | 35 | (unary MUL)     | VPR[rd] = VPR[rd]^2        |
 * | DBL       | 20 |  3 | (unary ADD)     | VPR[rd] = VPR[rd] * 2      |
 * | ZERO      | 20 | 11 | (unary SUB)     | VPR[rd] = 0                |
 *
 * ============================================================================
 * IMPORTANT: DIV, SQRT, RSQRT, RECIP do NOT exist in MXUv3!
 * ============================================================================
 *
 * Analysis of libvenus.so confirms that division and square root operations
 * use the standard MIPS FPU instructions (div.s, sqrt.s), NOT MXU:
 *   - 511 occurrences of FPU div.s in libvenus
 *   - 35 occurrences of FPU sqrt.s in libvenus
 *   - 0 occurrences of any MXU reciprocal instruction
 *
 * For neural network inference, division is typically avoided by:
 *   1. Pre-computing reciprocals during model quantization (on host CPU)
 *   2. Storing 1/sqrt(variance) as weights for batch norm fusion
 *   3. Using MXU MUL with pre-computed reciprocals at runtime
 *
 * Example batch norm optimization:
 *   Instead of:  y = (x - mean) / sqrt(variance)
 *   Compute:     scale = 1 / sqrt(variance)  [on CPU, store in model]
 *   At runtime:  y = (x - mean) * scale      [MXU SUB + MUL]
 *
 * For Mars runtime: use scalar FPU for any division/sqrt, or pre-compute.
 */

/*
 * ============================================================================
 * INT8 MAC Instructions (S{n}MAC family)
 * ============================================================================
 *
 * These perform dot products of INT8 vectors with 32-bit accumulation into VSR.
 *
 * Instruction format:
 *   S{Lseg}MAC{signedness}{B} vsd, vrs, vrp
 *
 * Where:
 *   Lseg = segment length (1,2,4,8,16,32 words)
 *   signedness = UUB (u8×u8), SUB (s8×u8), SSB (s8×s8)
 *   vsd = destination VSR register (0-3)
 *   vrs, vrp = source VPR registers
 *
 * Operation (for S4MACSSB as example):
 *   M = (Lseg * 32) / 8 = 16 bytes per segment
 *   For each segment j:
 *     sum = 0
 *     for i in 0..15:
 *       sum += signed(vrs[byte,i]) * signed(vrp[byte,i])
 *     VSR[vsd][word,j] += sum
 *
 * Encoding (COP2, based on PDF analysis):
 *   [31:26] = 010010 (COP2 = 0x12)
 *   [25:21] = Lseg encoding
 *   [20:16] = vrs (source 1)
 *   [15:11] = vrp (source 2)
 *   [10:6]  = vsd (dest VSR)
 *   [5:0]   = function code
 *
 * Function codes for MAC:
 *   MACUUB (unsigned×unsigned byte): 0x20
 *   MACSUB (signed×unsigned byte):   0x21
 *   MACSSB (signed×signed byte):     0x22
 *
 * Lseg encoding (bits 25:21):
 *   S1  = 0, S2  = 1, S4  = 2, S8  = 3, S16 = 4, S32 = 5
 */

/*
 * INT8 MAC instruction encoding from MXUv3 PDF:
 *
 * Decoded from PDF example: S4MACSSB vsr0[0], vr0, vr1 = 0x4BC00F08
 *
 * Format (standard COP2):
 *   [31:26] opcode = 0x12 (COP2)
 *   [25:21] rs = 30 (0x1E) for MACSSB, 29 for MACSUB, 28 for MACUUB
 *   [20:16] rt = vrs (first source VPR)
 *   [15:11] rd = vrp (second source VPR)
 *   [10:6]  sa = (mode << 2) | vsd, where mode=7 for S4
 *   [5:0]   func = 8 for S4 segment
 *
 * Signedness (rs field):
 *   rs=28 (0x1C) = MACUUB (U×U)
 *   rs=29 (0x1D) = MACSUB (S×U)
 *   rs=30 (0x1E) = MACSSB (S×S)
 */

#define MXUV3_MACUUB_RS  28  /* U×U */
#define MXUV3_MACSUB_RS  29  /* S×U */
#define MXUV3_MACSSB_RS  30  /* S×S byte */

#define MXUV3_S4_MODE    7   /* Mode bits for S4 segment */
#define MXUV3_S4_FUNC    8   /* Function code for S4 */

/* Build S4MAC instruction: opcode=0x12, rs=type, rt=vrs, rd=vrp, sa=(mode<<2)|vsd, func=8 */
#define MXUV3_S4MAC_INST(rs, vrs, vrp, vsd) \
    (0x48000000 | ((rs) << 21) | ((vrs) << 16) | ((vrp) << 11) | \
     ((MXUV3_S4_MODE << 2 | (vsd)) << 6) | MXUV3_S4_FUNC)

/*
 * S4MACSSB - 4-word segment, signed×signed byte MAC
 * Encoding verified: S4MACSSB(0, 0, 1) = 0x4BC00F08
 *
 * Computes 4 dot products of 16 bytes each, accumulating to VSR:
 *   VSR[vsd][0] += dot(vrs[0:15], vrp[0:15])
 *   VSR[vsd][1] += dot(vrs[16:31], vrp[16:31])
 *   VSR[vsd][2] += dot(vrs[32:47], vrp[32:47])
 *   VSR[vsd][3] += dot(vrs[48:63], vrp[48:63])
 */
#define S4MACSSB(vsd, vrs, vrp) do { \
    __asm__ __volatile__( \
        ".word %0\n sync\n" \
        :: "i"(MXUV3_S4MAC_INST(MXUV3_MACSSB_RS, vrs, vrp, vsd)) \
        : "memory" \
    ); \
} while(0)

/* S4MACSUB - 4-word segment, signed×unsigned byte MAC */
#define S4MACSUB(vsd, vrs, vrp) do { \
    __asm__ __volatile__( \
        ".word %0\n sync\n" \
        :: "i"(MXUV3_S4MAC_INST(MXUV3_MACSUB_RS, vrs, vrp, vsd)) \
        : "memory" \
    ); \
} while(0)

/* S4MACUUB - 4-word segment, unsigned×unsigned byte MAC */
#define S4MACUUB(vsd, vrs, vrp) do { \
    __asm__ __volatile__( \
        ".word %0\n sync\n" \
        :: "i"(MXUV3_S4MAC_INST(MXUV3_MACUUB_RS, vrs, vrp, vsd)) \
        : "memory" \
    ); \
} while(0)

/*
 * S1MACSSB - 1-word segment (whole VPR as single segment)
 * Computes single dot product of all 64 bytes
 * Result: VSR[vsd] += sum(vrs[0:63] * vrp[0:63])
 * Uses func=0 for S1 segment (tentative)
 */
#define MXUV3_S1_FUNC    0
#define S1MACSSB(vsd, vrs, vrp) do { \
    __asm__ __volatile__( \
        ".word %0\n sync\n" \
        :: "i"(0x48000000 | (MXUV3_MACSSB_RS << 21) | ((vrs) << 16) | ((vrp) << 11) | \
               ((MXUV3_S4_MODE << 2 | (vsd)) << 6) | MXUV3_S1_FUNC) \
        : "memory" \
    ); \
} while(0)

/*
 * Sum Register operations (from docs/mxuv3_instructions.md)
 *
 * Encoding: COP2 with rs=19 (0x4a600000 base)
 *   SUMZ(vsd)           : rt=0, rd=0,   sa=vsd, fn=0x1c
 *   MFSUM(vrd, vss)     : rt=0, rd=vss, sa=vrd, fn=0x0f
 *   MFSUMZ(vrd, vsd)    : rt=0, rd=vsd, sa=vrd, fn=0x1e
 *   MTSUM(vsd, vrs)     : rt=vrs, rd=0, sa=vsd, fn=0x1d
 */

/* SUMZ - Zero sum register */
#define VSR_ZERO(vsr) do { \
    __asm__ __volatile__( \
        ".word %0\n sync\n" \
        :: "i"(0x4a60001c | ((vsr) << 6)) \
        : "memory" \
    ); \
} while(0)

/* MFSUM - Move From SUM register to VPR (non-destructive) */
#define MFSUM(vrd, vss) do { \
    __asm__ __volatile__( \
        ".word %0\n sync\n" \
        :: "i"(0x4a60000f | ((vss) << 11) | ((vrd) << 6)) \
        : "memory" \
    ); \
} while(0)

/* MFSUMZ - Move From SUM register to VPR and Zero the sum register */
#define MFSUMZ(vrd, vsd) do { \
    __asm__ __volatile__( \
        ".word %0\n sync\n" \
        :: "i"(0x4a60001e | ((vsd) << 11) | ((vrd) << 6)) \
        : "memory" \
    ); \
} while(0)

/* MTSUM - Move To SUM register from VPR */
#define MTSUM(vsd, vrs) do { \
    __asm__ __volatile__( \
        ".word %0\n sync\n" \
        :: "i"(0x4a60001d | ((vrs) << 16) | ((vsd) << 6)) \
        : "memory" \
    ); \
} while(0)

#else /* !__mips__ */

/* Stub implementations for non-MIPS builds */
#define LA0_VPR(vpr, ptr) ((void)0)
#define SA0_VPR(vpr, ptr) ((void)0)

static inline void mxuv3_load_vpr0(void *ptr) { (void)ptr; }
static inline void mxuv3_load_vpr1(void *ptr) { (void)ptr; }
static inline void mxuv3_store_vpr0(void *ptr) { (void)ptr; }
static inline void mxuv3_store_vpr1(void *ptr) { (void)ptr; }

static inline uint32_t mxuv3_read_mir(void) { return 0; }

/* Stub compute macros for non-MIPS builds */
#define VPR_ADD(dst, src1, src2) ((void)0)
#define VPR_SUB(dst, src1, src2) ((void)0)
#define VPR_MUL(dst, src1, src2) ((void)0)
#define VPR_SQR(reg) ((void)0)
#define VPR_DBL(reg) ((void)0)
#define VPR_ZERO(reg) ((void)0)
#define VPR_MAXSW(vrd, vrs, vrp) ((void)0)
#define VPR_MINSW(vrd, vrs, vrp) ((void)0)
#define VPR_MAXUB(vrd, vrs, vrp) ((void)0)
#define VPR_MINUB(vrd, vrs, vrp) ((void)0)

/* INT8 MAC stubs */
#define S4MACSSB(vsd, vrs, vrp) ((void)0)
#define S4MACSUB(vsd, vrs, vrp) ((void)0)
#define S4MACUUB(vsd, vrs, vrp) ((void)0)
#define S1MACSSB(vsd, vrs, vrp) ((void)0)
#define MFSUM(vrd, vsr) ((void)0)
#define MFSUMZ(vrd, vsr) ((void)0)
#define VSR_ZERO(vsr) ((void)0)

static inline void mxuv3_add_vpr0_vpr1_vpr2(void) {}
static inline void mxuv3_sub_vpr0_vpr1_vpr2(void) {}
static inline void mxuv3_mul_vpr0_vpr1_vpr2(void) {}
static inline void mxuv3_sqr_vpr0(void) {}
static inline void mxuv3_dbl_vpr0(void) {}
static inline void mxuv3_zero_vpr0(void) {}

#define MXUV3_VPR_SIZE      64
#define MXUV3_VPR_ALIGN     64
#define MXUV3_VPR_COUNT     32

/* Fallback implementations using standard memcpy/memset */
#include <string.h>
static inline void mxuv3_memcpy_64(void *dst, const void *src, size_t size) {
    memcpy(dst, src, size);
}
static inline void mxuv3_memcpy_256(void *dst, const void *src, size_t size) {
    memcpy(dst, src, size);
}
static inline void mxuv3_memset_64(void *dst, uint8_t value, size_t size) {
    memset(dst, value, size);
}

#endif /* __mips__ */

#endif /* _MXUV3_H_ */

