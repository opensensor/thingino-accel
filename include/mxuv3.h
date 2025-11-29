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

#else /* !__mips__ */

/* Stub implementations for non-MIPS builds */
#define LA0_VPR(vpr, ptr) ((void)0)
#define SA0_VPR(vpr, ptr) ((void)0)

static inline void mxuv3_load_vpr0(void *ptr) { (void)ptr; }
static inline void mxuv3_load_vpr1(void *ptr) { (void)ptr; }
static inline void mxuv3_store_vpr0(void *ptr) { (void)ptr; }
static inline void mxuv3_store_vpr1(void *ptr) { (void)ptr; }

static inline uint32_t mxuv3_read_mir(void) { return 0; }

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

