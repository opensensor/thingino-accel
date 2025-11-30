/*
 * MXUv3-accelerated element-wise operations for Mars Runtime
 *
 * Uses confirmed working MXU VPR operations:
 * - VPR_ADD(dst, src): VPR[dst] = VPR[src] + VPR[dst]
 * - VPR_SUB(dst, src): VPR[dst] = VPR[src] - VPR[dst]
 * - VPR_MUL(dst, src): VPR[dst] = VPR[src] * VPR[dst]
 *
 * Each VPR register is 512-bit (64 bytes) = 16 floats
 *
 * Copyright (c) 2024 OpenSensor Project
 * SPDX-License-Identifier: MIT
 */

#include <stdint.h>
#include <string.h>
#include <stdio.h>

#ifdef __mips__
#include "mxuv3.h"

/* MXU must be initialized before compute operations work */
static int g_mxu_initialized = 0;

/*
 * Initialize MXU for compute operations.
 * Must be called once with NNA-allocated memory before using VPR compute ops.
 */
void mxu_init(void *nna_mem) {
    if (g_mxu_initialized) return;
    
    int8_t *data = (int8_t *)nna_mem;
    register void *t5_reg __asm__("t5");
    
    __asm__ __volatile__("sync" ::: "memory");
    
    /* MXU control register initialization sequence */
    __asm__ __volatile__(".word 0x4b200180\n .word 0x4b200140\n" ::: "memory");
    __asm__ __volatile__("move $2, %0\n .word 0x70400991\n .word 0x70410951\n" 
                         :: "r"(data) : "$2", "memory");
    __asm__ __volatile__(".word 0x4b200100\n" ::: "memory");
    __asm__ __volatile__("move $2, %0\n .word 0x70400911\n" 
                         :: "r"(data) : "$2", "memory");
    __asm__ __volatile__(".word 0x70f031b0\n .word 0x70f02970\n .word 0x70f02130\n" 
                         ::: "memory");
    __asm__ __volatile__(".word 0x4a684004\n" ::: "memory");
    __asm__ __volatile__("move $2, %0\n .word 0x70401811\n .word 0x70415811\n" 
                         :: "r"(data) : "$2", "memory");
    t5_reg = data;
    __asm__ __volatile__(
        ".word 0x71a80231\n .word 0x71a00031\n .word 0x71a141f1\n"
        ".word 0x71a90071\n .word 0x71a100b1\n .word 0x71a940f1\n"
        :: "r"(t5_reg) : "memory");
    
    g_mxu_initialized = 1;
}

int mxu_is_initialized(void) {
    return g_mxu_initialized;
}

/*
 * Element-wise multiply: out = a * b
 * Processes 16 floats at a time using VPR registers.
 * 
 * For in-place: call with out == a or out == b
 */
void mxu_mul_f32(float *out, const float *a, const float *b, size_t count) {
    size_t i = 0;
    
    /* Process 16 floats at a time using VPR2 and VPR4 */
    for (; i + 16 <= count; i += 16) {
        /* Load a into VPR2, b into VPR4 */
        LA0_VPR(2, a + i);
        LA0_VPR(4, b + i);
        
        /* VPR2 = VPR4 * VPR2 */
        VPR_MUL(2, 4);
        
        /* Store VPR2 to output */
        SA0_VPR(2, out + i);
    }
    __asm__ __volatile__("sync" ::: "memory");
    
    /* Handle remaining elements with scalar code */
    for (; i < count; i++) {
        out[i] = a[i] * b[i];
    }
}

/*
 * Element-wise add: out = a + b
 */
void mxu_add_f32(float *out, const float *a, const float *b, size_t count) {
    size_t i = 0;
    
    for (; i + 16 <= count; i += 16) {
        LA0_VPR(2, a + i);
        LA0_VPR(4, b + i);
        VPR_ADD(2, 4);  /* VPR2 = VPR4 + VPR2 */
        SA0_VPR(2, out + i);
    }
    __asm__ __volatile__("sync" ::: "memory");
    
    for (; i < count; i++) {
        out[i] = a[i] + b[i];
    }
}

/*
 * Element-wise subtract: out = a - b
 * Note: VPR_SUB computes VPR[dst] = VPR[src] - VPR[dst]
 * So to get a - b, we load b into dst, a into src
 */
void mxu_sub_f32(float *out, const float *a, const float *b, size_t count) {
    size_t i = 0;
    
    for (; i + 16 <= count; i += 16) {
        LA0_VPR(2, b + i);  /* Load b into VPR2 (dst) */
        LA0_VPR(4, a + i);  /* Load a into VPR4 (src) */
        VPR_SUB(2, 4);      /* VPR2 = VPR4 - VPR2 = a - b */
        SA0_VPR(2, out + i);
    }
    __asm__ __volatile__("sync" ::: "memory");
    
    for (; i < count; i++) {
        out[i] = a[i] - b[i];
    }
}

/*
 * ReLU: out = max(0, x)
 * Uses VPR_ZERO to create zero vector, then compares
 * Note: This is a placeholder - we may need a different approach
 * since we don't have comparison instructions decoded yet.
 */
void mxu_relu_f32(float *out, const float *in, size_t count) {
    /* Scalar fallback for now - no comparison ops decoded */
    for (size_t i = 0; i < count; i++) {
        out[i] = in[i] > 0.0f ? in[i] : 0.0f;
    }
}

#else
/* Non-MIPS fallback */
void mxu_init(void *nna_mem) { (void)nna_mem; }
int mxu_is_initialized(void) { return 0; }

void mxu_mul_f32(float *out, const float *a, const float *b, size_t count) {
    for (size_t i = 0; i < count; i++) out[i] = a[i] * b[i];
}

void mxu_add_f32(float *out, const float *a, const float *b, size_t count) {
    for (size_t i = 0; i < count; i++) out[i] = a[i] + b[i];
}

void mxu_sub_f32(float *out, const float *a, const float *b, size_t count) {
    for (size_t i = 0; i < count; i++) out[i] = a[i] - b[i];
}

void mxu_relu_f32(float *out, const float *in, size_t count) {
    for (size_t i = 0; i < count; i++) out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}
#endif /* __mips__ */

