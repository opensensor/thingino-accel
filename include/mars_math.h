/*
 * Mars basic math helpers (float32 vector/matrix)
 *
 * These provide a clean, hardware-agnostic semantic layer for
 * vector/matrix operations that we can later map to NNA commands.
 */

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* dst[i] = a[i] + b[i] for i in [0, n) */
void mars_vec_add_f32(float *dst,
                      const float *a,
                      const float *b,
                      size_t n);

/* Dot product: sum_{i=0..n-1} a[i] * b[i] */
float mars_vec_dot_f32(const float *a,
                       const float *b,
                       size_t n);

/* Row-major matrix multiply: C[M x N] = A[M x K] * B[K x N]. */
void mars_matmul_f32(float *C,
                     const float *A,
                     const float *B,
                     size_t M, size_t K, size_t N);

#ifdef __cplusplus
}
#endif

