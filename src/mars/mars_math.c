/*
 * Mars basic math helpers implementation
 *
 * Float32 vector/matrix helpers. On MIPS T41, these can use MXU
 * acceleration where available; elsewhere, they fall back to scalar
 * implementations.
 */

#include <stddef.h>

#include "mxu_ops.h"
#include "mars_math.h"

void mars_vec_add_f32(float *dst,
                      const float *a,
                      const float *b,
                      size_t n)
{
#if defined(__mips__)
    if (mxu_is_initialized()) {
        mxu_add_f32(dst, a, b, n);
        return;
    }
#endif
    for (size_t i = 0; i < n; i++) {
        dst[i] = a[i] + b[i];
    }
}

float mars_vec_dot_f32(const float *a,
                       const float *b,
                       size_t n)
{
    float acc = 0.0f;
    for (size_t i = 0; i < n; i++) {
        acc += a[i] * b[i];
    }
    return acc;
}

void mars_matmul_f32(float *C,
                     const float *A,
                     const float *B,
                     size_t M, size_t K, size_t N)
{
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; k++) {
                acc += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = acc;
        }
    }
}

