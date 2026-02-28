#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "mars_nn_hw.h"
#include "mxu_ops.h"

static void *xaligned_alloc(size_t align, size_t size) {
    void *p = NULL;
    if (posix_memalign(&p, align, size) != 0)
        return NULL;
    return p;
}

static int check_f32_equal(const float *a, const float *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        /* Exact compare is OK here (simple ops + deterministic inputs). */
        if (a[i] != b[i])
            return 0;
    }
    return 1;
}

static void scalar_add(float *dst, const float *a, const float *b, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = a[i] + b[i];
}

int main(int argc, char **argv) {
#ifndef __mips__
    (void)argc; (void)argv;
    printf("mxuv3_f32_vec_test: built for non-MIPS host, nothing to do.\n");
    return 0;
#else
    size_t n = 16384;
    if (argc > 1) {
        long v = atol(argv[1]);
        if (v > 0) n = (size_t)v;
    }
    printf("MXUv3 float32 vec ops test (n=%zu)\n", n);

    /* Initialize NNA HW and run mxu_init() with NNA-managed memory.
     * This matches the constraint described in docs/mxuv3_instructions.md.
     */
    mars_nn_hw_ctx_t ctx;
    if (mars_nn_hw_init(&ctx) < 0) {
        fprintf(stderr, "mars_nn_hw_init failed (need /dev/mem + /dev/soc-nna, likely root + module).\n");
        return 1;
    }
    mxu_init(ctx.ddr_vaddr);

    float *a = xaligned_alloc(64, n * sizeof(float));
    float *b = xaligned_alloc(64, n * sizeof(float));
    float *out = xaligned_alloc(64, n * sizeof(float));
    float *ref = xaligned_alloc(64, n * sizeof(float));
    if (!a || !b || !out || !ref) {
        fprintf(stderr, "alloc failed\n");
        mars_nn_hw_cleanup(&ctx);
        return 1;
    }

    for (size_t i = 0; i < n; i++) {
        a[i] = (float)(i % 97) * 0.25f + 1.0f;
        b[i] = (float)((i * 7) % 101) * 0.125f + 0.5f;
    }

    scalar_add(ref, a, b, n);
    mxu_add_f32(out, a, b, n);

    int ok = check_f32_equal(out, ref, n);
    printf("mxu_add_f32 correctness: %s\n", ok ? "PASS" : "FAIL");

    free(a); free(b); free(out); free(ref);
    mars_nn_hw_cleanup(&ctx);
    return ok ? 0 : 1;
#endif
}
