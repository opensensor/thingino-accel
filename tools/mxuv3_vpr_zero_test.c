#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "mxuv3.h"

static void *xaligned_alloc(size_t align, size_t size)
{
    void *p = NULL;
    if (posix_memalign(&p, align, size) != 0)
        return NULL;
    return p;
}

static int all_zero_64(const uint8_t *p)
{
    for (int i = 0; i < 64; i++)
        if (p[i] != 0)
            return 0;
    return 1;
}

int main(void)
{
#ifndef __mips__
    printf("mxuv3_vpr_zero_test: built for non-MIPS host, nothing to do.\n");
    return 0;
#else
    uint8_t *in  = xaligned_alloc(64, 64);
    uint8_t *out = xaligned_alloc(64, 64);
    if (!in || !out) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    for (int i = 0; i < 64; i++)
        in[i] = (uint8_t)(i + 1);

    printf("MXUv3 VPR_ZERO test (VPR0)\n");
    printf("- Step 1: LA0(VPR0) <- pattern\n");
    mxuv3_load_vpr0(in);

    printf("- Step 2: VPR_ZERO once; SA0(VPR0) -> out\n");
    mxuv3_zero_vpr0();
    mxuv3_store_vpr0(out);
    printf("  after 1x VPR_ZERO: %s\n", all_zero_64(out) ? "ALL ZERO" : "NOT ZERO");

    printf("- Step 3: VPR_ZERO again; SA0(VPR0) -> out\n");
    mxuv3_zero_vpr0();
    mxuv3_store_vpr0(out);
    printf("  after 2x VPR_ZERO: %s\n", all_zero_64(out) ? "ALL ZERO" : "NOT ZERO");

    int ok = all_zero_64(out);
    free(in);
    free(out);
    return ok ? 0 : 1;
#endif
}
