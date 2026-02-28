/*
 * MXUv3 shift/logic instruction probe for Ingenic XBurst2 (A1/T41)
 *
 * Probes untested rs values (17, 18, 21, 22) looking for VPR shift
 * and XOR instructions.  These would unlock:
 *   - QPEL 6-tap filter vectorization (<<2, <<4, >>5)
 *   - H.264 IDCT vectorization (>>1)
 *   - VPR byte averaging without XOR
 *
 * Uses power-of-2 input values so shifts produce distinctive output.
 *
 * Usage: mxuv3_shift_probe [rs_value]
 *        Default: probes rs=17,18,21,22
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <setjmp.h>
#include <sys/mman.h>

#include "mxuv3.h"

#ifdef __mips__

static sigjmp_buf sigill_jmp;
static volatile sig_atomic_t sigill_caught;

static void sigill_handler(int sig)
{
    (void)sig;
    sigill_caught = 1;
    siglongjmp(sigill_jmp, 1);
}

static void (*build_probe(void *page, uint32_t cop2_enc))(void)
{
    uint32_t *code = (uint32_t *)page;
    code[0] = cop2_enc;
    code[1] = 0x0000000F;    /* sync */
    code[2] = 0x03E00008;    /* jr $ra */
    code[3] = 0x00000000;    /* nop */
    __builtin___clear_cache((char *)code, (char *)(code + 4));
    return (void (*)(void))code;
}

/*
 * Probe one rs value with shift-friendly inputs.
 *
 * VPR0 = int32 powers of 2:  [256, 512, 1024, 2048, ...]
 * VPR1 = int32 small values:  [1, 2, 3, 4, 5, ...]
 *
 * Encoding: COP2 | rs | rt=VPR0 | rd=VPR1 | sa=VPR2(dest) | fn
 *
 * Expected shift signatures:
 *   SHR variable: all outputs = 128 (256>>1, 512>>2, 1024>>3, ...)
 *   SHL variable: [512, 2048, 8192, 32768, ...]
 *   SHR by 1 (imm): [128, 256, 512, 1024, ...]
 *   XOR: [257, 514, 1027, 2052, ...]  (256^1, 512^2, 1024^3, 2048^4)
 */
static void probe_rs_shift(void *page, uint8_t rs)
{
    int32_t a[16] __attribute__((aligned(64)));
    int32_t b[16] __attribute__((aligned(64)));
    int32_t out[16] __attribute__((aligned(64)));
    int valid_count = 0;

    for (int i = 0; i < 16; i++) {
        a[i] = 256 << i;      /* 256, 512, 1024, 2048, ... */
        b[i] = i + 1;         /* 1, 2, 3, 4, ... */
    }

    printf("Probing rs=%d  (VPR0=pow2[256..], VPR1=int[1..16])\n", rs);
    printf("  Encoding: COP2 rs=%d rt=0(VPR0) rd=1(VPR1) sa=2(VPR2) fn=?\n", rs);
    printf("  Looking for: SHR→[128,128,...] SHL→[512,2048,...] XOR→[257,514,...]\n\n");

    struct sigaction sa_act, old;
    memset(&sa_act, 0, sizeof(sa_act));
    sa_act.sa_handler = sigill_handler;
    sigemptyset(&sa_act.sa_mask);

    for (int fn = 0; fn < 64; fn++) {
        /* Standard 3-operand: rt=0(src), rd=1(src2), sa=2(dest), fn */
        uint32_t enc = 0x48000000 | ((uint32_t)rs << 21) |
                       (0u << 16) | (1u << 11) | (2u << 6) | (uint32_t)fn;
        void (*probe_fn)(void) = build_probe(page, enc);

        LA0_VPR(0, a);
        LA0_VPR(1, b);
        memset(out, 0xAA, sizeof(out));

        sigaction(SIGILL, &sa_act, &old);
        sigill_caught = 0;

        if (sigsetjmp(sigill_jmp, 1) == 0)
            probe_fn();
        sigaction(SIGILL, &old, NULL);

        if (!sigill_caught) {
            SA0_VPR(2, out);
            printf("  fn=%2d (0x%02x): OK  [0..3]=%d %d %d %d",
                   fn, fn, out[0], out[1], out[2], out[3]);

            /* Flag likely shift/xor patterns */
            if (out[0] == 128 && out[1] == 128 && out[2] == 128)
                printf("  *** LIKELY SHR VARIABLE ***");
            else if (out[0] == 512 && out[1] == 2048)
                printf("  *** LIKELY SHL VARIABLE ***");
            else if (out[0] == 128 && out[1] == 256 && out[2] == 512)
                printf("  *** LIKELY SHR BY 1 ***");
            else if (out[0] == 257 && out[1] == 514)
                printf("  *** LIKELY XOR ***");
            else if (out[0] == (256|1) || out[0] == (256^1))
                printf("  *** CHECK: XOR or OR? ***");
            printf("\n");
            valid_count++;
        }
    }
    if (valid_count == 0)
        printf("  (no valid instructions found)\n");
    printf("\n");
}

int main(int argc, char **argv)
{
    printf("MXUv3 shift/XOR probe (A1/T41)\n");
    printf("================================\n\n");

    void *page = mmap(NULL, 4096, PROT_READ | PROT_WRITE | PROT_EXEC,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page == MAP_FAILED) { perror("mmap"); return 1; }

    /* Prime CU2 */
    __asm__ __volatile__(".word 0x4a80000b" ::: "memory");

    if (argc > 1) {
        probe_rs_shift(page, (uint8_t)atoi(argv[1]));
    } else {
        /* Probe all untested rs values adjacent to known ones */
        int rs_values[] = {17, 18, 21, 22, 23, 24};
        for (int i = 0; i < 6; i++) {
            printf("=== rs=%d ===\n", rs_values[i]);
            probe_rs_shift(page, (uint8_t)rs_values[i]);
        }
    }

    printf("--- Done ---\n");
    munmap(page, 4096);
    return 0;
}
#else
int main(void) { printf("Non-MIPS host.\n"); return 0; }
#endif

