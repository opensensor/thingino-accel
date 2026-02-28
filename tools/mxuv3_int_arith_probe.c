/*
 * MXUv3 integer arithmetic instruction probe for Ingenic XBurst2 (A1/T41)
 *
 * Probes VPR instruction function codes by writing them as machine code
 * into executable memory (mmap), then calling and catching SIGILL.
 *
 * For each valid instruction, dumps VPR2 output so we can infer semantics.
 *
 * Scans rs=16 (where MAX/MIN live) and rs=20 (where ADD/SUB live).
 *
 * Usage: mxuv3_int_arith_probe [rs_value]
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

/*
 * Build a tiny function in executable memory:
 *   .word <cop2_encoding>   // the COP2 instruction under test
 *   sync
 *   jr $ra
 *   nop                     // delay slot
 */
static void (*build_probe(void *page, uint32_t cop2_enc))(void)
{
    uint32_t *code = (uint32_t *)page;
    code[0] = cop2_enc;      /* COP2 instruction */
    code[1] = 0x0000000F;    /* sync             */
    code[2] = 0x03E00008;    /* jr $ra           */
    code[3] = 0x00000000;    /* nop (delay slot) */

    /* Flush dcache + invalidate icache for these 16 bytes */
    __builtin___clear_cache((char *)code, (char *)(code + 4));

    return (void (*)(void))code;
}

static void probe_rs(void *page, uint8_t rs)
{
    int32_t a[16] __attribute__((aligned(64)));
    int32_t b[16] __attribute__((aligned(64)));
    int32_t out[16] __attribute__((aligned(64)));

    for (int i = 0; i < 16; i++) {
        a[i] = 100 + i;
        b[i] = 50 + i;
    }

    printf("Probing rs=%d  (VPR0=int32[100..115], VPR1=int32[50..65])\n", rs);
    printf("  Instruction: VPR2 = op(VPR0, VPR1)\n\n");

    struct sigaction sa, old;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = sigill_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;

    for (int fn = 0; fn < 64; fn++) {
        uint32_t enc = 0x48000000 | ((uint32_t)rs << 21) |
                       (0u << 16) | (1u << 11) | (2u << 6) | (uint32_t)fn;
        void (*probe_fn)(void) = build_probe(page, enc);

        LA0_VPR(0, a);
        LA0_VPR(1, b);
        memset(out, 0xAA, sizeof(out));

        sigaction(SIGILL, &sa, &old);
        sigill_caught = 0;

        if (sigsetjmp(sigill_jmp, 1) == 0) {
            probe_fn();
        }
        sigaction(SIGILL, &old, NULL);

        if (!sigill_caught) {
            SA0_VPR(2, out);
            printf("  fn=%2d (0x%02x): OK  [0..3]=%d %d %d %d\n",
                   fn, fn, out[0], out[1], out[2], out[3]);
        }
    }
}

int main(int argc, char **argv)
{
    int rs = 16;
    if (argc > 1) rs = atoi(argv[1]);

    printf("MXUv3 integer arithmetic probe (A1/T41)\n");
    printf("=========================================\n\n");

    void *page = mmap(NULL, 4096, PROT_READ | PROT_WRITE | PROT_EXEC,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page == MAP_FAILED) { perror("mmap"); return 1; }

    /* Prime CU2 */
    __asm__ __volatile__(".word 0x4a80000b" ::: "memory");

    probe_rs(page, (uint8_t)rs);

    printf("\n--- Done ---\n");
    munmap(page, 4096);
    return 0;
}
#else
int main(void) { printf("Non-MIPS host, nothing to do.\n"); return 0; }
#endif

