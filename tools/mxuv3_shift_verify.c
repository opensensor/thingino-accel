/*
 * MXUv3 shift instruction verification for Ingenic XBurst2 (A1/T41)
 *
 * Targeted tests to confirm:
 *  1. rs=17 fn=33 is variable SHL (VPR[sa] = VPR[rt] << VPR[rd])
 *  2. rs=17 fn=49 is variable SHR
 *  3. fn=33 vs fn=34: element width (byte vs half vs word)
 *  4. rs=21: whether rd encodes immediate shift amount
 *
 * Usage: mxuv3_shift_verify
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <signal.h>
#include <setjmp.h>
#include <sys/mman.h>
#include "mxuv3.h"

#ifdef __mips__
static sigjmp_buf jmp;
static volatile sig_atomic_t caught;
static void handler(int s) { (void)s; caught = 1; siglongjmp(jmp, 1); }

static void (*mkprobe(void *p, uint32_t enc))(void)
{
    uint32_t *c = (uint32_t *)p;
    c[0] = enc; c[1] = 0x0000000F; c[2] = 0x03E00008; c[3] = 0;
    __builtin___clear_cache((char *)c, (char *)(c+4));
    return (void(*)(void))c;
}

static int run(void *page, uint32_t enc,
               const int32_t a[16], const int32_t b[16], int32_t out[16])
{
    struct sigaction sa, old;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handler;
    void (*fn)(void) = mkprobe(page, enc);

    LA0_VPR(0, a);
    LA0_VPR(1, b);
    memset(out, 0xDD, 64);

    sigaction(SIGILL, &sa, &old);
    caught = 0;
    if (sigsetjmp(jmp, 1) == 0) fn();
    sigaction(SIGILL, &old, NULL);

    if (!caught) SA0_VPR(2, out);
    return !caught;
}

#define COP2(rs, rt, rd, sa, fn) \
    (0x48000000|((rs)<<21)|((rt)<<16)|((rd)<<11)|((sa)<<6)|(fn))

static void print4(const char *label, const int32_t v[16])
{
    printf("  %-28s [0..3]=%d %d %d %d\n", label, v[0], v[1], v[2], v[3]);
}

int main(void)
{
    int32_t a[16] __attribute__((aligned(64)));
    int32_t b[16] __attribute__((aligned(64)));
    int32_t out[16] __attribute__((aligned(64)));

    void *page = mmap(NULL, 4096, PROT_READ|PROT_WRITE|PROT_EXEC,
                       MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    __asm__ __volatile__(".word 0x4a80000b":::"memory"); /* CU2 prime */

    printf("MXUv3 shift verification (A1/T41)\n");
    printf("===================================\n\n");

    /* ---- Test 1: rs=17 variable SHL (fn=33) ---- */
    printf("=== Test 1: rs=17 fn=33 variable SHL ===\n");
    for (int i = 0; i < 16; i++) { a[i] = 1; b[i] = i; }
    print4("VPR0 (all 1s):", a);
    print4("VPR1 (shift amts 0..15):", b);
    /* enc: rt=0(VPR0), rd=1(VPR1), sa=2(VPR2/dest), fn=33 */
    if (run(page, COP2(17, 0, 1, 2, 33), a, b, out))
        print4("Result (expect 1,2,4,8):", out);

    /* ---- Test 2: rs=17 variable SHR (fn=49) ---- */
    printf("\n=== Test 2: rs=17 fn=49 variable SHR ===\n");
    for (int i = 0; i < 16; i++) { a[i] = 0x80000000U >> i; b[i] = i; }
    print4("VPR0 (0x80000000>>i):", a);
    print4("VPR1 (shift amts 0..15):", b);
    if (run(page, COP2(17, 0, 1, 2, 49), a, b, out))
        print4("Result (all 0x80000000?):", out);

    /* ---- Test 3: fn=33 vs fn=34 element width discrimination ---- */
    printf("\n=== Test 3: fn=33 vs fn=34 — element width ===\n");
    /* Use value that overflows at byte boundary: 0x80 << 1 = 0x100 (byte overflow) */
    for (int i = 0; i < 16; i++) { a[i] = 0x00800080; b[i] = 1; }
    print4("VPR0 (0x00800080 = two 0x80 in int16):", a);
    print4("VPR1 (shift by 1):", b);
    if (run(page, COP2(17, 0, 1, 2, 33), a, b, out))
        printf("  fn=33 result: 0x%08X (word SHL: 0x01000100)\n", (unsigned)out[0]);
    if (run(page, COP2(17, 0, 1, 2, 34), a, b, out))
        printf("  fn=34 result: 0x%08X (half SHL: 0x01000100, byte: 0x00000000)\n",
               (unsigned)out[0]);

    /* Test int16 overflow: 0xFF00 << 1 */
    for (int i = 0; i < 16; i++) { a[i] = 0xFF00FF00; b[i] = 1; }
    if (run(page, COP2(17, 0, 1, 2, 33), a, b, out))
        printf("  fn=33 0xFF00FF00<<1: 0x%08X (word: 0xFE01FE00)\n", (unsigned)out[0]);
    if (run(page, COP2(17, 0, 1, 2, 34), a, b, out))
        printf("  fn=34 0xFF00FF00<<1: 0x%08X (half: 0xFE00FE00)\n", (unsigned)out[0]);

    /* ---- Test 4: rs=21 immediate shift — vary rd field ---- */
    printf("\n=== Test 4: rs=21 — is rd the immediate shift amount? ===\n");
    for (int i = 0; i < 16; i++) { a[i] = 1024; b[i] = 99; }
    print4("VPR0 (all 1024):", a);
    for (int rd = 1; rd <= 5; rd++) {
        /* enc: rt=0(VPR0), rd=rd(immediate?), sa=2(dest), fn=49 */
        if (run(page, COP2(21, 0, rd, 2, 49), a, b, out))
            printf("  rd=%d fn=49: %d (expect 1024>>%d = %d)\n",
                   rd, out[0], rd, 1024 >> rd);
    }
    /* Also test SHL at rs=21 */
    for (int rd = 1; rd <= 5; rd++) {
        if (run(page, COP2(21, 0, rd, 2, 33), a, b, out))
            printf("  rd=%d fn=33: %d (expect 1024<<%d = %d)\n",
                   rd, out[0], rd, 1024 << rd);
    }

    /* ---- Test 5: Arithmetic right shift (signed) ---- */
    printf("\n=== Test 5: signed SHR — negative values ===\n");
    for (int i = 0; i < 16; i++) { a[i] = -100; b[i] = 1; }
    print4("VPR0 (all -100):", a);
    if (run(page, COP2(17, 0, 1, 2, 49), a, b, out))
        printf("  fn=49 SHR:  %d (logical: %u, arithmetic: %d)\n",
               out[0], (unsigned)out[0], -100 >> 1);
    if (run(page, COP2(17, 0, 1, 2, 50), a, b, out))
        printf("  fn=50 SHR:  %d (logical: %u, arithmetic: %d)\n",
               out[0], (unsigned)out[0], -100 >> 1);

    printf("\n--- Done ---\n");
    munmap(page, 4096);
    return 0;
}
#else
int main(void) { printf("Non-MIPS host.\n"); return 0; }
#endif

