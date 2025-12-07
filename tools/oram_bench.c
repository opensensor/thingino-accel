/*
 * ORAM vs DDR Memory Benchmark
 * 
 * Measures memory access latency and bandwidth for:
 * 1. DDR (via nna_malloc from nmem pool)
 * 2. ORAM (384KB on-chip SRAM)
 * 3. NNDMA transfers (DDR <-> ORAM)
 * 4. MXU compute from each memory type
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/time.h>

#ifdef __mips__
#include "mxuv3.h"
#endif

/* NNA hardware addresses */
#define NNA_ORAM_BASE       0x12600000
#define NNA_ORAM_MAX_SIZE   0xe0000         /* 896KB max */
#define L2CACHE_SIZE_REG    0x10010060

/* IOCTL definitions */
#define SOC_NNA_MAGIC       'c'
#define IOCTL_NNA_MALLOC    _IOWR(SOC_NNA_MAGIC, 0, int)
#define IOCTL_NNA_FREE      _IOWR(SOC_NNA_MAGIC, 1, int)
#define IOCTL_NNA_FLUSHCACHE _IOWR(SOC_NNA_MAGIC, 2, int)

struct soc_nna_buf {
    void *vaddr;
    void *paddr;
    int size;
};

static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static uint32_t get_l2cache_size(int memfd) {
    void *gpio = mmap(NULL, 0x1000, PROT_READ, MAP_SHARED, memfd, 0x10010000);
    if (gpio == MAP_FAILED) return 0x40000;
    uint32_t val = *(volatile uint32_t *)((char *)gpio + 0x60);
    munmap(gpio, 0x1000);
    uint32_t bits = (val >> 10) & 0x7;
    printf("L2 cache config bits: %u\n", bits);
    switch (bits) {
        case 1: return 0x20000;   /* 128KB */
        case 2: return 0x40000;   /* 256KB */
        case 3: return 0x80000;   /* 512KB */
        default: return 0x40000;  /* Default 256KB - matches working mars_nn_hw.c */
    }
}

/* Benchmark: sequential read bandwidth */
static double bench_seq_read(volatile float *ptr, size_t count, int iterations) {
    double start = get_time_ms();
    float sum = 0;
    for (int iter = 0; iter < iterations; iter++) {
        for (size_t i = 0; i < count; i++) {
            sum += ptr[i];
        }
    }
    double elapsed = get_time_ms() - start;
    /* Prevent optimization */
    if (sum == -999999.0f) printf("x");
    return elapsed;
}

/* Benchmark: sequential write bandwidth */
static double bench_seq_write(volatile float *ptr, size_t count, int iterations) {
    double start = get_time_ms();
    for (int iter = 0; iter < iterations; iter++) {
        for (size_t i = 0; i < count; i++) {
            ptr[i] = (float)i;
        }
    }
    double elapsed = get_time_ms() - start;
    return elapsed;
}

/* Benchmark: MXU dot product */
static double bench_mxu_dot(float *a, float *b, size_t count, int iterations) {
#ifdef __mips__
    float scratch[16] __attribute__((aligned(64)));
    double start = get_time_ms();
    
    for (int iter = 0; iter < iterations; iter++) {
        float sum = 0;
        for (size_t i = 0; i + 16 <= count; i += 16) {
            LA0_VPR(2, a + i);
            LA0_VPR(4, b + i);
            VPR_MUL(2, 4);
            SA0_VPR(2, scratch);
            __asm__ __volatile__("sync" ::: "memory");
            for (int j = 0; j < 16; j++) sum += scratch[j];
        }
        /* Prevent optimization */
        if (sum == -999999.0f) printf("x");
    }
    
    return get_time_ms() - start;
#else
    (void)a; (void)b; (void)count; (void)iterations;
    return 0;
#endif
}

/* Benchmark: memcpy */
static double bench_memcpy(void *dst, void *src, size_t size, int iterations) {
    double start = get_time_ms();
    for (int iter = 0; iter < iterations; iter++) {
        memcpy(dst, src, size);
    }
    return get_time_ms() - start;
}

int main(int argc, char **argv) {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  ORAM vs DDR Memory Benchmark                            ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    int memfd = open("/dev/mem", O_RDWR | O_SYNC);
    if (memfd < 0) { perror("open /dev/mem"); return 1; }

    int nna_fd = open("/dev/soc-nna", O_RDWR);
    if (nna_fd < 0) { perror("open /dev/soc-nna"); return 1; }

    /* Map ORAM */
    uint32_t l2_size = get_l2cache_size(memfd);
    uint32_t oram_paddr = NNA_ORAM_BASE + l2_size;
    uint32_t oram_size = NNA_ORAM_MAX_SIZE - l2_size;
    
    void *oram = mmap(NULL, oram_size, PROT_READ | PROT_WRITE,
                      MAP_SHARED, memfd, oram_paddr);
    if (oram == MAP_FAILED) { perror("mmap oram"); return 1; }
    
    printf("ORAM: %u KB at paddr 0x%08x, vaddr %p\n", oram_size/1024, oram_paddr, oram);

    /* Allocate DDR via NNA - returns kernel vaddr, need to mmap paddr ourselves */
    size_t test_size = 256 * 1024;  /* 256KB test buffer */
    struct soc_nna_buf ddr_buf = { .size = test_size };
    if (ioctl(nna_fd, IOCTL_NNA_MALLOC, &ddr_buf) < 0) {
        perror("ioctl malloc"); return 1;
    }

    /* The driver's vaddr is a kernel address - mmap the physical address for userspace */
    uint32_t ddr_paddr = (uint32_t)(uintptr_t)ddr_buf.paddr;
    void *ddr_user = mmap(NULL, test_size, PROT_READ | PROT_WRITE,
                          MAP_SHARED, memfd, ddr_paddr);
    if (ddr_user == MAP_FAILED) {
        perror("mmap ddr"); return 1;
    }
    printf("DDR:  %zu KB at paddr 0x%08x, user vaddr %p\n\n",
           test_size/1024, ddr_paddr, ddr_user);

    float *ddr = (float *)ddr_user;
    volatile float *oram_f = (volatile float *)oram;
    size_t count = test_size / sizeof(float);
    int iterations = 100;

    /* Test DDR access first */
    printf("Testing DDR access... ");
    fflush(stdout);
    ddr[0] = 1.0f;
    ddr[1] = 2.0f;
    printf("OK (wrote %.1f, %.1f)\n", ddr[0], ddr[1]);

    /* Test ORAM with single word access */
    printf("Testing ORAM single word write... ");
    fflush(stdout);
    oram_f[0] = 1.0f;
    printf("OK\n");

    printf("Testing ORAM single word read... ");
    fflush(stdout);
    float test_val = oram_f[0];
    printf("OK (read %.1f)\n", test_val);

    /* Initialize data */
    printf("Initializing DDR data... ");
    fflush(stdout);
    for (size_t i = 0; i < count; i++) {
        ddr[i] = (float)i * 0.001f;
    }
    printf("OK\n");

    printf("Initializing ORAM data... ");
    fflush(stdout);
    for (size_t i = 0; i < count; i++) {
        oram_f[i] = (float)i * 0.001f;
    }
    printf("OK\n");

    printf("Running benchmarks (%d iterations, %zu KB)...\n\n", iterations, test_size/1024);

    /* Sequential Read */
    printf("═══ Sequential Read ═══\n");
    double ddr_read = bench_seq_read(ddr, count, iterations);
    double oram_read = bench_seq_read(oram_f, count, iterations);
    double ddr_bw = (test_size * iterations) / (ddr_read * 1000.0);  /* MB/s */
    double oram_bw = (test_size * iterations) / (oram_read * 1000.0);
    printf("  DDR:  %.2f ms (%.1f MB/s)\n", ddr_read, ddr_bw);
    printf("  ORAM: %.2f ms (%.1f MB/s)\n", oram_read, oram_bw);
    printf("  Speedup: %.2fx\n\n", ddr_read / oram_read);

    /* Sequential Write */
    printf("═══ Sequential Write ═══\n");
    double ddr_write = bench_seq_write(ddr, count, iterations);
    double oram_write = bench_seq_write(oram_f, count, iterations);
    ddr_bw = (test_size * iterations) / (ddr_write * 1000.0);
    oram_bw = (test_size * iterations) / (oram_write * 1000.0);
    printf("  DDR:  %.2f ms (%.1f MB/s)\n", ddr_write, ddr_bw);
    printf("  ORAM: %.2f ms (%.1f MB/s)\n", oram_write, oram_bw);
    printf("  Speedup: %.2fx\n\n", ddr_write / oram_write);

    /* Memcpy DDR->ORAM and ORAM->DDR */
    printf("═══ Memcpy Transfer ═══\n");
    double ddr_to_oram = bench_memcpy(oram, ddr, test_size, iterations);
    double oram_to_ddr = bench_memcpy(ddr, oram, test_size, iterations);
    printf("  DDR->ORAM: %.2f ms (%.1f MB/s)\n", ddr_to_oram,
           (test_size * iterations) / (ddr_to_oram * 1000.0));
    printf("  ORAM->DDR: %.2f ms (%.1f MB/s)\n", oram_to_ddr,
           (test_size * iterations) / (oram_to_ddr * 1000.0));

#ifdef __mips__
    /* MXU Dot Product */
    printf("\n═══ MXU Dot Product ═══\n");
    /* Need second buffer for dot product */
    float *ddr2 = ddr + count/2;
    float *oram2 = oram_f + count/2;
    size_t dot_count = count / 2;

    double ddr_dot = bench_mxu_dot(ddr, ddr2, dot_count, iterations);
    double oram_dot = bench_mxu_dot(oram_f, oram2, dot_count, iterations);
    printf("  DDR:  %.2f ms\n", ddr_dot);
    printf("  ORAM: %.2f ms\n", oram_dot);
    printf("  Speedup: %.2fx\n", ddr_dot / oram_dot);
#endif

    printf("\n═══ Summary ═══\n");
    printf("ORAM provides %.1fx-%.1fx speedup over DDR for memory operations.\n",
           ddr_write / oram_write, ddr_read / oram_read);
    printf("For 384KB ORAM, tensors up to ~96K floats can be staged.\n");

    /* Cleanup */
    ioctl(nna_fd, IOCTL_NNA_FREE, &ddr_buf);
    munmap(oram, oram_size);
    close(nna_fd);
    close(memfd);

    return 0;
}

