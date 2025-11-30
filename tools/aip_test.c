/*
 * AIP Test - Verify communication with AI Processor and run convolution
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#include "aip.h"

static void alarm_handler(int sig) {
    (void)sig;
    /* Just return to interrupt the ioctl */
}

/* Physical memory allocation via /dev/mem - simplified test */
#define NMEM_BASE 0x06300000   /* DDR reserved for NNA */

/* Read AIP register */
static uint32_t aip_readl(volatile uint32_t *io, uint32_t offset) {
    return io[offset / 4];
}

/* Write AIP register */
static void aip_writel(volatile uint32_t *io, uint32_t offset, uint32_t val) {
    io[offset / 4] = val;
}

/* AIP-F node parameters structure (48 bytes = 12 x uint32_t) */
typedef struct {
    uint32_t in_addr;       /* [0] Input address */
    uint32_t out_addr;      /* [1] Output address */
    uint32_t in_size;       /* [2] Input size (written as +0x10) */
    uint32_t out_size;      /* [3] Output size (written as +0x10) */
    uint32_t kernel_addr;   /* [4] Kernel weights address */
    uint32_t bias_addr;     /* [5] Bias address */
    uint32_t kernel_size;   /* [6] Kernel size encoding */
    uint32_t stride;        /* [7] Stride */
    uint16_t in_ch;         /* [8] lo: Input channels */
    uint16_t out_ch;        /* [8] hi: Output channels */
    uint16_t pad;           /* [9] lo: Padding */
    uint16_t pool;          /* [9] hi: Pooling */
    uint16_t shift;         /* [10] lo: Shift */
    uint16_t scale;         /* [10] hi: Scale */
} __attribute__((packed)) aip_f_params_t;

int main(int argc, char **argv) {
    int fd_mem, fd_f, fd_p, fd_t, fd_nna;
    volatile uint32_t *aip_io;

    /* Set up alarm handler for ioctl timeout */
    signal(SIGALRM, alarm_handler);

    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  AIP (AI Processor) Test                                 ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    /* Open /dev/soc-nna to enable NNA clocks */
    fd_nna = open("/dev/soc-nna", O_RDWR);
    if (fd_nna < 0) {
        perror("open /dev/soc-nna");
        printf("  WARNING: NNA clocks may not be enabled!\n");
    } else {
        printf("Opened /dev/soc-nna (fd=%d) - NNA clocks enabled\n\n", fd_nna);
    }

    /* Open AIP devices */
    fd_f = open(AIP_DEV_F, O_RDWR);
    fd_p = open(AIP_DEV_P, O_RDWR);
    fd_t = open(AIP_DEV_T, O_RDWR);
    
    printf("AIP device file descriptors:\n");
    printf("  jzaip_f: %d\n", fd_f);
    printf("  jzaip_p: %d\n", fd_p);
    printf("  jzaip_t: %d\n", fd_t);
    
    /* Map AIP I/O registers */
    fd_mem = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd_mem < 0) {
        perror("open /dev/mem");
        return 1;
    }
    
    aip_io = mmap(NULL, AIP_IOSIZE, PROT_READ | PROT_WRITE,
                  MAP_SHARED, fd_mem, AIP_IOBASE);
    if (aip_io == MAP_FAILED) {
        perror("mmap AIP I/O");
        close(fd_mem);
        return 1;
    }
    
    printf("\nAIP I/O mapped at %p (phys 0x%08x)\n\n", (void*)aip_io, AIP_IOBASE);
    
    /* Dump AIP-T registers */
    printf("=== AIP-T (Tensor/Resize) Registers ===\n");
    printf("  [0x000] CTRL:      0x%08x\n", aip_readl(aip_io, AIP_T_CTRL));
    printf("  [0x008] CFG:       0x%08x\n", aip_readl(aip_io, AIP_T_CFG));
    printf("  [0x010] SRC_ADDR:  0x%08x\n", aip_readl(aip_io, AIP_T_SRC_ADDR));
    printf("  [0x014] DST_ADDR:  0x%08x\n", aip_readl(aip_io, AIP_T_DST_ADDR));
    
    /* Dump AIP-F registers */
    printf("\n=== AIP-F (Feature/Conv) Registers ===\n");
    printf("  [0x200] CTRL:      0x%08x\n", aip_readl(aip_io, AIP_F_CTRL));
    printf("  [0x208] CFG:       0x%08x\n", aip_readl(aip_io, AIP_F_CFG));
    printf("  [0x20C] NODECFG:   0x%08x\n", aip_readl(aip_io, AIP_F_NODECFG));
    printf("  [0x210] IN_ADDR:   0x%08x\n", aip_readl(aip_io, AIP_F_IN_ADDR));
    printf("  [0x214] OUT_ADDR:  0x%08x\n", aip_readl(aip_io, AIP_F_OUT_ADDR));
    printf("  [0x218] IN_SIZE:   0x%08x\n", aip_readl(aip_io, AIP_F_IN_SIZE));
    printf("  [0x21C] OUT_SIZE:  0x%08x\n", aip_readl(aip_io, AIP_F_OUT_SIZE));
    printf("  [0x220] KERN_ADDR: 0x%08x\n", aip_readl(aip_io, AIP_F_KERNEL_ADDR));
    printf("  [0x224] BIAS_ADDR: 0x%08x\n", aip_readl(aip_io, AIP_F_BIAS_ADDR));
    printf("  [0x228] KERN_SIZE: 0x%08x\n", aip_readl(aip_io, AIP_F_KERNEL_SIZE));
    printf("  [0x22C] STRIDE:    0x%08x\n", aip_readl(aip_io, AIP_F_STRIDE));
    printf("  [0x230] IN/OUT_CH: 0x%08x\n", aip_readl(aip_io, AIP_F_IN_CH_OUT_CH));
    printf("  [0x234] PAD_POOL:  0x%08x\n", aip_readl(aip_io, AIP_F_PAD_POOL));
    printf("  [0x238] SCALE_SH:  0x%08x\n", aip_readl(aip_io, AIP_F_SCALE_SHIFT));
    
    /* Dump AIP-P registers */
    printf("\n=== AIP-P (Perspective) Registers ===\n");
    printf("  [0x300] CTRL:      0x%08x\n", aip_readl(aip_io, AIP_P_CTRL));
    printf("  [0x308] CFG:       0x%08x\n", aip_readl(aip_io, AIP_P_CFG));
    printf("  [0x310] SRC_ADDR:  0x%08x\n", aip_readl(aip_io, AIP_P_SRC_ADDR));
    printf("  [0x314] DST_ADDR:  0x%08x\n", aip_readl(aip_io, AIP_P_DST_ADDR));
    
    /* Try allocating a chain buffer via ioctl */
    if (fd_f >= 0) {
        printf("\n=== Testing AIP-F chain buffer allocation ===\n");
        
        struct {
            void *vaddr;
            void *paddr;
            int size;
        } buf = { .vaddr = NULL, .paddr = NULL, .size = 0x36000 };
        
        int ret = ioctl(fd_f, IOCTL_AIP_MALLOC, &buf);
        if (ret >= 0) {
            printf("  Chain buffer allocated:\n");
            printf("    vaddr: %p\n", buf.vaddr);
            printf("    paddr: %p\n", buf.paddr);
            printf("    size:  0x%x\n", buf.size);
            
            /* Map and verify we can write to it */
            void *chainbuf = mmap(NULL, buf.size, PROT_READ | PROT_WRITE,
                                  MAP_SHARED, fd_f, (size_t)buf.paddr);
            if (chainbuf != MAP_FAILED) {
                printf("    Mapped to: %p\n", chainbuf);
                /* Write test pattern */
                memset(chainbuf, 0xAB, 64);
                printf("    Write test: OK\n");
                munmap(chainbuf, buf.size);
            } else {
                perror("    mmap chainbuf");
            }
            
            /* Free buffer */
            ioctl(fd_f, IOCTL_AIP_FREE, 0);
        } else {
            perror("  ioctl MALLOC");
        }
    }
    
    /* === Examine ORAM for AIP usage === */
    printf("\n=== Testing ORAM access ===\n");

    /* ORAM is at 0x12620000 (after 128KB L2 cache) with ~384KB available */
    #define ORAM_BASE   0x12620000
    #define ORAM_SIZE   0x60000     /* 384 KB */

    void *oram = mmap(NULL, ORAM_SIZE, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd_mem, ORAM_BASE);
    if (oram == MAP_FAILED) {
        perror("mmap ORAM");
        goto cleanup;
    }
    printf("  ORAM mapped at %p (phys 0x%08x)\n", oram, ORAM_BASE);

    /* Layout in ORAM:
     * 0x00000 - 0x00FFF: Input  (4KB)
     * 0x01000 - 0x01FFF: Output (4KB)
     * 0x02000 - 0x02FFF: Kernel (4KB)
     * 0x03000 - 0x03FFF: Bias   (4KB)
     */
    uint32_t in_paddr = ORAM_BASE + 0x0000;
    uint32_t out_paddr = ORAM_BASE + 0x1000;
    uint32_t kern_paddr = ORAM_BASE + 0x2000;
    uint32_t bias_paddr = ORAM_BASE + 0x3000;

    int8_t *in_buf = (int8_t*)((char*)oram + 0x0000);
    int8_t *out_buf = (int8_t*)((char*)oram + 0x1000);
    int8_t *kern_buf = (int8_t*)((char*)oram + 0x2000);
    int32_t *bias_buf = (int32_t*)((char*)oram + 0x3000);

    /* Initialize test data */
    /* Input: 8x8x1 filled with value 1 */
    memset(in_buf, 1, 64);
    /* Kernel: 1x1x1x1 with value 2 */
    memset(kern_buf, 0, 64);
    kern_buf[0] = 2;
    /* Bias: 0 */
    memset(bias_buf, 0, 64);

    /* Flush CPU cache to ensure DMA sees the data */
    __sync_synchronize();

    printf("  Input:  %d %d %d %d ...\n", in_buf[0], in_buf[1], in_buf[2], in_buf[3]);
    printf("  Kernel: %d\n", kern_buf[0]);

    /* Reset AIP-F */
    printf("  Resetting AIP-F...\n");
    aip_writel(aip_io, AIP_F_CTRL, 0x02);  /* Set reset bit */
    usleep(100);
    while (aip_readl(aip_io, AIP_F_CTRL) & 0x02) {
        usleep(10);  /* Wait for reset to complete */
    }
    printf("  AIP-F reset complete, CTRL=0x%08x\n", aip_readl(aip_io, AIP_F_CTRL));

    /* Configure convolution: 8x8x1 -> 8x8x1, kernel 1x1, stride 1, no padding */
    printf("  Configuring convolution...\n");
    aip_writel(aip_io, AIP_F_NODECFG, 0xFFFFFFFF);  /* Single node */
    aip_writel(aip_io, AIP_F_IN_ADDR, in_paddr);
    aip_writel(aip_io, AIP_F_OUT_ADDR, out_paddr);

    /* Size encoding: the +0x10 offset might be causing issues */
    /* Try raw sizes first */
    uint32_t in_size = (8 << 16) | 8;   /* 8x8 (h<<16|w) */
    uint32_t out_size = (8 << 16) | 8;
    aip_writel(aip_io, AIP_F_IN_SIZE, in_size);  /* Try without +0x10 */
    aip_writel(aip_io, AIP_F_OUT_SIZE, out_size);

    aip_writel(aip_io, AIP_F_KERNEL_ADDR, kern_paddr);
    aip_writel(aip_io, AIP_F_BIAS_ADDR, bias_paddr);

    /* Kernel size: 1x1 */
    uint32_t kern_size = (1 << 16) | 1;
    aip_writel(aip_io, AIP_F_KERNEL_SIZE, kern_size);

    /* Stride: 1x1 */
    uint32_t stride = (1 << 16) | 1;
    aip_writel(aip_io, AIP_F_STRIDE, stride);

    /* Channels: in=1, out=1 */
    uint32_t channels = (1 << 16) | 1;  /* out_ch << 16 | in_ch */
    aip_writel(aip_io, AIP_F_IN_CH_OUT_CH, channels);

    /* Padding/pooling: none */
    aip_writel(aip_io, AIP_F_PAD_POOL, 0);

    /* Scale/shift: identity (scale=1, shift=0) */
    aip_writel(aip_io, AIP_F_SCALE_SHIFT, (0 << 16) | 1);

    /* CFG: 0x14 = single node mode
     * Add 0x03 to indicate src+dst are in ORAM (bits 0,1)
     * CFG = 0x17 for ORAM mode
     */
    aip_writel(aip_io, AIP_F_CFG, 0x17);

    printf("  Registers configured:\n");
    printf("    IN_ADDR:   0x%08x\n", aip_readl(aip_io, AIP_F_IN_ADDR));
    printf("    OUT_ADDR:  0x%08x\n", aip_readl(aip_io, AIP_F_OUT_ADDR));
    printf("    IN_SIZE:   0x%08x\n", aip_readl(aip_io, AIP_F_IN_SIZE));
    printf("    KERN_ADDR: 0x%08x\n", aip_readl(aip_io, AIP_F_KERNEL_ADDR));
    printf("    CFG:       0x%08x\n", aip_readl(aip_io, AIP_F_CFG));

    /* Clear output buffer */
    memset(out_buf, 0xAA, 64);

    printf("  Registers configured, NOT starting yet (safety check)\n");
    printf("  CTRL: 0x%08x\n", aip_readl(aip_io, AIP_F_CTRL));

    /* Read back to verify ORAM is accessible */
    printf("  Verifying ORAM read/write...\n");
    in_buf[0] = 0x42;
    __sync_synchronize();
    printf("  ORAM[0] write 0x42, read back: 0x%02x\n", (unsigned char)in_buf[0]);

    if (in_buf[0] == 0x42) {
        printf("  ORAM access OK!\n");

        /* Check CGU/CPM for AIP clock status */
        /* CPM is at 0x10000000, we'll check common clock gate registers */
        volatile uint32_t *cpm = mmap(NULL, 0x1000, PROT_READ | PROT_WRITE,
                                      MAP_SHARED, fd_mem, 0x10000000);
        if (cpm != MAP_FAILED) {
            printf("\n  Checking CPM clock registers:\n");
            printf("    CLKGR0 (0x20): 0x%08x\n", cpm[0x20/4]);
            printf("    CLKGR1 (0x28): 0x%08x\n", cpm[0x28/4]);
            /* Also check the ORAM clock bit mentioned in soc-nna */
            volatile uint32_t *intc = mmap(NULL, 0x100, PROT_READ | PROT_WRITE,
                                          MAP_SHARED, fd_mem, 0x12200000);
            if (intc != MAP_FAILED) {
                printf("    ORAM clk (0x12200060): 0x%08x\n", intc[0x60/4]);
                munmap((void*)intc, 0x100);
            }
            munmap((void*)cpm, 0x1000);
        }

        /* Clear any pending status by doing a dummy ioctl first */
        printf("\n  Clearing any pending status...\n");
        int dummy_status = 0;
        ioctl(fd_f, IOCTL_AIP_IRQ_WAIT_CMP, &dummy_status);  /* Non-blocking clear */

        /* Also try clearing bit 3 which might be "done" flag */
        aip_writel(aip_io, AIP_F_CTRL, 0x00);
        usleep(100);
        printf("  CTRL after clear: 0x%08x\n", aip_readl(aip_io, AIP_F_CTRL));

        /* Now try starting */
        printf("  Starting convolution...\n");
        aip_writel(aip_io, AIP_F_CTRL, 0x01);  /* Set only start bit */
        printf("  CTRL after start: 0x%08x\n", aip_readl(aip_io, AIP_F_CTRL));

        /* Try ioctl wait */
        int status = 0;
        printf("  Trying ioctl IRQ_WAIT_CMP...\n");

        /* Set a short timeout before ioctl (use alarm) */
        alarm(2);
        int ret = ioctl(fd_f, IOCTL_AIP_IRQ_WAIT_CMP, &status);
        alarm(0);

        if (ret >= 0) {
            printf("  ioctl returned %d, status=%d\n", ret, status);
            printf("  CTRL after: 0x%08x\n", aip_readl(aip_io, AIP_F_CTRL));
            printf("  Output: %d %d %d %d ...\n", out_buf[0], out_buf[1], out_buf[2], out_buf[3]);
        } else {
            printf("  ioctl failed or timed out, CTRL=0x%08x\n", aip_readl(aip_io, AIP_F_CTRL));
            /* Reset AIP-F to recover */
            aip_writel(aip_io, AIP_F_CTRL, 0x02);
            usleep(1000);
        }
    }

    munmap(oram, ORAM_SIZE);

cleanup:
    /* Cleanup */
    munmap((void*)aip_io, AIP_IOSIZE);
    close(fd_mem);
    if (fd_f >= 0) close(fd_f);
    if (fd_p >= 0) close(fd_p);
    if (fd_t >= 0) close(fd_t);
    if (fd_nna >= 0) close(fd_nna);

    printf("\n✓ AIP test complete\n");
    return 0;
}

