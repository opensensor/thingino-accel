/*
 * AIP Library Test - Replicate libdrivers.so + libaip.so initialization
 * Based on reverse engineering of __aie_mmap, aip_mem_init, get_aip_ioaddr
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

/* AIP I/O base (get_aip_ioaddr maps this) */
#define AIP_IOBASE      0x12b00000
#define AIP_IOSIZE      0x1000

/* ORAM initial base (libdrivers starts here, adjusts based on L2 cache) */
#define ORAM_BASE_INIT  0x12600000
#define ORAM_MAX_SIZE   0x80000  /* 512KB max */

/* CPM (Clock/Power Management) base */
#define CPM_BASE        0x10000000
#define CPM_SIZE        0x1000
#define CPM_CLKGR0      0x20   /* Clock Gate Register 0 */
#define CPM_CLKGR1      0x28   /* Clock Gate Register 1 */
#define CPM_SRBC        0xC4   /* Soft Reset and Bus Control */

/* AHB/Bus config (used by soc-nna driver) */
#define AHB_BASE        0x13010000
#define AHB_SIZE        0x1000

/* NNA/L2 cache registers - libdrivers reads 0x12200060 for L2 size */
#define NNA_L2_BASE     0x12200000
#define NNA_L2_SIZE     0x1000
#define NNA_L2_CFG_REG  0x60

/* NNDMA I/O - libdrivers maps 0x12508000, size 0x20 */
#define NNDMA_IO_BASE   0x12508000
#define NNDMA_IO_SIZE   0x20

/* NNDMA DESRAM - libdrivers maps 0x12500000 */
#define NNDMA_DESRAM_BASE  0x12500000
#define NNDMA_DESRAM_SIZE  0x1000

/* AIP ioctls (magic 'P' = 0x50) */
#define IOCTL_AIP_IRQ_WAIT_CMP  0xc0045000
#define IOCTL_AIP_MALLOC        0xc0045001
#define IOCTL_AIP_FREE          0xc0045002

/* AIP-T (Tensor/Resize) register offsets */
#define AIP_T_CTRL      0x000
#define AIP_T_CFG       0x008
#define AIP_T_NODECFG   0x00C
#define AIP_T_IN_ADDR   0x010
#define AIP_T_SRC_STR   0x014
#define AIP_T_SRC_SIZE  0x018
#define AIP_T_DST_ADDR  0x01C
#define AIP_T_DST_SIZE  0x020
#define AIP_T_DST_STR   0x024

/* AIP-F (Feature/Conv) register offsets */
#define AIP_F_CTRL      0x200
#define AIP_F_CFG       0x208
#define AIP_F_NODECFG   0x20C
#define AIP_F_IN_ADDR   0x210
#define AIP_F_OUT_ADDR  0x214
#define AIP_F_IN_SIZE   0x218
#define AIP_F_OUT_SIZE  0x21C
#define AIP_F_KERN_ADDR 0x220
#define AIP_F_BIAS_ADDR 0x224
#define AIP_F_KERN_SIZE 0x228
#define AIP_F_STRIDE    0x22C
#define AIP_F_INOUT_CH  0x230
#define AIP_F_PAD_POOL  0x234
#define AIP_F_SCALE_SH  0x238

static inline uint32_t aip_readl(volatile uint32_t *base, uint32_t offset) {
    return base[offset / 4];
}

static inline void aip_writel(volatile uint32_t *base, uint32_t offset, uint32_t val) {
    base[offset / 4] = val;
}

/* Reset AIP-T: write bit 1, wait for it to clear (from decompiled aip_t_reset) */
static void aip_t_reset(volatile uint32_t *aip_io) {
    printf("  Resetting AIP-T...\n");
    aip_writel(aip_io, AIP_T_CTRL, 0x02);  /* Reset bit */
    int timeout = 1000;
    while ((aip_readl(aip_io, AIP_T_CTRL) & 0x02) && timeout-- > 0) {
        usleep(10);
    }
    printf("  AIP-T reset complete, CTRL=0x%08x\n", aip_readl(aip_io, AIP_T_CTRL));
}

/* Reset AIP-F: write bit 1, wait for it to clear (from decompiled aip_f_reset) */
static void aip_f_reset(volatile uint32_t *aip_io) {
    printf("  Resetting AIP-F...\n");
    aip_writel(aip_io, AIP_F_CTRL, 0x02);  /* Reset bit */
    int timeout = 1000;
    while ((aip_readl(aip_io, AIP_F_CTRL) & 0x02) && timeout-- > 0) {
        usleep(10);
    }
    printf("  AIP-F reset complete, CTRL=0x%08x\n", aip_readl(aip_io, AIP_F_CTRL));
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    int fd_nna = -1, fd_mem = -1, fd_aip_t = -1, fd_aip_f = -1, fd_nna_lock = -1;
    volatile uint32_t *aip_io = NULL;
    volatile uint32_t *cpm = NULL;
    volatile uint8_t *oram = NULL;
    volatile uint32_t *nna_l2 = NULL;
    volatile uint32_t *nndma_io = NULL;
    volatile uint8_t *nndma_desram = NULL;
    uint32_t oram_base = ORAM_BASE_INIT;
    uint32_t oram_size = 0;
    int ret;

    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  AIP Direct Test (Mimicking libdrivers + libaip.so flow)     ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    /* === Phase 1: Open devices like __aie_mmap does === */
    printf("=== Phase 1: Opening devices (like __aie_mmap) ===\n");

    /* Open /dev/mem first */
    fd_mem = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd_mem < 0) {
        perror("open /dev/mem");
        goto cleanup;
    }
    printf("Opened /dev/mem (fd=%d)\n", fd_mem);

    /* Open /dev/soc-nna (enables NNA clocks) */
    fd_nna = open("/dev/soc-nna", O_RDWR);
    if (fd_nna < 0) {
        perror("open /dev/soc-nna");
        printf("  ERROR: soc-nna required for NNA clock enable!\n");
        goto cleanup;
    }
    printf("Opened /dev/soc-nna (fd=%d)\n", fd_nna);

    /* Open /dev/nna_lock for multiprocess mode (optional but recommended) */
    fd_nna_lock = open("/dev/nna_lock", O_RDWR);
    if (fd_nna_lock < 0) {
        perror("open /dev/nna_lock (optional)");
    } else {
        printf("Opened /dev/nna_lock (fd=%d)\n", fd_nna_lock);
    }

    /* Open AIP device files */
    fd_aip_t = open("/dev/jzaip_t", O_RDWR);
    if (fd_aip_t >= 0) printf("Opened /dev/jzaip_t (fd=%d)\n", fd_aip_t);

    fd_aip_f = open("/dev/jzaip_f", O_RDWR);
    if (fd_aip_f >= 0) printf("Opened /dev/jzaip_f (fd=%d)\n", fd_aip_f);

    /* === Phase 2: Map NNA L2 config to compute ORAM base (like __aie_mmap) === */
    printf("\n=== Phase 2: Computing ORAM base from L2 cache config ===\n");

    nna_l2 = mmap(NULL, NNA_L2_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd_mem, NNA_L2_BASE);
    if (nna_l2 == MAP_FAILED) {
        perror("mmap NNA L2");
        nna_l2 = NULL;
        goto cleanup;
    }

    /* Read L2 cache size from register 0x60 bits 10-12 (like libdrivers) */
    uint32_t l2_reg = nna_l2[NNA_L2_CFG_REG / 4];
    uint32_t l2_size_code = (l2_reg >> 10) & 0x7;
    uint32_t l2_cache_size = 0;

    switch (l2_size_code) {
        case 1: l2_cache_size = 0x20000; oram_size = 0x60000; break;  /* 128KB L2, 384KB ORAM */
        case 2: l2_cache_size = 0x40000; oram_size = 0x40000; break;  /* 256KB L2, 256KB ORAM */
        case 3: l2_cache_size = 0x80000; oram_size = 0; break;        /* 512KB L2, 0 ORAM */
        case 4: l2_cache_size = 0x100000; oram_size = 0; break;       /* 1MB L2 */
        default: l2_cache_size = 0; oram_size = 0x80000; break;
    }
    oram_base = ORAM_BASE_INIT + l2_cache_size;

    printf("  L2 config reg 0x60 = 0x%08x\n", l2_reg);
    printf("  L2 size code = %d, L2 cache = 0x%x bytes\n", l2_size_code, l2_cache_size);
    printf("  ORAM base = 0x%08x, size = 0x%x bytes\n", oram_base, oram_size);

    /* === Phase 3: Map NNDMA I/O and DESRAM (like __aie_mmap) === */
    printf("\n=== Phase 3: Mapping NNDMA regions ===\n");

    /* Map NNDMA I/O at 0x12508000 (size 0x20) */
    nndma_io = mmap(NULL, NNDMA_IO_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd_mem, NNDMA_IO_BASE);
    if (nndma_io == MAP_FAILED) {
        perror("mmap NNDMA I/O");
        nndma_io = NULL;
    } else {
        printf("  Mapped NNDMA I/O at %p (paddr 0x%08x)\n", (void*)nndma_io, NNDMA_IO_BASE);
    }

    /* Map NNDMA DESRAM at 0x12500000 */
    nndma_desram = mmap(NULL, NNDMA_DESRAM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd_mem, NNDMA_DESRAM_BASE);
    if (nndma_desram == MAP_FAILED) {
        perror("mmap NNDMA DESRAM");
        nndma_desram = NULL;
    } else {
        printf("  Mapped NNDMA DESRAM at %p (paddr 0x%08x)\n", (void*)nndma_desram, NNDMA_DESRAM_BASE);
    }

    /* === Phase 4: Map ORAM with computed base === */
    printf("\n=== Phase 4: Mapping ORAM ===\n");
    if (oram_size > 0) {
        oram = mmap(NULL, oram_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_mem, oram_base);
        if (oram == MAP_FAILED) {
            perror("mmap ORAM");
            oram = NULL;
        } else {
            printf("  Mapped ORAM at %p (paddr 0x%08x, size 0x%x)\n", (void*)oram, oram_base, oram_size);
        }
    } else {
        printf("  ORAM not available (all memory used as L2 cache)\n");
    }

    /* === Phase 5: Map AIP I/O (like get_aip_ioaddr in libaip.so) === */
    printf("\n=== Phase 5: Mapping AIP I/O (like get_aip_ioaddr) ===\n");
    aip_io = mmap(NULL, AIP_IOSIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd_mem, AIP_IOBASE);
    if (aip_io == MAP_FAILED) {
        perror("mmap AIP");
        aip_io = NULL;
        goto cleanup;
    }
    printf("  Mapped AIP I/O at %p (paddr 0x%08x)\n", (void*)aip_io, AIP_IOBASE);

    /* Map CPM for diagnostics */
    cpm = mmap(NULL, CPM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd_mem, CPM_BASE);
    if (cpm != MAP_FAILED) {
        printf("  Mapped CPM at %p\n", (void*)cpm);
    } else {
        cpm = NULL;
    }

    /* === Phase 6: Check clocks and status === */
    if (cpm) {
        printf("\n=== CPM Clock Gate Registers ===\n");
        printf("  CLKGR0: 0x%08x\n", cpm[CPM_CLKGR0/4]);
        printf("  CLKGR1: 0x%08x\n", cpm[CPM_CLKGR1/4]);
    }

    /* Print NNDMA I/O status */
    if (nndma_io) {
        printf("\n=== NNDMA I/O Registers (0x12508000) ===\n");
        for (int i = 0; i < 8; i++) {
            printf("  [%02x]: 0x%08x\n", i * 4, nndma_io[i]);
        }
    }

    /* Apply NNA register clear like jz_aip_open does */
    printf("\n=== Clearing NNA registers (like jz_aip_open) ===\n");
    nna_l2[0] = 0;  /* Write 0 to 0x12200000 */
    __sync_synchronize();
    printf("  Wrote 0 to NNA base register\n");

    /* Enable AIP clock - bit 25 of register 0x12200060 (like aip_open_clk) */
    printf("\n=== Enabling AIP clock ===\n");
    uint32_t clk_reg = nna_l2[0x60/4];
    printf("  Register 0x60 before: 0x%08x (bit 25=%d)\n", clk_reg, (clk_reg >> 25) & 1);
    clk_reg |= (1 << 25);  /* Set bit 25 to enable AIP clock */
    nna_l2[0x60/4] = clk_reg;
    __sync_synchronize();
    uint32_t clk_after = nna_l2[0x60/4];
    printf("  Register 0x60 after:  0x%08x (bit 25=%d)\n", clk_after, (clk_after >> 25) & 1);

    /* Also check CPM SRBC for AIP soft reset */
    if (cpm) {
        uint32_t srbc = cpm[CPM_SRBC/4];
        printf("  CPM SRBC (0xC4): 0x%08x\n", srbc);
    }

    /* Map and write the mystery register from soc_nna_open (0x13012038) */
    printf("\n=== Configuring system register (like soc_nna_open) ===\n");
    volatile uint32_t *sys_reg = mmap(NULL, 0x1000, PROT_READ | PROT_WRITE,
                                       MAP_SHARED, fd_mem, 0x13012000);
    if (sys_reg != MAP_FAILED) {
        printf("  System bus regs at 0x13012000:\n");
        for (int i = 0; i < 0x80; i += 4) {
            if (i % 16 == 0) printf("    %02x:", i);
            printf(" %08x", sys_reg[i/4]);
            if (i % 16 == 12) printf("\n");
        }

        uint32_t old_val = sys_reg[0x38/4];
        sys_reg[0x38/4] = 0x88404002;
        __sync_synchronize();
        uint32_t new_val = sys_reg[0x38/4];
        printf("  Wrote 0x88404002 to 0x13012038 (was 0x%08x, now 0x%08x)\n", old_val, new_val);
        munmap((void*)sys_reg, 0x1000);
    } else {
        printf("  Failed to map 0x13012000\n");
    }

    /* Check NNDMA registers - this might need configuration */
    printf("\n=== NNDMA Control Check ===\n");
    volatile uint32_t *nndma_ctrl = mmap(NULL, 0x1000, PROT_READ | PROT_WRITE,
                                          MAP_SHARED, fd_mem, 0x12502000);
    if (nndma_ctrl != MAP_FAILED) {
        printf("  NNDMA regs at 0x12502000:\n");
        for (int i = 0; i < 0x40; i += 4) {
            if (i % 16 == 0) printf("    %02x:", i);
            printf(" %08x", nndma_ctrl[i/4]);
            if (i % 16 == 12) printf("\n");
        }
        munmap((void*)nndma_ctrl, 0x1000);
    }

    /* Dump initial state - all AIP-T registers */
    printf("\n=== Initial State - AIP-T Registers ===\n");
    for (int i = 0; i < 0x80; i += 4) {
        if (i % 16 == 0) printf("  0x%02x:", i);
        printf(" %08x", aip_readl(aip_io, i));
        if (i % 16 == 12) printf("\n");
    }
    printf("\n=== Initial State - AIP-F Registers ===\n");
    for (int i = 0x200; i < 0x280; i += 4) {
        if ((i - 0x200) % 16 == 0) printf("  0x%02x:", i - 0x200);
        printf(" %08x", aip_readl(aip_io, i));
        if ((i - 0x200) % 16 == 12) printf("\n");
    }

    /* ==================== TEST 1: AIP-T RESIZE ==================== */
    printf("\n========== TEST 1: AIP-T Resize ==========\n");
    if (fd_aip_t >= 0 && oram) {
        /* Reset AIP-T first (critical - clears stuck state) */
        aip_t_reset(aip_io);

        /* Set up 8x8 -> 4x4 resize using ORAM */
        uint32_t src_off = 0x0000;
        uint32_t dst_off = 0x1000;

        /* Fill source with pattern 0-63 */
        for (int i = 0; i < 64; i++) oram[src_off + i] = i;
        /* Clear dest */
        memset((void*)(oram + dst_off), 0xAA, 16);

        printf("  Source (8x8):\n");
        for (int y = 0; y < 8; y++) {
            printf("    ");
            for (int x = 0; x < 8; x++) printf("%3d ", oram[src_off + y*8 + x]);
            printf("\n");
        }

        /* Configure AIP-T registers based on aip_t_config_nodes decompilation */
        aip_writel(aip_io, AIP_T_NODECFG, 0xffffffff);  /* Single node */
        aip_writel(aip_io, AIP_T_SRC_STR, 8);           /* Source stride */
        aip_writel(aip_io, AIP_T_SRC_SIZE, (8 << 16) | 8);  /* 8x8 */
        aip_writel(aip_io, AIP_T_DST_SIZE, (4 << 16) | 4);  /* 4x4 */
        aip_writel(aip_io, AIP_T_DST_STR, 4);           /* Dest stride */

        /* Set resize coefficients (registers 0x28-0x4c from aip_t_init)
         * For 2x downscale (8->4), scale = 2.0 in Q16 = 0x20000 */
        uint32_t scale_x = 0x20000;  /* 2.0 in Q16 for 8->4 */
        uint32_t scale_y = 0x20000;
        aip_writel(aip_io, 0x28, scale_x);      /* X scale */
        aip_writel(aip_io, 0x2c, 0);            /* X start offset */
        aip_writel(aip_io, 0x30, scale_y);      /* Y scale */
        aip_writel(aip_io, 0x34, 0);            /* Y start offset */
        aip_writel(aip_io, 0x38, 0);            /* Padding/margin? */
        aip_writel(aip_io, 0x3c, 0);
        aip_writel(aip_io, 0x40, 0);
        aip_writel(aip_io, 0x44, 0);
        aip_writel(aip_io, 0x48, 0);
        aip_writel(aip_io, 0x4c, 0);

        /* CFG: Based on aip_t_init: arg5 | 0x43c */
        aip_writel(aip_io, AIP_T_CFG, 0x43c);

        /* aip_t_init writes 1 to CTRL after configuration! */
        aip_writel(aip_io, AIP_T_CTRL, 1);
        __sync_synchronize();

        /* Set addresses and start (like aip_t_start) */
        aip_writel(aip_io, AIP_T_IN_ADDR, oram_base + src_off);
        aip_writel(aip_io, AIP_T_DST_ADDR, oram_base + dst_off);

        printf("  Configured: SRC=0x%08x DST=0x%08x\n",
               aip_readl(aip_io, AIP_T_IN_ADDR), aip_readl(aip_io, AIP_T_DST_ADDR));
        printf("  CTRL after init config (=1): 0x%08x\n", aip_readl(aip_io, AIP_T_CTRL));
        printf("  Starting resize (CTRL = 0x10)...\n");

        __sync_synchronize();
        aip_writel(aip_io, AIP_T_CTRL, 0x10);  /* Start bit for AIP-T */
        __sync_synchronize();

        printf("  CTRL after start: 0x%08x\n", aip_readl(aip_io, AIP_T_CTRL));

        /* Manual polling first to see if hardware ever completes */
        printf("  Polling CTRL for completion (bit 4 should clear)...\n");
        int poll_cnt = 0;
        for (int i = 0; i < 100; i++) {
            uint32_t ctrl_val = aip_readl(aip_io, AIP_T_CTRL);
            if ((ctrl_val & 0x10) == 0) {
                printf("    Completed after %d polls, CTRL=0x%08x\n", i, ctrl_val);
                poll_cnt = i;
                break;
            }
            usleep(1000);  /* 1ms */
            if (i == 99) printf("    Timeout after 100ms, CTRL=0x%08x\n", ctrl_val);
        }

        /* Also try ioctl wait */
        int status = 0;
        ret = ioctl(fd_aip_t, IOCTL_AIP_IRQ_WAIT_CMP, &status);
        printf("  ioctl returned: %d, status=%d\n", ret, status);
        printf("  CTRL now: 0x%08x  IRQ(0x04): 0x%08x\n",
               aip_readl(aip_io, AIP_T_CTRL), aip_readl(aip_io, 0x04));

        /* Check output */
        printf("  Dest (4x4) after resize:\n");
        for (int y = 0; y < 4; y++) {
            printf("    ");
            for (int x = 0; x < 4; x++) printf("%3d ", oram[dst_off + y*4 + x]);
            printf("\n");
        }
    } else {
        printf("  Skipped - /dev/jzaip_t not available\n");
    }

    /* ==================== TEST 2: AIP-F CONVOLUTION ==================== */
    printf("\n========== TEST 2: AIP-F Convolution ==========\n");

    /* Allocate chain buffer via ioctl */
    printf("Allocating chain buffer via ioctl...\n");
    struct { uint32_t vaddr; uint32_t paddr; uint32_t size; } buf = {0, 0, 0x36000};
    ret = ioctl(fd_aip_f, IOCTL_AIP_MALLOC, &buf);
    if (ret < 0) {
        perror("IOCTL_AIP_MALLOC");
        printf("  Chain buffer allocation failed!\n");
        goto cleanup;
    }
    printf("  Chain buffer: vaddr=0x%08x paddr=0x%08x size=0x%x\n",
           buf.vaddr, buf.paddr, buf.size);

    /* Map the chain buffer */
    volatile uint8_t *ddr_buf = mmap(NULL, buf.size, PROT_READ | PROT_WRITE,
                                     MAP_SHARED, fd_aip_f, buf.paddr);
    if (ddr_buf == MAP_FAILED) {
        perror("mmap chain buffer");
        ddr_buf = NULL;
    } else {
        printf("  Mapped chain buffer at %p\n", (void*)ddr_buf);
    }

    /* Reset AIP-F first (critical - clears stuck state) */
    aip_f_reset(aip_io);

    /* Set up simple 8x8 convolution - use ORAM instead of DDR */
    printf("\n=== Setting up 8x8 convolution (using ORAM) ===\n");

    /* Use ORAM for data - offset from where we put input for AIP-T */
    volatile uint8_t *data_buf = oram + 0x10000;  /* Use ORAM at offset 64KB */
    uint32_t data_paddr = oram_base + 0x10000;

    printf("  Using ORAM buffer at paddr 0x%08x (vaddr %p)\n", data_paddr, (void*)data_buf);

    uint32_t in_off = 0x0000;       /* Input at offset 0 */
    uint32_t out_off = 0x1000;      /* Output at offset 4KB */
    uint32_t kern_off = 0x2000;     /* Kernel at offset 8KB */
    uint32_t bias_off = 0x3000;     /* Bias at offset 12KB */

    /* Fill input with 1s (8x8 = 64 bytes) */
    memset((void*)(data_buf + in_off), 1, 64);
    /* Fill kernel with 2 (1x1 kernel) */
    memset((void*)(data_buf + kern_off), 2, 4);
    /* Fill bias with 0 */
    memset((void*)(data_buf + bias_off), 0, 4);
    /* Clear output */
    memset((void*)(data_buf + out_off), 0xAA, 64);

    printf("  Input[0-3]:  %d %d %d %d\n", data_buf[in_off], data_buf[in_off+1], data_buf[in_off+2], data_buf[in_off+3]);
    printf("  Kernel[0]:   %d\n", data_buf[kern_off]);
    printf("  Output[0-3]: %d %d %d %d (before)\n", data_buf[out_off], data_buf[out_off+1], data_buf[out_off+2], data_buf[out_off+3]);

    /* Memory barrier and cache sync for DMA coherency */
    __sync_synchronize();

    /* Configure AIP-F registers (like aip_f_config_nodes for single node) */
    printf("\n=== Configuring AIP-F registers ===\n");

    /* NODECFG = 0xffffffff for single node */
    aip_writel(aip_io, AIP_F_NODECFG, 0xffffffff);

    /* Set addresses using DDR or ORAM */
    aip_writel(aip_io, AIP_F_IN_ADDR, data_paddr + in_off);
    aip_writel(aip_io, AIP_F_OUT_ADDR, data_paddr + out_off);

    /* Size: (h << 16) | w, then +0x10 as per decompilation */
    uint32_t size = (8 << 16) | 8;
    aip_writel(aip_io, AIP_F_IN_SIZE, size + 0x10);
    aip_writel(aip_io, AIP_F_OUT_SIZE, size + 0x10);

    aip_writel(aip_io, AIP_F_KERN_ADDR, data_paddr + kern_off);
    aip_writel(aip_io, AIP_F_BIAS_ADDR, data_paddr + bias_off);

    /* Kernel size: 1x1 */
    aip_writel(aip_io, AIP_F_KERN_SIZE, (1 << 16) | 1);

    /* Stride: 1x1 */
    aip_writel(aip_io, AIP_F_STRIDE, (1 << 16) | 1);

    /* In/Out channels: 1/1 */
    aip_writel(aip_io, AIP_F_INOUT_CH, (1 << 16) | 1);

    /* Padding/pooling: 0 */
    aip_writel(aip_io, AIP_F_PAD_POOL, 0);

    /* Scale/shift: (scale << 16) | shift, scale=1, shift=0 */
    aip_writel(aip_io, AIP_F_SCALE_SH, (1 << 16) | 0);

    /* CFG: 0x14 for single node mode (from decompiled aip_f_init) */
    uint32_t cfg = 0x14;  /* Always use 0x14 as per decompiled code */
    aip_writel(aip_io, AIP_F_CFG, cfg);

    printf("  IN_ADDR:   0x%08x\n", aip_readl(aip_io, AIP_F_IN_ADDR));
    printf("  OUT_ADDR:  0x%08x\n", aip_readl(aip_io, AIP_F_OUT_ADDR));
    printf("  IN_SIZE:   0x%08x\n", aip_readl(aip_io, AIP_F_IN_SIZE));
    printf("  KERN_SIZE: 0x%08x\n", aip_readl(aip_io, AIP_F_KERN_SIZE));
    printf("  CFG:       0x%08x\n", aip_readl(aip_io, AIP_F_CFG));

    /* Start convolution - use read-modify-write like aip_f_start */
    printf("\n=== Starting convolution ===\n");
    uint32_t ctrl_before = aip_readl(aip_io, AIP_F_CTRL);
    printf("  CTRL before: 0x%08x\n", ctrl_before);

    /* Ensure all register writes are complete */
    __sync_synchronize();

    /* aip_f_start does: CTRL |= 1 for single node, CTRL |= 4 for chain */
    uint32_t ctrl_start = ctrl_before | 0x01;  /* Single node mode */
    aip_writel(aip_io, AIP_F_CTRL, ctrl_start);
    __sync_synchronize();

    printf("  CTRL after:  0x%08x\n", aip_readl(aip_io, AIP_F_CTRL));

    /* Poll for completion first to see if hardware ever finishes */
    printf("\n=== Polling for completion ===\n");
    for (int i = 0; i < 100; i++) {
        uint32_t ctrl = aip_readl(aip_io, AIP_F_CTRL);
        uint32_t irq = aip_readl(aip_io, 0x204);
        if (i < 5 || (ctrl & 0x01) == 0 || irq != 0) {
            printf("  [%d] CTRL=0x%08x IRQ=0x%08x\n", i, ctrl, irq);
        }
        if ((ctrl & 0x01) == 0) {
            printf("  Hardware completed at iteration %d!\n", i);
            break;
        }
        usleep(1000);  /* 1ms */
    }

    /* Wait for completion via ioctl (like aip_f_wait) */
    printf("\n=== Waiting for completion via ioctl ===\n");
    int status = 0;
    ret = ioctl(fd_aip_f, IOCTL_AIP_IRQ_WAIT_CMP, &status);
    printf("  ioctl returned: %d, status=%d\n", ret, status);
    printf("  CTRL now: 0x%08x  IRQ(0x204): 0x%08x\n",
           aip_readl(aip_io, AIP_F_CTRL), aip_readl(aip_io, 0x204));

    /* Memory barrier to ensure we see any DMA writes */
    __sync_synchronize();

    /* Check output */
    printf("\n=== Results ===\n");
    printf("  Output[0-7]: %d %d %d %d %d %d %d %d\n",
           data_buf[out_off], data_buf[out_off+1], data_buf[out_off+2], data_buf[out_off+3],
           data_buf[out_off+4], data_buf[out_off+5], data_buf[out_off+6], data_buf[out_off+7]);

    /* Also check via direct ORAM read to rule out cache issues */
    printf("  ORAM direct[0-7]: %d %d %d %d %d %d %d %d\n",
           oram[out_off], oram[out_off+1], oram[out_off+2], oram[out_off+3],
           oram[out_off+4], oram[out_off+5], oram[out_off+6], oram[out_off+7]);

cleanup:
    if (ddr_buf) munmap((void*)ddr_buf, buf.size);
    if (nndma_desram) munmap((void*)nndma_desram, NNDMA_DESRAM_SIZE);
    if (nndma_io) munmap((void*)nndma_io, NNDMA_IO_SIZE);
    if (nna_l2) munmap((void*)nna_l2, NNA_L2_SIZE);
    if (oram) munmap((void*)oram, oram_size);
    if (cpm) munmap((void*)cpm, CPM_SIZE);
    if (aip_io) munmap((void*)aip_io, AIP_IOSIZE);
    if (fd_mem >= 0) close(fd_mem);
    if (fd_aip_f >= 0) close(fd_aip_f);
    if (fd_aip_t >= 0) close(fd_aip_t);
    if (fd_nna_lock >= 0) close(fd_nna_lock);
    if (fd_nna >= 0) close(fd_nna);

    printf("\n✓ Test complete\n");
    return 0;
}
