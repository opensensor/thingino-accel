/*
 * NNA DMA Descriptor Interface
 * Based on reverse-engineering of soc-nna kernel driver
 */

#ifndef NNA_DMA_H
#define NNA_DMA_H

#include <stdint.h>

/* NNA DMA I/O register base (from soc_nna_hw.h) */
#define NNA_DMA_IOBASE          0x12502000
#define NNA_DMA_IOSIZE          0x1000

/* NNA DMA descriptor RAM (from soc_nna_hw.h) */
#define NNA_DMA_DESRAM_ADDR     0x1250f000
#define NNA_DMA_DESRAM_SIZE     0x4000          /* 16KB */

/* Alternative addresses seen in some code */
#define NNA_DMA_FASTIO_ADDR     0x12508000      /* Fast I/O registers */
#define NNA_DMA_FASTIO_SIZE     0x20            /* 32 bytes */

/* ORAM base and size */
#define NNA_ORAM_BASE_ADDR      0x12600000
#define NNA_ORAM_BASE_SIZE      0xe0000         /* (1024-128)*1024 = 896KB */

/* NNA DMA I/O registers */
#define NNA_DMA_RCFG            0x0     /* Read config */
#define NNA_DMA_WCFG            0x4     /* Write config */
#define NNA_DMA_RCNT            0x8     /* Read count */
#define NNA_DMA_WCNT            0xc     /* Write count */

/* RCFG/WCFG register bits */
#define DMA_CFG_START           (1 << 0)
#define DMA_CFG_DES_ADDR_SHIFT  1
#define DMA_CFG_DES_ADDR_MASK   (0x7ff << 1)

/* Descriptor format (64-bit):
 * Bits 50-51: Flag (0=CNT, 1=LINK, 2=END)
 * Bits 40-49: Data length / 64 - 1 (10 bits, max 64KB)
 * Bits 26-39: ORAM address / 64 (14 bits)
 * Bits 0-25:  DDR address / 64 (26 bits)
 * 
 * For CNT descriptor (first in chain):
 * Bits 50-51: Flag = 0 (CNT)
 * Bits 0-19:  Total bytes in chain (20 bits)
 */
#define DES_CFG_FLAG_SHIFT      50ULL
#define DES_CFG_FLAG_MASK       (0x3ULL << 50)
#define DES_CFG_CNT             0ULL    /* Count descriptor (first in chain) */
#define DES_CFG_LINK            1ULL    /* Link to next descriptor */
#define DES_CFG_END             2ULL    /* End of chain */

#define DES_TOTAL_BYTES_SHIFT   0ULL
#define DES_TOTAL_BYTES_MASK    (0xfffffULL)    /* 20 bits, max 1MB */

#define DES_DATA_LEN_SHIFT      40ULL
#define DES_DATA_LEN_MASK       (0x3ffULL << 40)  /* 10 bits */

#define DES_ORAM_ADDR_SHIFT     26ULL
#define DES_ORAM_ADDR_MASK      (0x3fffULL << 26) /* 14 bits */

#define DES_DDR_ADDR_SHIFT      0ULL
#define DES_DDR_ADDR_MASK       (0x3ffffffULL)    /* 26 bits */

#define NNA_ADDR_ALIGN          64      /* All addresses must be 64-byte aligned */
#define NNA_MAX_TRANSFER        65536   /* Max bytes per descriptor (64KB) */

/* DMA command structure (matches kernel nna_dma_cmd_t) */
typedef struct {
    uint32_t d_va_st_addr;      /* DDR virtual start address */
    uint32_t o_va_st_addr;      /* ORAM virtual start address */
    uint32_t o_va_mlc_addr;     /* ORAM malloc base address */
    uint32_t o_mlc_bytes;       /* ORAM malloc size */
    uint32_t data_bytes;        /* Total bytes to transfer */
    uint32_t des_link;          /* Link to next command? */
} nna_dma_cmd_t;

/* DMA command set (matches kernel nna_dma_cmd_set_t) */
typedef struct {
    uint32_t rd_cmd_cnt;        /* Number of read commands */
    uint32_t rd_cmd_st_idx;     /* Read command start index */
    uint32_t wr_cmd_cnt;        /* Number of write commands */
    uint32_t wr_cmd_st_idx;     /* Write command start index */
    nna_dma_cmd_t *d_va_cmd;    /* Command array */
    uint32_t *d_va_chn;         /* Channel array */
    struct {
        uint32_t rcmd_st_idx;
        uint32_t wcmd_st_idx;
        uint32_t dma_chn_num;
        uint32_t finish;
    } des_rslt;
} nna_dma_cmd_set_t;

/* Build a count descriptor (first in chain) */
static inline uint64_t nna_des_cnt(uint32_t total_bytes) {
    return ((DES_CFG_CNT << DES_CFG_FLAG_SHIFT) & DES_CFG_FLAG_MASK)
         | ((uint64_t)total_bytes & DES_TOTAL_BYTES_MASK);
}

/* Build a transfer descriptor */
static inline uint64_t nna_des_transfer(uint32_t ddr_addr, uint32_t oram_addr, 
                                         uint32_t len, int is_last) {
    uint64_t flag = is_last ? DES_CFG_END : DES_CFG_LINK;
    uint64_t len_field = ((len >> 6) - 1) & 0x3ff;  /* len/64 - 1 */
    uint64_t oram_field = (oram_addr >> 6) & 0x3fff;
    uint64_t ddr_field = (ddr_addr >> 6) & 0x3ffffff;
    
    return ((flag << DES_CFG_FLAG_SHIFT) & DES_CFG_FLAG_MASK)
         | ((len_field << DES_DATA_LEN_SHIFT) & DES_DATA_LEN_MASK)
         | ((oram_field << DES_ORAM_ADDR_SHIFT) & DES_ORAM_ADDR_MASK)
         | (ddr_field & DES_DDR_ADDR_MASK);
}

/* NNA DMA context */
typedef struct {
    int fd;                     /* /dev/soc-nna file descriptor */
    void *desram_vaddr;         /* Mapped descriptor RAM */
    void *dma_io_vaddr;         /* Mapped DMA I/O registers */
    void *oram_vaddr;           /* Mapped ORAM */
    uint32_t oram_paddr;        /* ORAM physical address */
} nna_dma_ctx_t;

/* Initialize NNA DMA */
int nna_dma_init(nna_dma_ctx_t *ctx);

/* Cleanup NNA DMA */
void nna_dma_cleanup(nna_dma_ctx_t *ctx);

/* Transfer data from DDR to ORAM */
int nna_dma_ddr_to_oram(nna_dma_ctx_t *ctx, uint32_t ddr_paddr, 
                        uint32_t oram_offset, uint32_t size);

/* Transfer data from ORAM to DDR */
int nna_dma_oram_to_ddr(nna_dma_ctx_t *ctx, uint32_t oram_offset,
                        uint32_t ddr_paddr, uint32_t size);

/* Wait for DMA completion */
int nna_dma_wait(nna_dma_ctx_t *ctx);

#endif /* NNA_DMA_H */

