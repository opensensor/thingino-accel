/*
 * AIP (AI Processor) Interface
 * Reverse-engineered from libaip.so decompilation
 * 
 * The T41 has three AIP pipes:
 * - AIP-T (Tensor/Resize): registers 0x000-0x04C
 * - AIP-F (Feature/Conv):  registers 0x200-0x240
 * - AIP-P (Perspective):   registers 0x300-0x398
 */

#ifndef AIP_H
#define AIP_H

#include <stdint.h>

/* AIP I/O base address (from libaip.so get_aip_ioaddr) */
#define AIP_IOBASE              0x12b00000
#define AIP_IOSIZE              0x1000

/* AIP device paths */
#define AIP_DEV_F               "/dev/jzaip_f"
#define AIP_DEV_P               "/dev/jzaip_p"
#define AIP_DEV_T               "/dev/jzaip_t"

/* AIP ioctl commands - using 'P' (0x50) as magic number */
/* These decode as _IOWR('P', num, int) */
#define AIP_MAGIC               'P'         /* 0x50 */
#define IOCTL_AIP_IRQ_WAIT_CMP  0xc0045000  /* Wait for completion, returns status in int* */
#define IOCTL_AIP_MALLOC        0xc0045001  /* Allocate chain buffer */
#define IOCTL_AIP_FREE          0xc0045002  /* Free chain buffer */

/* AIP malloc buffer structure (from aip_chainbuf_alloc decompilation) */
typedef struct {
    void *vaddr;      /* Returned: virtual address */
    void *paddr;      /* Returned: physical address */
    int size;         /* Input/Output: requested/actual size */
} aip_buf_t;

/*=== AIP-T (Tensor/Resize) Registers ===*/
#define AIP_T_CTRL              0x000   /* Control: bit0=start, bit1=reset, bit4=chain_start */
#define AIP_T_CFG               0x008   /* Configuration */
#define AIP_T_NODECFG           0x00C   /* Node config (0xffffffff for single node) */
#define AIP_T_SRC_ADDR          0x010   /* Source address */
#define AIP_T_DST_ADDR          0x014   /* Destination address */
#define AIP_T_SRC_SIZE          0x018   /* Source size */
#define AIP_T_DST_SIZE          0x01C   /* Destination size */
#define AIP_T_SRC_STRIDE        0x020   /* Source stride (h<<16|w) */
#define AIP_T_DST_STRIDE        0x024   /* Destination stride (h<<16|w) */
#define AIP_T_SCALE             0x028   /* Scale factors */

/*=== AIP-F (Feature/Convolution) Registers ===*/
#define AIP_F_CTRL              0x200   /* Control: bit0=start, bit1=reset, bit2=chain_start */
#define AIP_F_CFG               0x208   /* Configuration (mode | 0x14 for single, | 0x0c for chain) */
#define AIP_F_NODECFG           0x20C   /* Node config (0xffffffff for single node) */
#define AIP_F_IN_ADDR           0x210   /* Input address */
#define AIP_F_OUT_ADDR          0x214   /* Output address */
#define AIP_F_IN_SIZE           0x218   /* Input size + 0x10 */
#define AIP_F_OUT_SIZE          0x21C   /* Output size + 0x10 */
#define AIP_F_KERNEL_ADDR       0x220   /* Kernel/weight address */
#define AIP_F_BIAS_ADDR         0x224   /* Bias address */
#define AIP_F_KERNEL_SIZE       0x228   /* Kernel size */
#define AIP_F_STRIDE            0x22C   /* Stride */
#define AIP_F_IN_CH_OUT_CH      0x230   /* (out_ch << 16) | in_ch */
#define AIP_F_PAD_POOL          0x234   /* (pool << 16) | padding */
#define AIP_F_SCALE_SHIFT       0x238   /* (shift << 16) | scale */
#define AIP_F_CHAIN_ADDR        0x23C   /* Chain buffer physical address */
#define AIP_F_CHAIN_SIZE        0x240   /* Chain buffer size */

/*=== AIP-P (Perspective) Registers ===*/
#define AIP_P_CTRL              0x300   /* Control: bit0=start, bit1=reset, bit2=chain_start */
#define AIP_P_CFG               0x308   /* Configuration */
#define AIP_P_NODECFG           0x30C   /* Node config */
#define AIP_P_SRC_ADDR          0x310   /* Source address */
#define AIP_P_DST_ADDR          0x314   /* Destination address */
/* ... more registers up to 0x398 */

/* AIP-F node structure (0x68 = 104 bytes per node in chain buffer) */
typedef struct {
    uint32_t node_cfg;          /* 0x00: 0xffffffff for valid node */
    uint32_t in_addr_reg;       /* 0x04: 0x12b0020c */
    uint32_t in_addr;           /* 0x08: Input address */
    uint32_t out_addr_reg;      /* 0x0C: 0x12b00210 */
    uint32_t out_addr;          /* 0x10: Output address */
    uint32_t in_size_reg;       /* 0x14: 0x12b00214 */
    uint32_t in_size;           /* 0x18: Input size + 0x10 */
    uint32_t out_size_reg;      /* 0x1C: 0x12b00218 */
    uint32_t out_size;          /* 0x20: Output size + 0x10 */
    uint32_t kernel_addr_reg;   /* 0x24: 0x12b0021c */
    uint32_t kernel_addr;       /* 0x28: Kernel address */
    uint32_t bias_addr_reg;     /* 0x2C: 0x12b00220 */
    uint32_t bias_addr;         /* 0x30: Bias address */
    uint32_t kernel_size_reg;   /* 0x34: 0x12b00224 */
    uint32_t kernel_size;       /* 0x38: Kernel size */
    uint32_t stride_reg;        /* 0x3C: 0x12b00228 */
    uint32_t stride;            /* 0x40: Stride */
    uint32_t ch_reg;            /* 0x44: 0x12b0022c */
    uint32_t in_out_ch;         /* 0x48: (out_ch << 16) | in_ch */
    uint32_t pad_pool_reg;      /* 0x4C: 0x12b00230 */
    uint32_t pad_pool;          /* 0x50: (pool << 16) | padding */
    uint32_t scale_shift_reg;   /* 0x54: 0x12b00234 */
    uint32_t scale_shift;       /* 0x58: (shift << 16) | scale */
    uint32_t reserved_reg;      /* 0x5C: 0x12b00238 */
    uint32_t reserved;          /* 0x60: 1 */
    uint32_t ctrl_reg;          /* 0x64: 0x92b00200 (start bit set) */
} aip_f_node_t;

/* AIP context */
typedef struct {
    int fd_f;               /* /dev/jzaip_f */
    int fd_p;               /* /dev/jzaip_p */
    int fd_t;               /* /dev/jzaip_t */
    int fd_mem;             /* /dev/mem */
    volatile uint32_t *io;  /* Mapped I/O registers at 0x12b00000 */
    void *chainbuf_f;       /* AIP-F chain buffer (mmap'd) */
    uint32_t chainbuf_f_paddr;
    uint32_t chainbuf_f_size;
} aip_ctx_t;

/* Initialize AIP */
int aip_init(aip_ctx_t *ctx);

/* Cleanup AIP */
void aip_cleanup(aip_ctx_t *ctx);

/* Configure and run a single convolution */
int aip_conv2d(aip_ctx_t *ctx, 
               uint32_t in_addr, uint32_t out_addr,
               uint32_t kernel_addr, uint32_t bias_addr,
               uint32_t in_w, uint32_t in_h, uint32_t in_c,
               uint32_t out_w, uint32_t out_h, uint32_t out_c,
               uint32_t kernel_w, uint32_t kernel_h,
               uint32_t stride, uint32_t pad);

/* Wait for AIP-F completion */
int aip_f_wait(aip_ctx_t *ctx);

#endif /* AIP_H */

