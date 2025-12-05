# T41 NNA (Neural Network Accelerator) Architecture Reference

**Status: Work in Progress - Reverse Engineered from soc-nna.ko and libdrivers.so**

## Overview

The Ingenic T41 SoC contains a dedicated Neural Network Accelerator (NNA) for efficient inference of neural networks. The NNA works in conjunction with the MXUv3 vector unit to accelerate both memory transfers and compute operations.

### Key Components

| Component | Physical Address | Size | Description |
|-----------|-----------------|------|-------------|
| ORAM | 0x12620000 (varies) | 384 KB | On-chip accelerator RAM for fast tensor storage |
| NNDMA I/O | 0x12508000 | 32 bytes | DMA control registers |
| NNDMA DESRAM | 0x12500000 | 16-32 KB | DMA descriptor RAM |
| L2 Cache Config | 0x12200060 | 4 bytes | L2 cache size configuration register |

### Memory Regions (Boot Configuration)

The T41 requires three memory regions configured via U-Boot environment variables:

```bash
# Example configuration for 128MB RAM
fw_setenv osmem 77M@0x0        # OS memory: 77MB starting at 0x0
fw_setenv rmem 22M@0x4D00000   # Reserved memory: 22MB for ISP/encoder
fw_setenv nmem 29M@0x6300000   # NNA memory: 29MB for neural network
```

**Critical**: These regions must NOT overlap! The layout is:
```
0x00000000 +-----------+
           |  osmem    |  OS/kernel memory (cacheable)
           |  (77MB)   |
0x04D00000 +-----------+
           |  rmem     |  Reserved for ISP/video encoder
           |  (22MB)   |
0x06300000 +-----------+
           |  nmem     |  NNA DMA-capable memory pool
           |  (29MB)   |
0x08000000 +-----------+  End of 128MB RAM
```

## Kernel Driver Interface

The NNA is accessed through `/dev/soc-nna` character device.

### IOCTL Commands

| Command | Code | Description |
|---------|------|-------------|
| IOCTL_SOC_NNA_MALLOC | 0xc0046300 | Allocate DMA-coherent memory from nmem |
| IOCTL_SOC_NNA_FREE | 0xc0046301 | Free DMA memory |
| IOCTL_SOC_NNA_FLUSHCACHE | 0xc0046302 | Flush CPU cache for DMA coherency |
| IOCTL_SOC_NNA_SETUP_DES | 0xc0046303 | Setup DMA descriptors |
| IOCTL_SOC_NNA_RDCH_START | 0xc0046304 | Start read DMA channel |
| IOCTL_SOC_NNA_WRCH_START | 0xc0046305 | Start write DMA channel |
| IOCTL_SOC_NNA_VERSION | 0xc0046306 | Get NNA hardware version |

### Memory Allocation Structure

```c
struct soc_nna_buf {
    void *vaddr;    /* Virtual address (output) */
    void *paddr;    /* Physical address (output) */
    int size;       /* Requested size (input) */
};
```

### Cache Flush Structure

```c
struct flush_cache_info {
    unsigned int addr;   /* Virtual address */
    unsigned int len;    /* Length in bytes */
    unsigned int dir;    /* DMA direction: 0=BIDIRECTIONAL, 1=TO_DEVICE, 2=FROM_DEVICE */
};
```

## ORAM (On-chip RAM)

ORAM provides fast on-chip storage for neural network tensors. Its base address depends on the L2 cache configuration.

### L2 Cache Size Detection

The L2 cache size is read from GPIO register at `0x12200060`, bits [12:10]:

| Value | L2 Cache Size | ORAM Base | ORAM Size |
|-------|--------------|-----------|-----------|
| 1 | 128 KB | 0x12620000 | 384 KB |
| 2 | 256 KB | 0x12640000 | 256 KB |
| 3 | 512 KB | 0x12680000 | 0 KB |
| 4 | 1 MB | 0x12700000 | N/A |

**Note**: Most T41 devices use 128KB L2 cache, giving 384KB ORAM.

### ORAM Usage

ORAM is ideal for:
- Intermediate feature maps during layer execution
- Small, frequently accessed tensors
- Weights for small layers

```c
/* ORAM allocation via nna_oram_malloc() */
void *oram_ptr = nna_oram_malloc(1024);
/* Use for fast tensor operations */
nna_oram_free(oram_ptr);
```

## NNDMA (Neural Network DMA)

The NNDMA engine transfers data between DDR and ORAM using descriptor chains.

### DMA Registers (at 0x12502000)

| Offset | Name | Description |
|--------|------|-------------|
| 0x00 | NNA_DMA_RCFG | Read channel configuration |
| 0x04 | NNA_DMA_WCFG | Write channel configuration |
| 0x08 | NNA_DMA_RCNT | Read channel count/status |
| 0x0C | NNA_DMA_WCNT | Write channel count/status |

### Descriptor Format (64-bit)

Each DMA descriptor is 64 bits with the following fields:

```
Bits 50-51: Flag (descriptor type)
  0 = CNT  (Count descriptor - first in chain)
  1 = LINK (Continue to next descriptor)
  2 = END  (Last descriptor in chain)

For CNT descriptor:
  Bits 0-19: Total bytes in chain (max 1MB)

For LINK/END descriptor:
  Bits 40-49: Data length / 64 - 1 (10 bits, max 64KB per descriptor)
  Bits 26-39: ORAM address / 64 (14 bits)
  Bits 0-25:  DDR address / 64 (26 bits)
```

### Building Descriptors

```c
/* Count descriptor (first in chain) */
uint64_t nna_des_cnt(uint32_t total_bytes) {
    return (DES_CFG_CNT << 50) | (total_bytes & 0xFFFFF);
}

/* Transfer descriptor */
uint64_t nna_des_transfer(uint32_t ddr_addr, uint32_t oram_addr,
                          uint32_t len, int is_last) {
    uint64_t flag = is_last ? DES_CFG_END : DES_CFG_LINK;
    return (flag << 50)
         | (((len >> 6) - 1) << 40)
         | ((oram_addr >> 6) << 26)
         | (ddr_addr >> 6);
}
```

### Alignment Requirements

- All DMA addresses must be **64-byte aligned**
- Maximum transfer per descriptor: **64 KB**
- Maximum chain total: **1 MB**

## Memory Allocation Strategy

### DDR Allocation via nna_malloc()

The `nna_malloc()` function allocates DMA-coherent memory from the nmem region:

```c
void *nna_malloc(size_t size);
void *nna_malloc_phys(size_t size, void **paddr_out);
void nna_free(void *ptr);
```

**How it works:**
1. Calls `ioctl(fd, IOCTL_SOC_NNA_MALLOC, &buf)` to allocate physical memory
2. Kernel allocates from nmem pool using `dma_alloc_coherent()`
3. Returns both virtual and physical addresses
4. Memory is automatically cache-coherent (uncached or write-through)

**Limitations:**
- Each allocation is contiguous
- Kernel may fail for large allocations (>8MB) due to fragmentation
- Limited by nmem pool size (typically 29MB)
- Many small allocations can fragment the pool

### Best Practices

1. **Allocate large buffers first** - Claim contiguous blocks before fragmentation
2. **Minimize allocation count** - Pool small tensors into larger buffers
3. **Use working buffer reuse** - Mars runtime uses ping-pong buffers for intermediates
4. **Output tensors can use regular malloc** - They only need CPU access after inference

## NNA Hardware Version

The version IOCTL returns hardware identification:

```c
struct nna_version_info {
    uint32_t version_buf;        /* Hardware version (0x41 = T41) */
    uint32_t nmem_extension_buf; /* Extension flags */
    uint32_t nmem_paddr;         /* nmem physical address from cmdline */
    uint32_t nmem_size;          /* nmem size in bytes */
};
```

Version `0x00000041` indicates T41 NNA.

## Integration with MXUv3

The NNA and MXUv3 work together for neural network inference:

| Component | Role |
|-----------|------|
| **NNDMA** | Moves data between DDR ↔ ORAM |
| **ORAM** | Fast on-chip buffer for active tensors |
| **MXUv3 VPR** | 32×512-bit vector registers for computation |
| **MXUv3 VSR** | 4 sum registers for MAC accumulation |

### Typical Inference Flow

```
1. Load weights to DDR (via nna_malloc)
2. Load input tensor to DDR
3. For each layer:
   a. NNDMA: DDR → ORAM (load inputs/weights)
   b. MXUv3: Compute convolution/activation in VPRs
   c. NNDMA: ORAM → DDR (store outputs)
4. Read output tensor from DDR
```

## Runtime Global Variables

The runtime exports these globals for legacy .mgk model compatibility:

```c
void *oram_base;           /* ORAM physical base address */
void *__oram_vbase;        /* ORAM virtual address (mmap'd) */
void *__ddr_pbase;         /* DDR physical base for weights */
void *__ddr_vbase;         /* DDR virtual address for weights */
void *__nndma_io_vbase;    /* NNDMA I/O registers virtual address */
void *__nndma_desram_vbase;/* NNDMA descriptor RAM virtual address */
void *__nndma_fastio_vbase;/* NNDMA fast I/O registers */
```

## Error Handling

Common issues and solutions:

| Error | Cause | Solution |
|-------|-------|----------|
| "dma_alloc_coherent failed" | Memory fragmentation | Reduce allocations or reboot |
| "page allocation failure: order:N" | No contiguous N-page block | Allocate smaller chunks |
| "IOCTL_SOC_NNA_MALLOC failed" | nmem exhausted | Free unused buffers |
| Kernel panic on access | Memory region overlap | Check fw_setenv settings |
| Partial/wrong results | Cache coherency issue | Use nna_cache_flush() |

## Performance Considerations

1. **ORAM vs DDR latency**: ORAM is ~10x faster for small tensors
2. **DMA transfer overhead**: Minimize DDR↔ORAM transfers
3. **Alignment**: Misaligned access causes performance penalty
4. **Cache effects**: DDR buffers may need explicit cache management
5. **Contiguous allocation**: Large contiguous blocks are precious

## Related Documentation

- [MXUv3 Instructions](mxuv3_instructions.md) - Vector unit ISA
- [MXUv3 Organic vs Later Discoveries](mxuv3_organic_vs_later_discoveries.md) - RE notes
- [Mars Model Format](../mars-compiler/README.md) - Custom model format

## References

- soc-nna kernel driver source (`soc-nna/`)
- Reverse-engineered from libdrivers.so and OEM Venus runtime

