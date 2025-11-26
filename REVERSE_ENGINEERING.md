# Reverse Engineering Guide

## Binary Ninja Smart-Diff MCP Tools

The OEM binaries are loaded in Binary Ninja and accessible via MCP tools.

### Available Binaries

```
port_9013: libvenus.so     - Main NNA inference library (398 KB)
port_9014: libdrivers.so   - Device drivers (25 KB)
port_9015: libaip_debug.so - AIP library (34 KB)
```

### Useful Commands

#### List functions in a binary
```
list_binary_functions_smart-diff(binary_id="port_9013")
```

#### Search for specific functions
```
list_binary_functions_smart-diff(binary_id="port_9013", search="venus_init")
list_binary_functions_smart-diff(binary_id="port_9013", search="load_model")
list_binary_functions_smart-diff(binary_id="port_9014", search="oram")
```

#### Decompile a function
```
decompile_binary_function_smart-diff(
    binary_id="port_9013",
    function_name="magik_venus_init"
)
```

## Key Functions to Analyze

### libvenus.so (port_9013)

**Initialization:**
- `magik_venus_init` - Initialize NNA hardware
- `magik_venus_deinit` - Cleanup
- `magik_venus_check` - Check NNA status

**Model Management:**
- `magik::venus::BaseNet::load_model` - Load .mgk model
- `magik::venus::BaseNet::run` - Execute inference
- `magik::venus::BaseNet::get_input` - Get input tensor
- `magik::venus::BaseNet::get_output` - Get output tensor

**Tensor Operations:**
- `magik::venus::Tensor::set_data` - Set tensor data
- `magik::venus::Tensor::get_bytes_size` - Get size
- `magik::venus::Tensor::reshape` - Reshape tensor

### libdrivers.so (port_9014)

**Memory Management:**
- `ddr_memory_init` - Initialize DDR allocator
- `ddr_malloc` - Allocate DDR memory
- `ddr_free` - Free DDR memory
- `ddr_memalign` - Aligned allocation

**Device Interface:**
- `__aie_mmap` - Map ORAM/DDR/registers
- `__aie_munmap` - Unmap memory
- `__aie_lock` - Lock NNA for exclusive access
- `__aie_unlock` - Unlock NNA
- `__aie_get_oram_info` - Get ORAM information
- `__aie_flushcache` - Flush CPU cache

**ORAM Management:**
- Functions use global variables:
  - `oram_base` - ORAM physical address
  - `oram_real_size` - ORAM size
  - `__oram_vbase` - ORAM virtual address

### libaip_debug.so (port_9015)

**AIP (AI Processing):**
- Higher-level AI operations
- Pre/post-processing
- Model-specific helpers

## Decompiled Code Examples

### Example 1: ORAM Info Retrieval

```c
// From __aie_get_oram_info
void __aie_get_oram_info(uint32_t *info) {
    info[0] = *__oram_vbase;      // Virtual address
    info[1] = *oram_base;         // Physical address
    info[2] = *oram_real_size;    // Size in bytes
}
```

### Example 2: DDR Memory Init

```c
// From ddr_memory_init
int ddr_memory_init(void *base, size_t size) {
    pthread_mutex_lock(&mem_mutex);
    
    mp_memory = base;
    mp_memory_size = size;
    mp_memory_used = 0;
    
    heap = malloc(0x34);
    Local_HeapInit(base, size, 0x14ca8);
    *heap = list;
    
    pthread_mutex_unlock(&mem_mutex);
    return 0;
}
```

### Example 3: NNA Lock

```c
// From __aie_lock
int __aie_lock(void) {
    if (__nnalockfd < 0) {
        printf("Error: __nnalockfd = %d is wrong\n", __nnalockfd);
        return -1;
    }
    
    if (ioctl(__nnalockfd, 0xc0046308, 0) >= 0) {
        return 0;
    }
    
    printf("Error: nna_lock failed: %s\n", strerror(errno));
    return -1;
}
```

## Memory Mapping Process

From `__aie_mmap` analysis:

1. **Open devices:**
   - `/dev/mem` - Physical memory access
   - `/dev/soc-nna` - NNA device
   - `/dev/nna_lock` - Multi-process lock (optional)

2. **Read L2 cache configuration:**
   - Map 0x12200000 (CPM registers)
   - Read L2 cache size from offset 0x60
   - Calculate ORAM base and size

3. **Map memory regions:**
   - NNDMA IO: 0x12508000 (32 bytes)
   - NNDMA DESC: 0x12500000 (16 KB)
   - ORAM: 0x12600000 + L2_offset (256-512 KB)
   - DDR: Allocated via ioctl

4. **Initialize allocators:**
   - DDR: Heap allocator
   - ORAM: Bump allocator

## IOCTL Commands

```c
// Memory allocation
struct nna_mem_req {
    unsigned long size;
    unsigned long phys_addr;
};
ioctl(fd, 0xc0046300, &req);  // Allocate
ioctl(fd, 0xc0046301, paddr); // Free

// ORAM info
struct nna_oram_info {
    unsigned long virt_addr;
    unsigned long phys_addr;
    unsigned long size;
};
ioctl(fd, 0xc0046303, &info);

// Version
uint32_t version;
ioctl(fd, 0xc0046306, &version);

// Locking
ioctl(fd, 0xc0046308, 0);  // Lock
ioctl(fd, 0xc0046309, 0);  // Unlock
```

## Next Functions to Analyze

Priority list for reverse engineering:

1. **Model Loading:**
   - `magik::venus::BaseNet::load_model`
   - Model parsing functions
   - Weight loading

2. **Inference Execution:**
   - `magik::venus::BaseNet::run`
   - Layer execution loop
   - Operator dispatch

3. **Operators:**
   - Conv2D implementation
   - Pooling operations
   - Activation functions
   - GRU/LSTM layers

4. **Quantization:**
   - INT8 quantization/dequantization
   - Scale/zero-point handling

## Tips for Analysis

1. **Start with simple functions** - Device init, memory allocation
2. **Look for global variables** - Often contain hardware addresses
3. **Follow the call graph** - Understand function relationships
4. **Compare with SDK headers** - Validate findings
5. **Test on hardware** - Verify assumptions

## Tools

- **Binary Ninja** - Decompilation and analysis
- **readelf** - ELF structure inspection
- **objdump** - Disassembly
- **strings** - String extraction
- **patchelf** - Binary patching (for musl conversion)

