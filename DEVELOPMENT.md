# thingino-accel Development Guide

## Overview

This library is being developed through reverse engineering of Ingenic's proprietary NNA libraries combined with official SDK documentation.

## Development Resources

### 1. Binary Ninja Analysis (Smart-Diff MCP)

We have three OEM binaries loaded in Binary Ninja for reverse engineering:

- **libvenus.so** (port_9013) - Main NNA inference library
- **libdrivers.so** (port_9014) - Low-level device drivers
- **libaip_debug.so** (port_9015) - AIP (AI Processing) library

Key functions analyzed:
- `magik_venus_init` - NNA initialization
- `__aie_mmap` - Memory mapping for ORAM/DDR
- `__aie_lock` - Multi-process locking
- `ddr_memory_init` - DDR memory allocator
- `__aie_get_oram_info` - ORAM information retrieval

### 2. Magik Toolkit

Located at `magik-toolkit/`, contains:
- **InferenceKit**: Official headers and libraries
  - `venus.h` - Main API header
  - `tensor.h` - Tensor operations
  - `type.h` - Type definitions
- **Documentation**: PDF manuals
  - Venus Programming Manual
  - Training Quantization Guide
  - Post-Training Quantization Guide
- **TransformKit**: Model conversion tools
- **TrainingKit**: PyTorch training plugins

### 3. Reverse Engineering Notes

Located at `ingenic-sdk-matteius/magik-nna/`:
- `MAGIK_NNA_ANALYSIS.md` - Overall NNA analysis
- `AEC_MODEL_REVERSE_ENGINEERING.md` - Model format analysis
- `LIBMERT_REVERSE_ENGINEERING.md` - Library internals
- `musl_issue.txt` - Notes on glibc/musl compatibility issues

## Current Implementation Status

### ✅ Completed

1. **Device Interface** (`src/device.c`)
   - `/dev/soc-nna` device opening
   - ORAM memory mapping
   - Hardware info retrieval
   - Basic initialization/deinitialization

2. **Memory Management** (`src/memory.c`)
   - DDR allocation (aligned to 64 bytes)
   - ORAM bump allocator
   - Memory statistics

3. **Tensor API** (`src/tensor.c`)
   - Tensor creation/destruction
   - Shape management
   - Data type handling
   - Basic reshape operations

4. **Build System**
   - Makefile for cross-compilation
   - musl libc compatibility
   - Static and shared library builds

5. **Test Program** (`examples/test_init.c`)
   - Initialization test
   - Memory allocation test
   - Tensor operations test

### 🚧 TODO

1. **Model Loading**
   - Parse `.mgk` ELF format
   - Load model weights
   - Initialize model structure

2. **Inference Engine**
   - Layer execution
   - Operator implementations
   - Forward pass

3. **Hardware Programming**
   - NNA register configuration
   - DMA setup
   - Interrupt handling

4. **Advanced Memory**
   - Proper ORAM allocator (buddy/free-list)
   - Cache coherency (MIPS cache ops)
   - Memory pooling

5. **Multi-Process Support**
   - `/dev/nna_lock` integration
   - Process-safe resource management

## Key Insights from Reverse Engineering

### Memory Layout (T41)

```
0x12200000 - L2 Cache Size Register
0x12500000 - NNDMA Descriptor RAM (16 KB)
0x12508000 - NNDMA IO Registers (32 bytes)
0x12600000 - ORAM Base (varies by L2 cache config)
             - 256 KB (L2=256KB)
             - 384 KB (L2=128KB) <- Wyze Cam V4
             - 512 KB (L2=0KB)
```

### IOCTL Commands

From `libdrivers.so` analysis:

```c
#define NNA_IOC_ALLOC_MEM    0xc0046300  // Allocate DMA memory
#define NNA_IOC_FREE_MEM     0xc0046301  // Free DMA memory
#define NNA_IOC_GET_ORAM     0xc0046303  // Get ORAM info
#define NNA_IOC_GET_VERSION  0xc0046306  // Get driver version
#define NNA_IOC_LOCK         0xc0046308  // Lock NNA
#define NNA_IOC_UNLOCK       0xc0046309  // Unlock NNA
```

### Model Format (.mgk)

Models are ELF shared objects containing:
- `.rodata` - Model weights (quantized INT8)
- `.data.rel.ro` - Layer descriptors
- Code sections - Operator implementations
- Symbol table - Layer names

Example: `AEC_T41_16K_NS_OUT_UC.mgk` (635 KB)
- 27 layers (Conv, GRU, BatchNorm)
- INT8 quantization
- Streaming audio processing

## Building

```bash
# Set toolchain
export PATH="/path/to/thingino/toolchain/bin:$PATH"
export CROSS_COMPILE=mipsel-linux-

# Build
./build.sh

# Or manually
make clean
make lib
make examples
```

## Testing on Hardware

```bash
# Deploy
scp build/lib/libnna.so root@192.168.x.x:/usr/lib/
scp build/bin/test_init root@192.168.x.x:/tmp/

# Run
ssh root@192.168.x.x
insmod /lib/modules/soc-nna.ko
/tmp/test_init
```

## Next Development Steps

1. **Analyze more OEM functions** using Binary Ninja
   - Model loading functions
   - Layer execution functions
   - Operator implementations

2. **Implement model parser**
   - ELF parsing
   - Weight extraction
   - Layer graph construction

3. **Add basic operators**
   - Conv2D
   - MaxPool
   - Activation functions

4. **Test with simple model**
   - Create minimal ONNX model
   - Convert to .mgk using magik-transform-tools
   - Run inference

## References

- Ingenic Magik Venus Programming Manual (PDF)
- OEM binaries in Binary Ninja (ports 9013-9015)
- Reverse engineering notes in `ingenic-sdk-matteius/magik-nna/`

