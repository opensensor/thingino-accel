# Venus Library Implementation Status

## 🎉🎉 MAJOR MILESTONE: Hardware Initialization Complete! 🎉🎉

**Date**: 2025-11-26

We have successfully implemented complete hardware initialization based on reverse-engineering the OEM libdrivers.so! The Venus library now:
- ✅ Properly initializes ALL hardware memory mappings
- ✅ Loads .mgk neural network models without ANY symbol resolution errors
- ✅ All base addresses are correctly mapped and accessible
- ✅ Model loads and passes all initialization tests

## Current Status

### ✅ Completed: Hardware Memory Initialization (NEW!)

Based on reverse-engineering OEM `libdrivers.so::__aie_mmap()`, we now properly initialize:

1. **L2 Cache Size Detection**
   - Maps GPIO register at 0x12200000
   - Reads L2 cache size from offset 0x60
   - Calculates ORAM base dynamically: `0x12600000 + l2cache_size`
   - **Result**: Detected 128KB L2 cache, ORAM at 0x12620000 ✅

2. **ORAM Mapping**
   - Physical address: 0x12620000
   - Size: 384KB (512KB - 128KB L2)
   - Virtual address: Mapped via `/dev/mem`
   - **Result**: `__oram_vbase = 0x77be3000` ✅

3. **NNA DMA I/O Registers**
   - Physical address: 0x12508000
   - Size: 32 bytes (0x20)
   - **Result**: `__nndma_io_vbase = 0x77f8b000` ✅

4. **NNA DMA Descriptor RAM**
   - Physical address: 0x12500000
   - Size: 32KB (0x8000)
   - **Result**: `__nndma_desram_vbase = 0x77f83000` ✅

5. **DDR DMA Memory**
   - Parses `/proc/cmdline` for `nmem=29M@0x6300000`
   - Allocates 4MB via IOCTL 0xc0046300
   - Maps to userspace via `/dev/mem`
   - **Result**: `__ddr_pbase = 0x03400000`, `__ddr_vbase = 0x777e3000` ✅

### ✅ Completed: Symbol Resolution (All 27+ symbols resolved)

The .mgk model now loads successfully. All required symbols have been implemented:

#### Global Variables & Runtime Environment (ALL PROPERLY INITIALIZED!)
- `oram_base` - ORAM base address (0x12620000) ✅
- `__oram_vbase` - ORAM virtual base address (0x77be3000) ✅
- `__ddr_pbase` - DDR physical base address (0x03400000) ✅
- `__ddr_vbase` - DDR virtual base address (0x777e3000) ✅
- `__nndma_io_vbase` - NNA DMA I/O virtual base address (0x77f8b000) ✅
- `__nndma_desram_vbase` - NNA DMA descriptor RAM virtual base address (0x77f83000) ✅
- `l2cache_size` - L2 cache size (128KB detected) ✅
- `net_mutex` - Global pthread mutex ✅
- `__assert` - Standard library assertion function ✅

#### Type Conversion Functions (venus_utils.cpp)
- `string2data_format(std::string)` - By-value overload for C++11 ABI
- `string2data_type(std::string)` - By-value overload for C++11 ABI
- `string2channel_layout(std::string)` - By-value overload for C++11 ABI
- `data_format2string(DataFormat)` - Separate enum for backward compatibility
- `data_type2string(DataType)`
- `channel_layout2string(ChannelLayout)`

#### TensorX Class (tensor.h/cpp)
- Constructor (non-inline implementation)
- `step(int dim)` - Both const and non-const versions
- `get_bytes_size()` - Returns tensor size in bytes

#### MagikLayerBase Class (magik_model.h/cpp)
- `set_inputs(std::vector<TensorXWrapper*>)`
- `set_outputs(std::vector<TensorXWrapper*>)`
- `_flush_cache(std::vector<TensorXWrapper*>)`
- `get_inputs()` - const method
- `get_outputs()` - const method
- `get_input_wrappers()` - const method (NEW!)
- `get_output_wrappers()` - const method

#### MagikModelBase Class (magik_model.h/cpp)
- Constructor with signature: `(long long, long long, void*&, void*, MemAllocMode, ModuleMode)`
  - **Critical**: Third parameter is `void*&` (reference), not `void**`
  - **Critical**: `ModuleMode` is nested enum inside `MagikModelBase`, not in `venus` namespace
- `run()`
- `reshape()`
- `pre_graph_run()`
- `free_forward_memory()`
- `free_inputs_memory()`
- `open_mnni_debug()`
- `open_mnni_profiler()`
- `set_main_pyramid_config(int level)`
- `create_and_add_pyramid_config()`
- `build_tensors(PyramidConfig*, std::vector<TensorInfo>)`
- `update_cache_buffer_ptr(std::vector<MagikLayerBase*>, void*)`
- `set_oram_address(void*, long long)` - const method
- `get_output_names()` - const method, returns std::string
- `get_input_names()` - const method, returns std::string
- `get_output(std::string&)` - const method, returns TensorXWrapper*
- `get_output(int)` - const method, returns TensorXWrapper* (NEW!)
- `get_forward_memory_size()` - const method, returns size_t

#### MagikModelBase::PyramidConfig (Nested Struct)
- `get_tensor_wrapper(std::string&)` - const method

### ✅ Current Status: ALL TESTS PASSING!

```
╔══════════════════════════════════════════════════════════╗
║  ✓ TESTS PASSED                                        ║
╚══════════════════════════════════════════════════════════╝

[1] NNA initialization
L2 cache: 128 KB, ORAM base: 0x12620000, ORAM size: 384 KB
nmem available: 29 MB at 0x06300000
NNA version: 0x00000041
Runtime initialized:
  oram_base = 0x12620000
  __oram_vbase = 0x77be3000
  __ddr_pbase = 0x3400000
  __ddr_vbase = 0x777e3000
  __nndma_io_vbase = 0x77f8b000
  __nndma_desram_vbase = 0x77f83000
NNA initialized successfully
  ✓ NNA initialized

[2] Load model
  ✓ Model loaded

[3] Model information
  ✓ Model info retrieved

[4] Input tensors
  ✓ Input tensors accessible

[5] Output tensors
  ✓ Output tensors accessible
```

**Next Steps**: Implement real functionality (TensorXWrapper, model parsing, inference execution)

## Key Technical Insights

### C++ ABI Compatibility Issues Solved
1. **String parameters**: Some functions need by-value `std::string` parameters, not const references
2. **DataFormat enum**: Must be a separate enum, not a typedef, to generate correct mangled names
3. **Nested types**: `ModuleMode` must be inside `MagikModelBase`, `PyramidConfig` must be nested
4. **Constructor parameters**: Third parameter must be `void*&` (reference to pointer)

### Build Configuration
- Compiler: `mipsel-linux-musl-g++`
- C++ Standard: `-std=c++14`
- ABI: `-D_GLIBCXX_USE_CXX11_ABI=1` (new C++11 ABI)
- Libraries: `-lpthread -lstdc++ -ldl`

## Key Implementation Details from OEM Reverse Engineering

### Hardware Initialization (from libdrivers.so::__aie_mmap)

The OEM implementation revealed the exact initialization sequence:

1. **Open devices**:
   - `/dev/mem` for physical memory mapping
   - `/dev/soc-nna` for NNA IOCTL interface

2. **L2 Cache Detection**:
   ```c
   // Map L2 cache size register
   void *l2cache_vaddr = mmap(NULL, 0x1000, PROT_READ|PROT_WRITE,
                               MAP_SHARED, memfd, 0x12200000);

   // Read GPIO register at offset 0x60
   uint32_t gpio_val = *(uint32_t*)(l2cache_vaddr + 0x60);
   uint32_t l2_bits = (gpio_val >> 10) & 0x7;

   // Decode L2 cache size
   switch (l2_bits) {
       case 1: l2cache_size = 128KB; break;
       case 2: l2cache_size = 256KB; break;
       case 3: l2cache_size = 512KB; break;
       case 4: l2cache_size = 1MB; break;
   }

   // Calculate ORAM base
   oram_base = 0x12600000 + l2cache_size;
   oram_size = 512KB - l2cache_size;
   ```

3. **Parse nmem from /proc/cmdline**:
   ```c
   // Read /proc/cmdline
   // Parse "nmem=29M@0x6300000"
   sscanf(nmem_str, "nmem=%uM@%x", &size, &addr);
   size = size << 20;  // Convert MB to bytes
   ```

4. **Map hardware regions**:
   - NNA DMA I/O: 0x12508000 (32 bytes)
   - NNA DMA DESRAM: 0x12500000 (32KB)
   - ORAM: calculated base (384KB)

5. **Allocate DDR DMA memory**:
   ```c
   struct soc_nna_buf buf;
   buf.size = 4MB;
   ioctl(nnafd, 0xc0046300, &buf);  // MALLOC
   ddr_pbase = buf.paddr;
   ddr_vbase = mmap(NULL, size, PROT_READ|PROT_WRITE,
                     MAP_SHARED, memfd, ddr_pbase);
   ```

### ✅ COMPLETED: All hardware initialization is now working!

## Next Steps to Get Inference Working

### 2. Implement TensorXWrapper
The model expects `TensorXWrapper` objects but we only have stubs. Need to:
- Create actual wrapper objects that hold tensor data
- Implement proper memory allocation for tensor buffers
- Link wrappers to NNA DMA memory

### 3. Implement Model Initialization
The `MagikModelBase` constructor needs to:
- Parse the .mgk ELF file to extract model metadata
- Allocate forward memory (1MB+ for inference workspace)
- Set up input/output tensor wrappers
- Initialize ORAM address mapping

### 4. Implement Inference Execution
The `run()` method needs to:
- Program NNA hardware registers
- Set up DMA descriptors for data transfer
- Trigger NNA execution via IOCTL commands
- Wait for completion
- Read results from output tensors

## Files Modified in This Session

### Core Implementation
- `thingino-accel/src/runtime.c` - Global variables and runtime environment
- `thingino-accel/src/venus/venus_types.h` - Type definitions
- `thingino-accel/src/venus/venus_utils.cpp` - Type conversion functions
- `thingino-accel/src/venus/tensor.h` - TensorX class declaration
- `thingino-accel/src/venus/tensor.cpp` - TensorX implementation
- `thingino-accel/src/venus/magik_model.h` - MagikModelBase/LayerBase declarations
- `thingino-accel/src/venus/magik_model.cpp` - MagikModelBase/LayerBase implementation
- `thingino-accel/Makefile` - Build configuration

## Test Results

```bash
# Build
cd thingino-accel && make clean && make

# Upload and test
sshpass -p "Ami23plop" scp -O build/lib/libvenus.so build/lib/libnna.so \
  build/bin/test_model_load root@192.168.50.117:/tmp/
sshpass -p "Ami23plop" scp -O ../ingenic-sdk-matteius/magik-nna/AEC_T41_16K_NS_OUT_UC.mgk \
  root@192.168.50.117:/tmp/
sshpass -p "Ami23plop" ssh root@192.168.50.117 \
  "LD_PRELOAD=/tmp/libvenus.so LD_LIBRARY_PATH=/tmp /tmp/test_model_load /tmp/AEC_T41_16K_NS_OUT_UC.mgk"
```

**Result**: ✅ ALL TESTS PASS! Model loads successfully, all hardware initialized, no crashes!

## Hardware Details

- **Platform**: Ingenic T41 (MIPS32 XBurst II)
- **Device**: Wyze Cam V4
- **NNA**: Hardware neural network accelerator
- **ORAM**: 384KB on-chip RAM at physical address 0x12620000
- **Kernel Driver**: `/dev/soc-nna` provides IOCTL interface

## References

- Model file: `ingenic-sdk-matteius/magik-nna/AEC_T41_16K_NS_OUT_UC.mgk`
- Reverse engineering docs: `ingenic-sdk-matteius/magik-nna/AEC_MODEL_REVERSE_ENGINEERING.md`
- Kernel driver source: `ingenic-sdk-matteius/4.4/misc/soc-nna/`
- Venus headers: `magik-toolkit/InferenceKit/nna2/mips720-glibc229/T41/include/venus.h`

