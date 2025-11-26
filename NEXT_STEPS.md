# Next Steps: Implementing Real Inference Execution

## Context

We have successfully implemented a Venus library that can load .mgk models without symbol errors. The model now crashes when trying to execute because all implementations are stubs. This document outlines the steps to implement real inference execution.

## Goal

Get the AEC_T41_16K_NS_OUT_UC.mgk model to successfully execute inference on the Ingenic T41 NNA hardware accelerator.

## Phase 1: Initialize Hardware Memory Mappings

### Task 1.1: Map ORAM to Virtual Address Space
**File**: `thingino-accel/src/device.c` (in `nna_init()`)

Currently `__oram_vbase` is NULL. Need to:
```c
// After opening /dev/soc-nna, map ORAM physical address
int mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
void *oram_virt = mmap(NULL, ORAM_SIZE, PROT_READ | PROT_WRITE, 
                       MAP_SHARED, mem_fd, ORAM_PHYS_ADDR);
__oram_vbase = oram_virt;
```

**Reference**: See how we map DMA memory in `nna_malloc()` in `src/memory.c`

### Task 1.2: Map NNA DMA Registers
**File**: `thingino-accel/src/device.c`

Need to map:
- `__nndma_io_vbase` - NNA DMA I/O registers (check kernel driver for physical address)
- `__nndma_desram_vbase` - NNA DMA descriptor RAM

**Reference**: `ingenic-sdk-matteius/4.4/misc/soc-nna/soc_nna_main.c` for register addresses

### Task 1.3: Initialize DDR Base Addresses
**File**: `thingino-accel/src/runtime.c`

The `__ddr_vbase` and `__ddr_pbase` should point to the DMA memory region allocated by the kernel driver.

**Approach**: 
- Allocate a large DMA buffer (e.g., 16MB) during `nna_init()`
- Set `__ddr_pbase` to the physical address returned by kernel
- Set `__ddr_vbase` to the mmap'd virtual address

## Phase 2: Implement TensorXWrapper

### Task 2.1: Define TensorXWrapper Structure
**File**: `thingino-accel/src/venus/tensor.h`

Currently it's just a forward declaration. Need to implement:
```cpp
class TensorXWrapper {
public:
    TensorX *tensor;
    void *data;
    size_t size;
    bool owns_memory;
    
    TensorXWrapper(TensorX *t);
    ~TensorXWrapper();
    
    void allocate_memory();
    void free_memory();
};
```

### Task 2.2: Implement TensorXWrapper Methods
**File**: `thingino-accel/src/venus/tensor.cpp`

Implement constructor, destructor, and memory management methods.

**Critical**: Memory should be allocated using `nna_malloc()` to get DMA-capable memory.

## Phase 3: Implement Model Initialization

### Task 3.1: Parse Model Metadata
**File**: `thingino-accel/src/venus/magik_model.cpp` (in constructor)

The `MagikModelBase` constructor receives parameters that likely contain:
- `param1`, `param2` - Model configuration (dimensions, layer count, etc.)
- `param3` (void*&) - Reference to a pointer, likely for returning allocated memory
- `param4` - Additional configuration

**Approach**:
1. Study the .mgk ELF file structure to understand what metadata is available
2. Parse ELF sections to extract input/output tensor shapes
3. Create TensorXWrapper objects for inputs and outputs

**Reference**: `ingenic-sdk-matteius/magik-nna/parse_mgk_model.py` shows ELF structure

### Task 3.2: Implement get_input_names() and get_output_names()
**File**: `thingino-accel/src/venus/magik_model.cpp`

Currently returns empty strings. Should return comma-separated tensor names.

**Approach**: Parse tensor names from .mgk ELF symbols or metadata sections.

### Task 3.3: Implement get_output() Method
**File**: `thingino-accel/src/venus/magik_model.cpp`

Currently returns nullptr. Should:
1. Maintain a map of tensor name -> TensorXWrapper*
2. Look up and return the wrapper for the requested name

### Task 3.4: Allocate Forward Memory
**File**: `thingino-accel/src/venus/magik_model.cpp`

The model needs workspace memory for intermediate computations.

```cpp
size_t MagikModelBase::get_forward_memory_size() const {
    // Parse from model metadata or use default
    return forward_memory_size;
}
```

Allocate this memory in the constructor using `nna_malloc()`.

## Phase 4: Implement Inference Execution

### Task 4.1: Implement run() Method
**File**: `thingino-accel/src/venus/magik_model.cpp`

This is the main inference execution method. Steps:
1. Validate input tensors are set
2. Set up NNA DMA descriptors
3. Program NNA hardware registers
4. Trigger execution via IOCTL
5. Wait for completion
6. Return results

**Reference**: 
- `ingenic-sdk-matteius/magik-nna/nna_test.c` for example usage
- `ingenic-sdk-matteius/4.4/misc/soc-nna/soc_nna.h` for IOCTL commands

### Task 4.2: Implement DMA Descriptor Setup
**File**: `thingino-accel/src/venus/magik_model.cpp` or new file

The NNA uses DMA descriptors to transfer data. Need to:
1. Create descriptor structures
2. Set up read channel (input data)
3. Set up write channel (output data)
4. Use `IOCTL_SETUP_DES` to configure kernel driver

**IOCTL Command**: `0xc0046303` (SETUP_DES)

### Task 4.3: Trigger NNA Execution
**File**: `thingino-accel/src/venus/magik_model.cpp`

Use IOCTL commands to start NNA:
- `IOCTL_RDCH_START` (0xc0046304) - Start read channel
- `IOCTL_WRCH_START` (0xc0046305) - Start write channel

Wait for completion (polling or interrupt-based).

## Phase 5: Testing and Debugging

### Task 5.1: Create Simple Test
**File**: `thingino-accel/examples/test_inference.c`

Create a test that:
1. Loads the AEC model
2. Sets dummy input data (zeros or random)
3. Runs inference
4. Prints output tensor values

### Task 5.2: Add Debug Logging
Add verbose logging to track:
- Memory allocations
- DMA descriptor setup
- IOCTL calls and return values
- NNA register states

### Task 5.3: Validate Against OEM Library
If possible, compare results with the original OEM library to verify correctness.

## Quick Start Commands

```bash
# Build
cd thingino-accel && make clean && make

# Upload to camera
sshpass -p "Ami23plop" scp -O build/lib/libvenus.so build/lib/libnna.so \
  build/bin/test_model_load root@192.168.50.117:/tmp/

# Test
sshpass -p "Ami23plop" ssh root@192.168.50.117 \
  "LD_PRELOAD=/tmp/libvenus.so LD_LIBRARY_PATH=/tmp /tmp/test_model_load /tmp/AEC_T41_16K_NS_OUT_UC.mgk"
```

## Important Notes

1. **Memory Safety**: All DMA memory must be allocated via `nna_malloc()` to ensure physical contiguity
2. **Cache Coherency**: Use `IOCTL_FLUSHCACHE` before/after DMA operations
3. **Error Handling**: Check all IOCTL return values and handle errors gracefully
4. **Hardware State**: Ensure NNA is properly initialized before use

## Expected Outcome

After completing these tasks, the test should:
1. ✅ Load the model successfully (already working)
2. ✅ Initialize all hardware mappings
3. ✅ Allocate tensors and forward memory
4. ✅ Execute inference without crashing
5. ✅ Return valid output tensor data

## Resources

- Kernel driver: `ingenic-sdk-matteius/4.4/misc/soc-nna/`
- Test examples: `ingenic-sdk-matteius/magik-nna/nna_test.c`
- Venus headers: `magik-toolkit/InferenceKit/nna2/mips720-glibc229/T41/include/`
- Current status: `thingino-accel/VENUS_IMPLEMENTATION_STATUS.md`

