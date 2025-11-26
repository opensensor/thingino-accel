# Prompt for New Agent Context

## Task

Continue implementing the Venus library for the thingino-accel project to enable neural network inference on the Ingenic T41 NNA (Neural Network Accelerator) hardware.

## Current Status

✅ **MAJOR MILESTONE ACHIEVED**: We have successfully implemented a Venus library from scratch that can load .mgk neural network models without any symbol resolution errors!

The model (AEC_T41_16K_NS_OUT_UC.mgk) now loads successfully via dlopen, but crashes with a segmentation fault when trying to execute because all implementations are currently stubs that return null pointers.

## What Has Been Done

1. ✅ Implemented all 26+ required Venus API symbols
2. ✅ Resolved all C++ ABI compatibility issues
3. ✅ Model loads successfully without dlopen errors
4. ✅ Created comprehensive documentation

## What Needs to Be Done

Implement real functionality to get actual inference execution working:

1. **Initialize hardware memory mappings** - Map ORAM, NNA DMA registers, and DDR base addresses
2. **Implement TensorXWrapper** - Create actual tensor wrapper objects with proper memory management
3. **Implement model initialization** - Parse model metadata and set up tensors
4. **Implement inference execution** - Program NNA hardware and execute inference

## Key Files to Review

1. **Status Document**: `thingino-accel/VENUS_IMPLEMENTATION_STATUS.md` - Complete list of what's been implemented
2. **Next Steps**: `thingino-accel/NEXT_STEPS.md` - Detailed task breakdown with code examples
3. **Current Implementation**:
   - `thingino-accel/src/runtime.c` - Global variables (currently NULL)
   - `thingino-accel/src/venus/magik_model.cpp` - Model class (stub implementations)
   - `thingino-accel/src/venus/tensor.cpp` - Tensor class (partial implementation)
   - `thingino-accel/src/device.c` - NNA device initialization

## Test Command

```bash
# Build
cd thingino-accel && make clean && make

# Upload to camera (Wyze Cam V4 at 192.168.50.117)
sshpass -p "Ami23plop" scp -O build/lib/libvenus.so build/lib/libnna.so \
  build/bin/test_model_load root@192.168.50.117:/tmp/

# Upload model file (if not already on camera)
sshpass -p "Ami23plop" scp -O ../ingenic-sdk-matteius/magik-nna/AEC_T41_16K_NS_OUT_UC.mgk \
  root@192.168.50.117:/tmp/

# Test
sshpass -p "Ami23plop" ssh root@192.168.50.117 \
  "LD_PRELOAD=/tmp/libvenus.so LD_LIBRARY_PATH=/tmp /tmp/test_model_load /tmp/AEC_T41_16K_NS_OUT_UC.mgk"
```

**Current Result**: Model loads successfully, then segfaults (expected with stub implementations)

## Hardware Details

- **Platform**: Ingenic T41 (MIPS32 XBurst II)
- **Device**: Wyze Cam V4
- **IP Address**: 192.168.50.117
- **Password**: Ami23plop
- **NNA**: Hardware neural network accelerator with 384KB ORAM
- **ORAM Physical Address**: 0x12620000
- **Kernel Driver**: `/dev/soc-nna`

## Important Technical Details

### C++ ABI Compatibility (Already Solved)
- Using new C++11 ABI: `-D_GLIBCXX_USE_CXX11_ABI=1`
- Some functions need by-value `std::string` parameters (not const references)
- `ModuleMode` must be nested inside `MagikModelBase`
- `DataFormat` must be a separate enum (not typedef)

### Memory Architecture
- ORAM: 384KB on-chip RAM at 0x12620000 (managed by kernel)
- nmem: 29MB DMA-capable memory region
- All DMA memory must be allocated via `nna_malloc()` which uses IOCTL to kernel driver
- Memory mapping: IOCTL returns physical address, then mmap via `/dev/mem`

### IOCTL Commands (from kernel driver)
- `0xc0046300` - MALLOC (allocate DMA memory)
- `0xc0046301` - FREE (free DMA memory)
- `0xc0046302` - FLUSHCACHE (cache coherency)
- `0xc0046303` - SETUP_DES (DMA descriptor setup)
- `0xc0046304` - RDCH_START (read channel start)
- `0xc0046305` - WRCH_START (write channel start)
- `0xc0046306` - VERSION (get NNA version)

## References

- Kernel driver source: `ingenic-sdk-matteius/4.4/misc/soc-nna/`
- Test examples: `ingenic-sdk-matteius/magik-nna/nna_test.c`
- Model file: `ingenic-sdk-matteius/magik-nna/AEC_T41_16K_NS_OUT_UC.mgk`
- Venus headers: `magik-toolkit/InferenceKit/nna2/mips720-glibc229/T41/include/venus.h`
- Reverse engineering docs: `ingenic-sdk-matteius/magik-nna/AEC_MODEL_REVERSE_ENGINEERING.md`

## Goal

Get the AEC model to successfully execute inference on the NNA hardware and return valid output tensor data.

## Suggested Starting Point

Start with Phase 1 from `NEXT_STEPS.md`: Initialize hardware memory mappings. This is the foundation for everything else.

Focus on:
1. Mapping ORAM to virtual address space in `nna_init()`
2. Setting up `__oram_vbase`, `__ddr_vbase`, `__ddr_pbase`
3. Mapping NNA DMA registers

Then move to implementing TensorXWrapper and model initialization.

## Success Criteria

The test should:
1. ✅ Load the model successfully (already working)
2. ✅ Initialize all hardware mappings (next step)
3. ✅ Allocate tensors and forward memory
4. ✅ Execute inference without crashing
5. ✅ Return valid output tensor data

Good luck! This is exciting work - we're very close to having a fully functional open-source NNA library! 🚀

