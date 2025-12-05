# Venus Runtime (Legacy)

Venus is the original OEM neural network runtime for the Ingenic T41 NNA. This directory contains reverse-engineered code for understanding the OEM model format.

> **Note**: Venus is being superseded by the **Mars** runtime which provides:
> - Open model format (MARS instead of proprietary .magik)
> - ONNX-based toolchain
> - Full control over quantization and layout

## Purpose

The Venus code was used to:
1. Understand the T41 NNA hardware interface
2. Decode the proprietary .magik model format
3. Study the NNA DMA command structure

## Files

| File | Description |
|------|-------------|
| `magik_model.cpp/.h` | OEM model format parser |
| `model_loader.cpp/.h` | Model loading from .magik files |
| `basenet.cpp/.h` | Network execution wrapper |
| `tensor.cpp/.h` | Tensor memory management |
| `venus_c_api.cpp` | C API for Python bindings |
| `venus_utils.cpp` | Utility functions |
| `memory_debug.cpp` | Memory debugging tools |

## OEM Model Format

The .magik format uses:
- Custom tensor layouts (NDHWC32, NMHWSOIB2)
- Pre-quantized INT8 weights
- NNA command sequences embedded in the model
- Hardware-specific memory alignment

## Status

**Deprecated** - Use Mars runtime instead:

```bash
# Instead of Venus:
# LD_LIBRARY_PATH=/opt ./venus_app model.magik input.jpg

# Use Mars:
LD_LIBRARY_PATH=/opt ./mars_detect model.mars input.jpg output.jpg
```

## Hardware Insights from Venus

Key learnings from reverse engineering:
1. NNA has 384KB on-chip ORAM at 0x12620000
2. DDR memory must be allocated via /dev/soc-nna ioctl
3. DMA descriptors use DESRAM at offset 0x778b0000
4. Weights use NMHWSOIB2 packed format (32x32 blocks)
5. Activations use NDHWC32 format (32-channel groups)

## See Also

- `src/mars/` - Active runtime development
- `mars-compiler/` - ONNX to Mars compiler
- `training/` - Model training scripts

