# Mars Runtime

Mars is an open-source neural network runtime for the **Ingenic T41 SoC** found in many IP cameras. It provides hardware-accelerated inference using the T41's MXUv3 SIMD unit and NNA memory subsystem.

**Key Achievement**: 20x speedup on object detection (35s → 1.75s) through MXUv3 vectorization and ORAM optimization.

## Why Mars?

The proprietary Venus SDK from Ingenic:
- Only works with glibc toolchains (not musl/uClibc)
- Requires closed-source `.magik` model format
- No source code available for debugging or optimization

Mars provides a fully open-source alternative that works with any toolchain.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Mars Runtime                             │
├─────────────────────────────────────────────────────────────────┤
│  mars_detect.c    - Object detection (person/vehicle/cat/dog)   │
│  mars_runtime.c   - Model loading, layer execution engine       │
│  mars_nn_hw.c     - ORAM/DDR memory, MXU vector operations      │
│  mxu_conv.c       - MXU-accelerated Conv2D (16 floats/cycle)    │
│  mxu_ops.c        - MXU MaxPool, ReLU, Upsample, Concat         │
│  mars_math.c      - SIMD math utilities                         │
├─────────────────────────────────────────────────────────────────┤
│  stb_image.h      - Image loading (JPEG/PNG)                    │
│  stb_image_write.h - Image writing (JPEG)                       │
└─────────────────────────────────────────────────────────────────┘
```

## Key Files

| File | Description |
|------|-------------|
| `mars_runtime.c` | Core model loading, buffer allocation, layer dispatch |
| `mars_nn_hw.c` | ORAM/DDR memory management, MXU infrastructure |
| `mxu_conv.c` | MXU-accelerated Conv2D (float32/int8, NHWC format) |
| `mxu_ops.c` | MaxPool, Upsample, Concat, activation functions |
| `mars_detect.c` | Object detection application using TinyDet model |

## Hardware Acceleration

### MXUv3 (Media Extension Unit v3)

The T41's MXUv3 is a 512-bit SIMD unit with:
- **32 VPR Registers**: 512-bit each (16 floats or 64 int8)
- **4 VSR Sum Registers**: Accumulators for MAC operations
- **Key Instructions**:
  - `LA0_VPR(reg, addr)` - Load 64 bytes to VPR
  - `SA0_VPR(reg, addr)` - Store 64 bytes from VPR
  - `VPR_MUL(dst, src)` - Elementwise multiply
  - `VPR_ADD(dst, src)` - Elementwise add

### Memory Hierarchy

| Memory | Size | Speed | Usage |
|--------|------|-------|-------|
| **ORAM** | 640KB | 314 MB/s read | Weight staging, hot tensors |
| **DDR** | 29MB pool | 41 MB/s read | Large tensors, all weights |

ORAM provides **7.6x faster reads** and **20x faster writes** than DDR.

## Tensor Formats

Mars uses **NHWC** (channels-last) format:
- **Input**: `[batch, height, width, channels]`
- **Weights**: `[out_ch, kH, kW, in_ch]` (OHWI)
- **Output**: `[batch, height, width, out_channels]`

NHWC enables efficient 64-byte aligned loads (16 channels at once).

## Building

```bash
# Cross-compile for T41 (MIPS)
make mars_detect

# Copy to device via NFS
cp build/bin/mars_detect /srv/thingino/
```

## Usage

```bash
# On T41 device
cd /opt
LD_LIBRARY_PATH=/opt ./mars_detect model.mars input.jpg output.jpg

# With confidence threshold
LD_LIBRARY_PATH=/opt ./mars_detect -t 0.25 model.mars image.jpg result.jpg
```

## Detection Output

TinyDet output grid is `[1, 9, H/16, W/16]`:
- Channel 0: Objectness score (sigmoid)
- Channels 1-4: Box offsets (x, y, w, h)
- Channels 5-8: Class scores (person, vehicle, cat, dog)

## Performance

**TinyDet 4-class on T41 @ 1.5GHz:**

| Optimization | Time | Speedup |
|--------------|------|---------|
| Scalar baseline | 35s | 1x |
| + MXU vectorization | 4s | 9x |
| + ORAM weight staging | 1.75s | 20x |

Conv2D accounts for ~90% of inference time; MXU acceleration is critical.

## Model Pipeline

```
PyTorch Model  -->  ONNX Export  -->  Mars Compiler  -->  .mars file
    |                   |                  |                  |
 train.py          export_onnx.py     mars-compiler/      mars_runtime
```

See `mars-compiler/` for the Rust-based ONNX-to-Mars compiler.

