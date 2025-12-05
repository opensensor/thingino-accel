# Mars Runtime

Mars is a custom neural network runtime for the Ingenic T41 NNA (Neural Network Accelerator). It executes `.mars` model files compiled from ONNX models.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Mars Runtime                           │
├─────────────────────────────────────────────────────────────┤
│  mars_detect.c    - YOLO/TinyDet detection application      │
│  mars_runtime.c   - Core model loading & execution          │
│  mars_nn_hw.c     - NNA hardware abstraction                │
│  mars_nn_cmd.c    - NNA command generation                  │
│  mxu_conv.c       - MXU convolution kernels (INT8/NHWC)     │
│  mxu_ops.c        - MXU operations (ReLU, pooling, etc.)    │
│  mars_math.c      - SIMD math utilities                     │
├─────────────────────────────────────────────────────────────┤
│  stb_image.h      - Image loading (JPEG/PNG/PPM)            │
│  stb_image_write.h - Image writing (JPEG)                   │
└─────────────────────────────────────────────────────────────┘
```

## Key Files

| File | Description |
|------|-------------|
| `mars_runtime.c` | Model loading, tensor management, layer execution |
| `mars_nn_hw.c` | NNA hardware init, DDR/ORAM memory management |
| `mars_nn_cmd.c` | NNA DMA command generation and execution |
| `mxu_conv.c` | INT8 convolution with MXU SIMD (NHWC format) |
| `mxu_ops.c` | MaxPool, ReLU, Upsample, Concat operations |
| `mars_detect.c` | Object detection application (TinyDet/YOLO) |

## Memory Layout

- **DDR**: Weights and large tensors (8MB allocated via /dev/soc-nna)
- **ORAM**: On-chip 384KB for intermediate activations
- **nmem**: 29MB reserved for additional tensor allocation

## Tensor Formats

Mars uses **NHWC** (channels-last) format for all tensors:
- Input: `[1, Height, Width, Channels]`
- Weights: `[OutCh, Height, Width, InCh]`
- Output: `[1, Height, Width, OutCh]`

## INT8 Quantization

All operations use INT8 with per-tensor scale factors:
```c
output_int8 = clamp((input_int8 * weight_int8 * combined_scale), -128, 127)
```

## Building

```bash
# Cross-compile for T41 (MIPS)
make mars_detect

# Deploy to device
cp build/mars_detect /mnt/nfs/
```

## Usage

```bash
# On T41 device
LD_LIBRARY_PATH=/opt ./mars_detect model.mars input.jpg output.jpg

# With confidence threshold
LD_LIBRARY_PATH=/opt ./mars_detect -t 0.25 model.mars input.ppm output.jpg
```

## Detection Output Format

For TinyDet models, output is `[1, 9, H/16, W/16]`:
- Channel 0: Objectness score
- Channels 1-4: Bounding box (x_off, y_off, width, height)
- Channels 5-8: Class scores (person, vehicle, cat, dog)

## Performance

Typical inference times on T41 @ 1.2GHz:
- TinyDet 320x192: ~1.8s (mostly CPU MXU, NNA acceleration WIP)
- Conv2D: 90% of time in MXU convolution kernels

