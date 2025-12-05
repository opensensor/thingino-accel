# Mars Runtime Implementation

The Mars Runtime is a neural network inference engine optimized for the **Ingenic T41 SoC**, leveraging its **MXUv3 (Media Extension Unit v3)** SIMD capabilities for accelerated compute.

## Performance

| Configuration | Inference Time | Speedup |
|---------------|---------------|---------|
| Scalar (baseline) | ~35 seconds | 1x |
| MXUv3 optimized | ~4 seconds | **~9x** |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Mars Runtime                              │
├─────────────────────────────────────────────────────────────┤
│  Model Loading (.mars)  │  Tensor Management  │  Execution  │
├─────────────────────────────────────────────────────────────┤
│                    MXUv3 Acceleration Layer                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Conv2D      │  │ Element-wise│  │ Activation Functions│  │
│  │ (im2col +   │  │ Mul/Add     │  │ ReLU, Sigmoid, SiLU │  │
│  │  VPR SIMD)  │  │ (VPR SIMD)  │  │ (Scalar)            │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    NNA Memory Management                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ DDR (8MB)   │  │ NMEM (29MB) │  │ ORAM (384KB)        │  │
│  │ Weights     │  │ Tensors     │  │ (Future use)        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Hardware Acceleration Summary

### MXUv3-Accelerated Operations

| Operation | Data Type | Acceleration | Notes |
|-----------|-----------|--------------|-------|
| **Conv2D** | Float32 NHWC | ✅ VPR SIMD | im2col + vectorized dot product |
| **Conv2D** | Float32 NCHW | ✅ VPR SIMD | Channel-first layout |
| **Conv2D** | INT8 NHWC | ✅ S4MACSSB | 4-way MAC with accumulator |
| **Conv2D** | INT8 NCHW | ✅ S4MACSSB | 4-way MAC with accumulator |
| **Mul** | Float32 | ✅ VPR_MUL | 16 floats/cycle |
| **Add** | Float32 | ✅ VPR_ADD | 16 floats/cycle |

### Scalar Operations (CPU)

| Operation | Notes |
|-----------|-------|
| ReLU, ReLU6, LeakyReLU | Simple branch per element |
| Sigmoid | exp() per element |
| SiLU | x * sigmoid(x) |
| MaxPool | Comparison loop |
| Concat | Memory copy |
| Upsample | Nearest-neighbor interpolation |
| Reshape, Transpose | Memory reordering |
| BatchNorm | Affine transform |

### NNA Memory & DMA (Available but Unused in Hot Path)

| Component | Status | Notes |
|-----------|--------|-------|
| **NNA Memory Allocation** | ✅ Used | `nna_malloc_phys()` for tensor memory from nmem pool |
| **ORAM** | ⚠️ Implemented | 384KB on-chip SRAM, ~10x faster than DDR |
| **NNDMA** | ⚠️ Implemented | DDR↔ORAM DMA transfers available |
| **NNR/NNMAC** | ❌ Not used | Dedicated matrix multiply hardware - not yet reverse-engineered |

**Current approach**: MXUv3 SIMD operates on tensors in DDR/nmem directly.

**Potential optimization**: Use ORAM for active tensors + NNDMA for prefetching could reduce memory latency significantly. The 384KB ORAM is ~10x faster than DDR access.

## MXUv3 Technical Details

### VPR Registers
- 32 VPR registers, each **512 bits** (64 bytes)
- Can hold **16 float32** or **64 int8** values per register

### Key Instructions Used

```c
// Load 512 bits (16 floats) from memory to VPR register
LA0_VPR(vpr_num, pointer);

// Store 512 bits from VPR register to memory
SA0_VPR(vpr_num, pointer);

// VPR multiply: VPR[dst] = VPR[src] * VPR[dst]
VPR_MUL(dst, src);

// VPR add: VPR[dst] = VPR[src] + VPR[dst]
VPR_ADD(dst, src);

// INT8 4-way MAC with saturation (for convolution)
S4MACSSB(acc, input, weight);

// Zero VSR accumulator register
VSR_ZERO(vsr_num);
```

## Conv2D Optimization Strategy

### Float32 NHWC Convolution

The optimized implementation uses **im2col + vectorized dot product**:

1. **Gather kernel window once** per output position into contiguous buffer
2. **Process all output channels** using the same gathered window
3. **Vectorized dot product**: 16 floats per MXU iteration

```c
// Pseudocode for optimized conv
for each output position (oh, ow):
    // Gather kernel window ONCE
    gather_im2col(input, im2col_buf, oh, ow, kh, kw, stride, pad);
    
    for each output channel (oc):
        sum = 0;
        // MXU vectorized: process 16 floats at a time
        for (i = 0; i + 16 <= kernel_size; i += 16):
            LA0_VPR(2, im2col_buf + i);  // Load input patch
            LA0_VPR(4, weights + i);      // Load weights
            VPR_MUL(2, 4);                // Multiply
            sum += horizontal_sum(VPR2);
        output[oh, ow, oc] = sum + bias[oc];
```

### INT8 NHWC Convolution

Uses `S4MACSSB` for 4-output-channel parallel processing:

- Processes **4 output channels simultaneously**
- Uses VSR accumulator registers for running sums
- Includes optional **spatial tiling** for larger outputs

## Memory Management

### Allocation Strategy
- **Weights**: Loaded to DDR (8MB region) at model load time
- **Tensors**: Dynamically allocated from NMEM pool (29MB) via `nna_malloc_phys()`
- **Working buffers**: 2 reusable buffers for intermediate results
- **Output tensors**: CPU-only allocation (no DMA access needed)

### Tensor Formats Supported
- **NHWC** (channels-last): Primary format, best for MXU
- **NCHW** (channels-first): Supported with separate code path
- **OHWI**: Weight format for NHWC convolutions

## Debug Mode

Enable verbose output with `--debug` flag:
```bash
./mars_detect --debug model.mars input.jpg output.jpg
```

This prints per-layer tensor values for debugging numerical issues.

## Files

| File | Description |
|------|-------------|
| `src/mars/mars_runtime.c` | Main runtime: model loading, layer dispatch |
| `src/mars/mxu_conv.c` | MXU-accelerated convolution implementations |
| `src/mars/mxu_ops.c` | MXU element-wise operations (mul, add) |
| `include/mxuv3.h` | MXU instruction intrinsics |
| `include/mars_runtime.h` | Public API |

