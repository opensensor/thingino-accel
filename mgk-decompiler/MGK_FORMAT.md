# MGK File Format Specification

## Overview

MGK (Magik) is the neural network model format used by Ingenic's T31/T41 Neural Network Accelerator (NNA). This document describes the format based on reverse engineering of `AEC_T41_16K_NS_OUT_UC.mgk`.

## File Structure

```
+------------------+
| ELF Header       |  Standard MIPS32 Little-Endian ELF
+------------------+
| .text section    |  Executable code (383 KB)
+------------------+
| .rodata section  |  Read-only data (29 KB)
|   - Layer names  |
|   - Tensor info  |
|   - Quant scales |
+------------------+
| .data.rel.ro     |  Relocation data (2.7 KB)
+------------------+
| Appended Data    |  Weight data (153 KB)
+------------------+
```

## ELF Sections

| Section       | Offset   | Size    | Description                    |
|---------------|----------|---------|--------------------------------|
| .text         | 0x0e6d0  | 383 KB  | MIPS32 executable code         |
| .rodata       | 0x6c640  | 29 KB   | Layer names, scales, metadata  |
| .data.rel.ro  | 0x772e8  | 2.7 KB  | Weight data pointers           |
| Appended      | 0x79294  | 153 KB  | INT8 quantized weights         |

## Weight Data Format

### Location
- Base offset: `0x79294` (after ELF sections)
- Total size: 153,644 bytes

### Quantization
- Weights are stored as INT8 (-128 to 127)
- Scale factors are stored in .rodata as FP32
- Dequantization: `float_weight = int8_weight * scale`

### NMHWSOIB2 Weight Layout (Convolutions)

The NNA uses a custom weight layout called **NMHWSOIB2** for optimal hardware acceleration:

```
NMHWSOIB2: [N_OFP, M_IFP, KERNEL_H, KERNEL_W, S_BIT2, OFP, IFP]
```

Where:
- **N_OFP**: Number of Output Feature Planes = ceil(out_channels / 32)
- **M_IFP**: Number of Input Feature Planes = ceil(in_channels / 32)
- **KERNEL_H, KERNEL_W**: Convolution kernel dimensions
- **S_BIT2**: 2-bit grouping factor (4 for 8-bit weights)
- **OFP**: Channels per output feature plane (32)
- **IFP**: Channels per input feature plane (32)

**Size Formula**: `N_OFP × M_IFP × KH × KW × 1024` bytes

| Layer Type | Size Formula | Example (32→32) |
|------------|--------------|-----------------|
| Conv 1x1   | 1 × 1 × 1 × 1 × 1024 | 1,024 bytes |
| Conv 3x3   | 1 × 1 × 3 × 3 × 1024 | 9,216 bytes |
| Conv 5x5   | 1 × 1 × 5 × 5 × 1024 | 25,600 bytes |

**Unpacking to OIHW**:
```python
# Reshape: [N_OFP, M_IFP, KH, KW, 32, 32]
reshaped = data.reshape(n_ofp, m_ifp, kh, kw, 32, 32)
# Transpose: [N_OFP, 32, M_IFP, 32, KH, KW]
transposed = reshaped.transpose(0, 4, 1, 5, 2, 3)
# Reshape to standard OIHW: [out_ch, in_ch, kh, kw]
output = transposed.reshape(n_ofp * 32, m_ifp * 32, kh, kw)
```

### GRU Weight Layout

GRU weights use a 1024-byte block structure with a 768+256 byte pattern:

**Bidirectional GRU (12,864 bytes)**:
- 12 blocks × 1024 bytes = 12,288 bytes (weight matrices)
- 576 bytes (biases)
- Each block: 768 bytes (24×32) + 256 bytes (8×32) = 32×32 matrix

Block organization:
- Blocks 0-5: Forward direction (weight_ir, weight_iz, weight_in, weight_hr, weight_hz, weight_hn)
- Blocks 6-11: Backward direction (same order)

**Unidirectional GRU (4,096 bytes)**:
- 4 blocks × 1024 bytes
- Higher sparsity (~57% zeros)
- Blocks 0-1: weight_ih (64×32)
- Blocks 2-3: weight_hh (64×32)

### Known Layer Offsets (AEC Model)

```
0x00000: layer_46_gru_bidir (12,864 bytes) - Bidirectional GRU
0x03500: layer_63_feature (448 bytes)
0x03900: layer_68_feature (448 bytes)
0x03d00: layer_35_feature (704 bytes)
0x04100: layer_73_feature (448 bytes)
0x04480: main_conv_region (55,168 bytes) - Encoder convolutions
0x11f00: layer_44_feature (576 bytes)
0x12300: layer_58_feature (576 bytes)
0x12700: layer_78_feature (320 bytes)
0x12a00: layer_4_feature (3,648 bytes)
0x13b00: layer_16_feature (2,112 bytes)
0x14b00: layer_2_feature (320 bytes)
0x16d00: secondary_conv_region (41,792 bytes) - Decoder convolutions
0x21180: layer_20_feature (832 bytes)
0x215c0: layer_26_feature (832 bytes)
0x21a40: layer_28_feature (1,408 bytes)
0x220c0: layer_37_gru (4,096 bytes) - Unidirectional GRU
0x231c0: layer_10_feature (2,496 bytes)
0x23cc0: layer_32_feature (768 bytes)
0x24100: layer_41_feature (704 bytes)
0x24500: layer_8_feature (1,024 bytes)
0x24a00: layer_14_feature (1,024 bytes)
0x25140: layer_22_feature (1,772 bytes)
```

## Quantization Scales

Scales are stored in .rodata section starting around offset `0x6d410`.

Format: Groups of 4 FP32 values (16 bytes each)
- Typically: [input_scale, input_scale, weight_scale, weight_scale]

Example scales:
```
Group 0: [0.0236, 0.0236, 0.1035, 0.1035]
Group 1: [0.0237, 0.0237, 0.0446, 0.0446]
```

## Layer Naming Convention

```
layer_{id}_Quantize{Type}
```

Types:
- `Feature` - Convolutional feature extraction
- `BatchNorm` - Batch normalization
- `GRU` - Gated Recurrent Unit

## Model Architecture (AEC Example)

- **Input**: `[1, 1, 256, 8]` FP32 - 256 frequency bins, 8 time frames
- **Output**: `[1, 1, 256, 2]` FP32 - Sigmoid mask for echo cancellation
- **Hidden**: `[64, 1, 1, 32]` UINT8 - GRU hidden state

Architecture: U-Net with GRU bottleneck
- Encoder: 5 downsampling stages with skip connections
- Bottleneck: Bidirectional GRU
- Decoder: 5 upsampling stages with skip connection concatenation

## Key Functions (from Binary Analysis)

| Function | Address | Purpose |
|----------|---------|---------|
| `conv2d_int8_param_init` | 0x2ad2c | Initialize conv layer weights |
| `gru_param_init` | 0x24f40 | Initialize GRU weights |
| `magik_set_tensor` | - | Set tensor data pointer |

## Data Format Enums

From `uranus_common_type.h`:

```c
enum DataFormat {
    NHWC = 0,      // Feature: [N, H, W, C]
    NDHWC32 = 1,   // Feature: [N, D_C32, H, W, CHN_32] - 32-channel groups
    HWIO = 2,      // Weight: [H, W, I, O]
    NMHWSOIB2 = 3, // Weight: [N_OFP, M_IFP, H, W, S_BIT2, OFP, IFP]
    NMC32 = 4,     // Bias/BN: [N_OFP, M_BT, CHN_32]
    D1 = 5,        // Scale/LUT: [d1]
    NV12 = 6,      // Image: NV12 format
    OHWI = 7,      // Weight: [O, H, W, I]
    NCHW = 8,      // Feature: [N, C, H, W]
};

enum DataType {
    FP32 = 0,
    UINT8 = 11,
    INT8 = 12,
    // Various bit widths: 2, 4, 6, 8, 10, 12, 14, 16 bit
};
```

## Tools

- `mgk-decompiler/` - Rust-based MGK parser
- `scripts/extract_weights_nmhwsoib2.py` - NMHWSOIB2 weight unpacker
- `scripts/aec_model_v2.py` - PyTorch AEC model implementation

## References

- Ingenic T41 SDK documentation
- Binary Ninja decompilation analysis
- Runtime log analysis from device execution
- magik-toolkit headers (uranus_common_type.h)

