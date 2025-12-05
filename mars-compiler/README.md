# Mars Compiler

Compiles ONNX neural network models to the `.mars` binary format for execution on the Ingenic T41 NNA.

## Overview

The compiler converts ONNX models through a two-stage process:

```
ONNX Model (.onnx)
       │
       ▼  onnx2mars.py
JSON + Binary Weights (.json + .bin)
       │
       ▼  mars (Rust compiler)
Mars Model (.mars)
```

## Installation

```bash
# Build the Rust compiler
cd mars-compiler
cargo build --release
```

## Usage

### Full Pipeline

```bash
# Convert ONNX to Mars
python3 onnx2mars.py model.onnx -o model.mars

# Then compile to binary
./target/release/mars --input model.json --output model.mars
```

### Using the Test Script

```bash
# From training directory - exports checkpoint and compiles
python3 test_model.py runs/balanced_4class/tinydet_best.pth
```

## File Formats

### JSON Intermediate Format

```json
{
  "inputs": [{"name": "input", "shape": [1, 3, 192, 320], "dtype": "float32"}],
  "outputs": [{"name": "output", "shape": [1, 9, 12, 20], "dtype": "float32"}],
  "nodes": [
    {"op_type": "Conv", "inputs": ["input", "weight", "bias"], ...}
  ],
  "initializers": [
    {"name": "weight", "shape": [16, 3, 3, 3], "dtype": "float32", "offset": 0}
  ]
}
```

### Mars Binary Format

```
┌────────────────────────────────────┐
│ Header (76 bytes)                  │
│   magic: 0x5352414D ("MARS")       │
│   version, num_layers, num_tensors │
│   weights_offset, weights_size     │
└────────────────────────────────────┘
┌────────────────────────────────────┐
│ Tensor Descriptors (124 bytes each)│
│   name, dtype, format, shape       │
│   scale, zero_point                │
└────────────────────────────────────┘
┌────────────────────────────────────┐
│ Layer Descriptors (112 bytes each) │
│   type, inputs, outputs            │
│   kernel, stride, padding          │
│   activation                       │
└────────────────────────────────────┘
┌────────────────────────────────────┐
│ Weight Data (INT8 quantized)       │
│   NHWC layout for all weights      │
└────────────────────────────────────┘
```

## Supported Operations

| ONNX Op | Mars Support | Notes |
|---------|--------------|-------|
| Conv | ✓ | 2D convolutions, groups=1 |
| BatchNormalization | ✓ | Fused with Conv |
| Relu | ✓ | Fused or standalone |
| MaxPool | ✓ | 2x2 stride 2 |
| Add | ✓ | Skip connections |
| Concat | ✓ | Channel dimension |
| Upsample | ✓ | Nearest neighbor 2x |
| Reshape | ✓ | No data movement |
| Transpose | ✓ | For format conversion |

## Quantization

The compiler performs INT8 quantization:

1. **Weight quantization**: Per-tensor symmetric
   ```
   scale = max(abs(weight)) / 127
   weight_int8 = round(weight / scale)
   ```

2. **Activation quantization**: Calibrated from representative data
   - Default scale: 1/255 for input (0-1 normalized)
   - Output scales propagated through layers

## Source Files

| File | Description |
|------|-------------|
| `onnx2mars.py` | ONNX parser, extracts structure and weights |
| `main.rs` | CLI and compilation pipeline |
| `mars_format.rs` | Mars binary format writer |
| `onnx_parser.rs` | JSON intermediate format parser |
| `darknet_parser.rs` | (Optional) Darknet config parser |

## Output Size

Typical compiled model sizes:
- TinyDet 4-class (202K params): ~210 KB
- YOLOv5n: ~2-4 MB (depending on classes)

