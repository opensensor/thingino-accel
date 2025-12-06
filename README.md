<p align="center">
  <img src="mars-logo.png" alt="Mars Logo" width="200"/>
</p>

# Mars: Open-Source Neural Network Runtime for Ingenic T41

**Hardware-accelerated inference on IP cameras using the MXUv3 SIMD unit and ORAM.**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## 🚀 Performance

| Metric | Before (Scalar) | After (Mars) | Speedup |
|--------|-----------------|--------------|---------|
| Inference Time | 35 seconds | **1.75 seconds** | **20x** |
| Memory Read | 41 MB/s (DDR) | 314 MB/s (ORAM) | 7.6x |
| Memory Write | 77 MB/s (DDR) | 1578 MB/s (ORAM) | 20.6x |

## Overview

Mars is an open-source neural network runtime for the Ingenic T41 SoC, reverse-engineered from Ingenic's proprietary Venus SDK. It enables hardware-accelerated inference on IP cameras running [Thingino](https://thingino.com) firmware.

**Why Mars?**
- 🔓 **Open Source**: No proprietary SDKs or closed toolchains
- 🐧 **musl Compatible**: Works with Thingino's musl libc (Venus requires glibc)
- ⚡ **Hardware Accelerated**: Uses MXUv3 SIMD (512-bit) and on-chip ORAM
- 🎯 **Purpose-Built**: Custom TinyDet model for security camera use cases

## Features

- ✅ MXUv3 SIMD acceleration (16 floats per instruction)
- ✅ ORAM weight staging (640KB on-chip, 7.6x faster than DDR)
- ✅ Conv2D, ReLU, MaxPool, Add, Concat operations
- ✅ NHWC tensor format (optimized for T41 memory access)
- ✅ Custom `.mars` model format
- ✅ ONNX → Mars compiler (Python + Rust)

## Quick Start

### Building

```bash
# Set cross-compiler (adjust path to your toolchain)
export CROSS_COMPILE=/path/to/mipsel-linux-

# Build runtime library and tools
make

# Output:
#   build/lib/libmars.so    - Runtime library
#   build/bin/mars_detect   - Detection CLI tool
```

### Running on Device

```bash
# Copy to camera
scp build/bin/mars_detect build/lib/libmars.so root@camera:/opt/

# Run detection
ssh root@camera
cd /opt
LD_LIBRARY_PATH=/opt ./mars_detect model.mars input.jpg output.jpg
```

### Compiling Models

```bash
cd mars-compiler

# Stage 1: ONNX → JSON + weights
python3 onnx2mars.py model.onnx -o model

# Stage 2: JSON → .mars binary
cargo run -- -i model.json -o model.mars --float32
```

## Architecture

```
thingino-accel/
├── src/mars/              # Mars runtime (C)
│   ├── mars_runtime.c     # Model loader and executor
│   ├── mxu_conv.c         # MXUv3 convolution kernels
│   └── mars_nn_hw.c       # Hardware initialization (ORAM, MXU)
├── mars-compiler/         # ONNX → Mars compiler
│   ├── onnx2mars.py       # Python: ONNX → JSON extraction
│   └── src/               # Rust: JSON → .mars binary
├── training/              # TinyDet model training
│   ├── tinydet.py         # Model architecture
│   └── train_*.py         # Training scripts
├── include/               # Public headers
└── docs/                  # Documentation
    └── MARS_PROJECT_WRITEUP.md  # Full research paper
```

## TinyDet: Custom Detection Model

We trained a purpose-built 4-class detector optimized for security cameras:

| Class | Description |
|-------|-------------|
| Person | Human detection |
| Vehicle | Cars, trucks |
| Cat | Feline pets |
| Dog | Canine pets |

**Model specs:**
- Input: 320×192 RGB (NHWC)
- Parameters: ~202K
- Architecture: Anchor-free, single-stage
- Training: COCO + Oxford Pets datasets

## Hardware Details

### Ingenic T41 SoC
- **CPU**: Dual XBurst2 @ 1.5GHz (MIPS)
- **MXUv3**: 512-bit SIMD, 32 VPR registers
- **ORAM**: 640KB @ 0x12640000 (on-chip SRAM)
- **NNA**: Neural Network Accelerator with NNDMA

### Memory Performance

| Region | Bandwidth | Latency |
|--------|-----------|---------|
| DDR | 41 MB/s read | High |
| ORAM | 314 MB/s read | Low |

Weights are staged to ORAM before convolution for maximum throughput.

## Documentation

- 📄 [Full Project Writeup](docs/MARS_PROJECT_WRITEUP.md) - Research paper covering reverse engineering, MXUv3 discovery, and model design
- 📘 [Mars Runtime README](src/mars/README.md) - Runtime architecture and API
- 🎓 [Training README](training/README.md) - Model training guide

## Comparison with Venus SDK

| Feature | Venus (OEM) | Mars |
|---------|-------------|------|
| License | Proprietary | GPLv3 |
| C Library | glibc only | musl/glibc |
| Model Format | `.mgk` (closed) | `.mars` (open) |
| Source Code | No | Yes |
| Compiler | Closed | Python + Rust |

## Contributing

Contributions welcome! See the [project writeup](docs/MARS_PROJECT_WRITEUP.md) for technical background.

## License

GPLv3 - See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Thingino](https://thingino.com) - Open-source IP camera firmware
- [OpenSensor](https://github.com/opensensor) - Project home

