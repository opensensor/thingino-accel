<p align="center">
  <img src="../mars-logo.png" alt="Mars Logo" width="200"/>
</p>

# Mars: Bringing Neural Network Acceleration to Open-Source IP Cameras

**An open-source inference runtime for the Ingenic T41 NNA, reverse-engineered from proprietary firmware and optimized through systematic hardware exploration.**

*Matt Davis, OpenSensor Engineering, 2025*

---

## Abstract

Modern IP cameras are beginning to contain powerful neural network accelerators, but these capabilities tend to be locked behind proprietary SDKs incompatible with open-source toolchains. We present **Mars**, an open-source neural network runtime for the Ingenic T41 SoC that achieves **20x faster inference** than naive implementations through reverse engineering of the MXUv3 SIMD unit and NNA memory subsystem. Our custom TinyDet object detector runs in **1.75 seconds** on device, enabling practical real-time person, vehicle, and pet detection on commodity camera hardware.

---

## 1. Introduction

### 1.1 The Problem: Locked Hardware

The Ingenic T41 SoC powers millions of IP cameras worldwide, including popular models from Wyze, Xiaomi, and various white-label manufacturers. The chip features a dual-core XBurst2 CPU at 1.5GHz with a Neural Network Accelerator (NNA) comprising:

- **MXUv3**: A 512-bit SIMD vector unit with 32 registers
- **ORAM**: 640KB of on-chip fast memory
- **NNDMA**: Dedicated DMA engine for tensor transfers

However, Ingenic's proprietary **Venus SDK** presents several problems:

1. **Toolchain Lock-in**: Venus only works with glibc-based toolchains. The popular Thingino open-source firmware uses musl libc, making Venus fairly incompatible.

2. **Closed Model Format**: Models must be compiled to the proprietary `.mgk` format using Ingenic's closed-source compiler.

3. **No Source Access**: When things break, there's no way to debug or optimize.

### 1.2 Our Solution: Mars

Mars is a complete open-source replacement for Venus, consisting of:

- **Mars Runtime** (C): Executes models on T41 hardware with MXU/ORAM acceleration
- **Mars Compiler** (Rust): Converts ONNX models to `.mars` format
- **TinyDet Model** (PyTorch): Custom 4-class detector optimized for T41

---

## 2. Reverse Engineering Venus

### 2.1 The Starting Point: A Black Box

We began with `libvenus.so`, a 2MB stripped binary with no symbols. The only documentation was example code showing function calls like:

```c
venus_load_model("model.magik", &handle);
venus_run(handle, input, output);
```

### 2.2 Methodology: Binary Archaeology

Our approach combined several techniques:

1. **Static Analysis (Ghidra)**: Decompile and annotate functions
2. **Dynamic Tracing**: Intercept library calls via `LD_PRELOAD` shims
3. **Memory Mapping**: Monitor `/dev/soc-nna` ioctls
4. **Register Probing**: Read hardware registers during execution

### 2.3 Key Discovery: The MXUv3 Instructions

Deep in the Venus binary, we found inline assembly sequences that didn't match any documented MIPS instructions:

```asm
.word 0x70401011    # What is this?
.word 0x70454051    # And this?
```

These were **MXUv3 coprocessor instructions**. Ingenic had added a custom SIMD unit to the XBurst2 core but provided no public documentation.

Through systematic experimentation, we decoded the instruction format:

```
Bits 31-26: Opcode (0x1C = COP2)
Bits 25-21: Function code
Bits 20-16: VPR destination register
Bits 15-11: VPR source register
Bits 10-6:  Additional operand
Bits 5-0:   Sub-function
```

---

## 3. The Shotgun Approach: Discovering MXUv3 Capabilities

### 3.1 When Documentation Doesn't Exist, Experiment

Without instruction set documentation, we took a "shotgun" approach: try every instruction encoding and observe the results.

```c
// Test program: Probe all COP2 function codes
for (int func = 0; func < 64; func++) {
    uint32_t insn = (0x1C << 26) | (func << 21) | ...;
    // Emit instruction, check VPR register state
    asm volatile(".word %0" : : "r"(insn));
    dump_vpr_registers();
}
```

### 3.2 Key Instructions Discovered

After hundreds of experiments, we identified the working instructions:

| Instruction | Encoding | Operation |
|-------------|----------|-----------|
| `LA0_VPR(r, addr)` | 0x70xx10xx | Load 64 bytes to VPR[r] |
| `SA0_VPR(r, addr)` | 0x70xx20xx | Store 64 bytes from VPR[r] |
| `VPR_ADD(d, s)` | 0x4a6xxxx0 | VPR[d] += VPR[s] (float32) |
| `VPR_MUL(d, s)` | 0x4a6xxxx8 | VPR[d] *= VPR[s] (float32) |
| `S4MACSSB` | Various | INT8 4-segment MAC |

### 3.3 The Breakthrough: 16 Floats Per Cycle

Each VPR register holds 512 bits = 16 float32 values. A single `VPR_MUL` instruction multiplies all 16 floats simultaneously:

```c
// Before: Scalar loop (16 iterations)
for (int i = 0; i < 16; i++)
    result[i] = a[i] * b[i];

// After: Single MXU instruction
LA0_VPR(2, a);     // Load 16 floats to VPR2
LA0_VPR(4, b);     // Load 16 floats to VPR4
VPR_MUL(2, 4);     // VPR2 = VPR2 * VPR4
SA0_VPR(2, result); // Store 16 results
```

This gave us an immediate **9x speedup** on convolution operations.

---

## 4. ORAM: The Secret Weapon

### 4.1 Discovery: Why Some Buffers Were Fast

During Venus tracing, we noticed that certain memory regions had dramatically lower latency. Investigation revealed **ORAM** - 640KB of on-chip SRAM at physical address 0x12640000.

### 4.2 Benchmarking: 20x Faster Memory

We built a benchmark tool to quantify the difference:

| Operation | DDR | ORAM | Speedup |
|-----------|-----|------|---------|
| Sequential Read | 41 MB/s | 314 MB/s | **7.6x** |
| Sequential Write | 77 MB/s | 1578 MB/s | **20.6x** |
| MXU Dot Product | 101 ms | 18 ms | **5.55x** |

### 4.3 Integration: Weight Staging

For Conv2D layers, weights are accessed repeatedly for each output position. By staging weights to ORAM before the compute loop, we eliminated the memory bottleneck:

```c
void conv2d_oram(float *input, float *weight, float *output, ...) {
    // Stage weights to ORAM (one-time cost)
    memcpy(oram_weights, weight, weight_size);

    for (each output position) {
        // MXU dot product now reads from fast ORAM
        LA0_VPR(4, oram_weights + offset);  // Fast!
        VPR_MUL(2, 4);
        // ...
    }
}
```

Combined with MXU vectorization, this achieved our **20x total speedup**.

---

## 5. The MGK Decompiler: Audio Models on Floodlight Cameras

### 5.1 An Unexpected Discovery

While examining firmware from a Ring Floodlight V2 camera (also T41-based), we found `.mgk` model files different from the standard image classification models:

```
floodlight_audio_event.mgk    # Audio event detection
floodlight_glass_break.mgk    # Glass breaking detection
```

### 5.2 Decoding the Magik Format

The MGK format proved to be relatively straightforward:

```
Header:
  Magic: "MGK\0"
  Version, layer count, tensor count
  Weights offset, weights size

Tensors:
  Name (64 bytes), shape, dtype, scale, zero_point

Layers:
  Type, input/output IDs, parameters (kernel, stride, etc.)

Weights:
  INT8 quantized, NHWC layout
```

This understanding directly informed our Mars format design.

---

## 6. Why Rust for the Mars Compiler?

### 6.1 The Two-Stage Pipeline

ONNX files are complex Protocol Buffer structures. Rather than implementing protobuf parsing in Rust, we use a pragmatic two-stage approach:

1. **Python Stage** (`onnx2mars.py`): Uses PyTorch/ONNX libraries to parse the model and extract structure to JSON + binary weights
2. **Rust Stage** (`mars`): Reads the JSON intermediate format, applies optimizations, and emits the `.mars` binary

This leverages PyTorch's mature ONNX support while using Rust for the performance-critical compilation.

### 6.2 Rust Advantages

We chose Rust for the core compiler because:

1. **Strong Typing**: ONNX has dozens of operator types with different attribute sets. Rust enums with exhaustive matching caught many bugs at compile time.

2. **Memory Safety**: Weight tensors can be large. Rust's ownership model prevents buffer overflows and memory leaks during tensor reshaping.

3. **Fast Binary Generation**: The Mars format requires careful byte-level layout. Rust's `byteorder` crate and zero-copy patterns make this efficient.

4. **Cross-Platform**: Same compiler works on x86 development machines and can cross-compile for ARM if needed.

```rust
// Example: Exhaustive operator matching
match node.op_type.as_str() {
    "Conv" => emit_conv(node, weights),
    "Relu" => emit_relu(node),
    "Add" => emit_add(node),
    _ => panic!("Unsupported op: {}", node.op_type),
}
```

### 6.3 The Python/Rust Split

The division of labor is deliberate:

| Stage | Language | Responsibility |
|-------|----------|----------------|
| ONNX Parsing | Python | Leverage existing `onnx` library |
| Weight Extraction | Python | NumPy for array manipulation |
| Format Conversion | Rust | NCHW→NHWC transpose, INT8 quantization |
| Binary Generation | Rust | Mars header, tensor descriptors, weight packing |

This approach avoids reimplementing protobuf parsing while gaining Rust's benefits for the CPU-intensive parts.

---

## 7. TinyDet: A Purpose-Built Detection Model

### 7.1 Why Not Just Use YOLO?

YOLOv5 is excellent, but problematic for T41:

| Model | Parameters | Input Size | T41 Inference |
|-------|-----------|------------|---------------|
| YOLOv5n | 1.9M | 640×640 | ~30+ seconds |
| YOLOv5s | 7.2M | 640×640 | Out of memory |
| **TinyDet** | **202K** | **320×192** | **1.75 seconds** |

### 7.2 Architecture Design Principles

We designed TinyDet with T41 constraints in mind:

1. **No Depthwise Convolutions**: MXU handles standard convs efficiently
2. **Channel Counts 16/32/64**: Aligned to VPR register size (16 floats)
3. **Single-Scale Output**: 20×12 grid sufficient for security camera FOV
4. **ReLU Activation**: Simple, fast on MXU
5. **NHWC Format**: Consecutive channels enable efficient VPR loads

### 7.3 Training on Balanced Data

Our 4-class detector targets home security scenarios:

| Class | Training Samples | Source |
|-------|------------------|--------|
| Person | 8,000 | COCO 2017 |
| Vehicle | 7,003 | COCO 2017 |
| Cat | 5,957 | COCO + Oxford Pets |
| Dog | 8,006 | COCO + Oxford Pets |

Training features:
- CIoU loss for box regression
- Focal loss (γ=2.0) for class imbalance
- Mosaic augmentation for small object robustness
- Cosine annealing LR schedule

### 7.4 Results

On COCO validation subset:

| Class | AP@0.5 |
|-------|--------|
| Person | 0.42 |
| Vehicle | 0.38 |
| Cat | 0.35 |
| Dog | 0.33 |
| **mAP** | **0.37** |

While not state-of-the-art, this is **practical accuracy** at **practical speed** on **commodity hardware**.

---

## 8. Open Source: Why It Matters

### 8.1 Toolchain Freedom

With Mars, users can:
- Use any C toolchain (glibc, musl, uClibc)
- Compile with custom optimization flags
- Debug with GDB and Valgrind
- Profile with perf and ftrace

### 8.2 Model Freedom

The ONNX→Mars pipeline means:
- Train in PyTorch, TensorFlow, or any framework
- Export to standard ONNX
- Compile to Mars with no vendor lock-in
- Iterate locally without cloud services

### 8.3 Community Development

Open source enables:
- Bug fixes without vendor support tickets
- Performance optimizations by the community
- Ports to new hardware (T40, T31, etc.)
- Educational use and documentation

---

## 9. Lessons Learned

### 9.1 Hardware Is Discoverable

Even without documentation, systematic experimentation can reveal hardware capabilities. Our shotgun approach to MXUv3 instruction discovery, while time-consuming, produced a working understanding.

### 9.2 Memory Hierarchy Is Critical

The 20x speedup came primarily from:
- 9x from MXU vectorization
- 2.3x from ORAM weight staging

Understanding the memory hierarchy was more important than optimizing compute.

### 9.3 Simple Models Can Be Effective

TinyDet's 202K parameters are sufficient for practical security camera detection. Over-engineered models are counterproductive on constrained hardware.

### 9.4 Reverse Engineering Is Legal (Usually)

In the United States, reverse engineering for interoperability purposes is protected under DMCA exemptions. We used clean-room techniques where possible and documented our methodology.

---

## 10. Future Work

### 10.1 Full NNA Hardware Acceleration

The T41's NNA includes dedicated MAC units beyond MXUv3. Venus uses these for INT8 convolution with specialized microcode sequences (`func=0x30`, `0x31`, `0x2e`). Decoding these could yield another 5-10x speedup.

### 10.2 Quantization-Aware Training

Currently TinyDet is float32. INT8 quantization would:
- Reduce model size 4x
- Enable faster MXU operations (64 int8 vs 16 float32 per VPR)
- Match Venus performance characteristics

### 10.3 Additional Architectures

Other camera SoCs use similar NNA designs:
- Ingenic T31: Older NNA with similar concepts
- Ingenic T40: Different register layout
- Other vendors: Fullhan, Novatek, etc.

---

## 11. Conclusion

Mars demonstrates that sophisticated neural network acceleration is achievable on commodity hardware through systematic reverse engineering and careful optimization. Our work enables the open-source camera community to leverage hardware capabilities that were previously locked behind proprietary SDKs.

The complete source code is available at: https://github.com/opensensor/thingino-accel

---

## Acknowledgments

- The Thingino Project for open-source camera firmware
- The folkas at BinaryNinja and the Ghidra team at NSA for the decompilers I leverage
- Ingenic for building interesting hardware (even if they won't document it publicly)
- Contributors who have tested on various camera models

---

## Appendix A: Performance Timeline

| Date | Optimization | Inference Time | Speedup |
|------|-------------|----------------|---------|
| Day 1 | Scalar C baseline | 40.0s | 1.0x |
| Week 1 | Basic loop optimizations | 28.0s | 1.25x |
| Week 2 | MXU VPR_MUL discovery | 12.0s | 2.9x |
| Week 3 | im2col + full MXU | 4.0s | 8.75x |
| Week 4 | ORAM weight staging | 1.75s | **20x** |

---

## Appendix B: Repository Structure

```
thingino-accel/
├── src/mars/           # Runtime source
│   ├── mars_runtime.c  # Model loading, layer execution
│   ├── mxu_conv.c      # MXU-accelerated convolution
│   ├── mars_nn_hw.c    # ORAM/DDR memory management
│   └── mars_detect.c   # Detection application
├── mars-compiler/      # ONNX→Mars compiler (Rust)
│   ├── src/main.rs     # CLI and pipeline
│   ├── onnx2mars.py    # ONNX parser
│   └── mars_format.rs  # Binary format writer
├── training/           # TinyDet training
│   ├── tinydet.py      # Model architecture
│   ├── train_improved.py # Training script
│   └── export_onnx.py  # ONNX export
├── docs/               # Documentation
│   ├── mxuv3_instructions.md
│   └── t41_nna_architecture.md
└── tools/              # Debugging and benchmarks
    ├── oram_bench.c    # Memory benchmark
    └── venus_trace.c   # Library tracer
```

---

*"The best way to understand hardware is to make it do things."*


