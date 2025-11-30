# MXUv3 Instruction Reference for Ingenic T41 (XBurst2)

**Status: Work in Progress - Reverse Engineered from libvenus.so**

## Overview

The T41 SoC uses MXUv3 (Media Extension Unit v3), which is significantly different from MXU1/MXU2:
- **32 VPR registers** (VPR0-VPR31), each **512 bits** (64 bytes)
- Uses **SPECIAL2 opcode** (0x1C = 28) for most instructions
- Uses **COP2 opcode** (0x12 = 18) for compute instructions
- Requires **64-byte aligned memory** for load/store
- May require **NNA-allocated memory** for proper operation

## Instruction Encoding

### SPECIAL2 Format (opcode = 0x1C = 28)
```
[31:26] = 011100 (SPECIAL2 = 28)
[25:21] = rs (base register or mode)
[20:16] = rt (source VPR or GPR)
[15:11] = rd (destination VPR or GPR)
[10:6]  = sa (VPR number or shift)
[5:0]   = func (function code)
```

### COP2 Format (opcode = 0x12 = 18)
```
[31:26] = 010010 (COP2 = 18)
[25]    = co (coprocessor operation = 1)
[24:21] = sub (sub-opcode)
[20:16] = rt (VPR number)
[15:11] = rd (VPR number)
[10:6]  = sa (VPR number or mode)
[5:0]   = func (function code)
```

## Verified Instructions

### LA0 - Load Aligned 0 (func=0x11)
Loads 256 bits from memory to VPR register. Need 2 loads for full 512-bit VPR.

```c
// Encoding: 0x71001811 | (offset << 16) | (n << 14) | (vprn << 6)
// offset = byte offset / 32, n = half (0=low, 1=high), vprn = VPR number
// Uses $t0 as base address

// Example: Load full VPR0
.word 0x71001811   // VPR0 low half
.word 0x71015811   // VPR0 high half

// VPR numbers (in sa field):
// VPR0:  0x71001811  VPR1:  0x71001851  VPR2:  0x71001891
// VPR29: 0x71001f51  VPR30: 0x71001f91
```

### SA0 - Store Aligned 0 (func=0x15)
Stores 256 bits from VPR register to memory. Need 2 stores for full 512-bit VPR.

```c
// Encoding: 0x710000d5 | (offset << 16) | (vprn << 11) | (n << 9)
// Uses $t0 as base address

// Example: Store full VPR0
.word 0x710000d5   // VPR0 low half
.word 0x710102d5   // VPR0 high half

// VPR numbers (in rd field):
// VPR0:  0x710000d5  VPR1:  0x710008d5  VPR2:  0x710010d5
// VPR6:  0x710030d5  VPR29: 0x7100e8d5  VPR30: 0x7100f0d5
```

### func=0x38 - VPR Concatenate
Concatenates two VPR registers into destination.

```c
// Encoding: 0x70000038 | (rs << 21) | (rt << 16) | (rd << 11) | (sa << 6)
// Operation: VPR[sa] = {VPR[rt], VPR[rd]}
//            VPR[sa][0:31]  = VPR[rt][0:31]   (first 32 bytes)
//            VPR[sa][32:63] = VPR[rd][0:31]   (second 32 bytes)

// Example: 0x70fdf038 (rs=7, rt=29, rd=30, sa=0)
// VPR0 = {VPR29, VPR30}
// VPR0[0:31] = VPR29, VPR0[32:63] = VPR30

// Other examples:
// 0x70fdf078 (sa=1): VPR1 = {VPR29, VPR30}
// 0x70fdf0b8 (sa=2): VPR2 = {VPR29, VPR30}
// 0x70fdf0f8 (sa=3): VPR3 = {VPR29, VPR30}
```

### func=0x39 - VPR Sync/Copy
Appears to be a sync or self-copy operation.

```c
// 0x7108ef79: rs=8, rt=7, rd=29, sa=29 - VPR29 sync
// 0x7108f7b9: rs=8, rt=7, rd=30, sa=30 - VPR30 sync
```

### func=0x30 - Setup/Clear
Setup instruction, possibly clears internal accumulators.

```c
// 0x70f0ef70: rs=7, rt=16, rd=29, sa=29
// 0x70f0f7b0: rs=7, rt=16, rd=30, sa=30
```

### c2 func=0x02 - Clear VPR
Clears a VPR register.

```c
// Encoding: c2 with sub=3, rd=6, func=0x02
// 0x4a603002: Clear VPR0 (sa=0, rt=0)
// 0x4a613042: Clear VPR1 (sa=1, rt=1)
// 0x4a623082: Clear VPR2 (sa=2, rt=2)
// 0x4a6330c2: Clear VPR3 (sa=3, rt=3)
```

### c2 func=0x08 - Copy VPR to VPR0
Copies VPR[rd] to VPR0.

```c
// Encoding: c2 with sub=0, func=0x08
// Operation: VPR0 = VPR[rd]
// 0x4a000808: VPR0 = VPR1 (rd=1)
// 0x4a001008: VPR0 = VPR2 (rd=2)
// 0x4a001808: VPR0 = VPR3 (rd=3)
// 0x4a002808: VPR0 = VPR5 (rd=5)
```

### c2 func=0x0b - MAC Operation (part 1)
Multiply-Accumulate operation, uses VPR29 as weight source.

```c
// Encoding: c2 with sub=4, rd=29, func=0x0b
// 0x4a82e88b: rt=2, sa=2
// 0x4a83e8cb: rt=3, sa=3
// 0x4a84e90b: rt=4, sa=4
// 0x4a85e94b: rt=5, sa=5
```

### c2 func=0x0d - Copy VPR to VPR0 (variant)
Similar to func=0x08, copies VPR[rd] to VPR0.

```c
// Encoding: c2 with sub=0, func=0x0d
// Operation: VPR0 = VPR[rd]
// 0x4a00080d: VPR0 = VPR1 (rd=1)
// 0x4a00100d: VPR0 = VPR2 (rd=2)
// 0x4a00180d: VPR0 = VPR3 (rd=3)
```

### c2 func=0x23 - MAC Operation (part 2)
Second part of MAC, uses VPR30 as weight source.

```c
// Encoding: c2 with sub=3, rd=30, func=0x23
// 0x4a62f0a3: rt=2, sa=2
// 0x4a63f0e3: rt=3, sa=3
// 0x4a64f123: rt=4, sa=4
// 0x4a65f163: rt=5, sa=5
```

## Typical MAC Sequence (from libvenus)

```asm
# Setup phase
.word 0x70f0ef70    # func=0x30 setup VPR29
.word 0x70f0f7b0    # func=0x30 setup VPR30

# Load input data (LA0 to VPR0)
.word 0x70401811    # LA0 VPR0 low (base in $v0)
.word 0x70415811    # LA0 VPR0 high

# Load weights (func=0x31 - uses $t5 as base)
.word 0x71a00071    # Load weights part 1
.word 0x71a80031    # Load weights part 2
# ... more weight loads

# Setup for MAC (func=0x2e)
.word 0x704200ae
.word 0x7049026e
# ... more setup

# MAC operations
c2 0x82e88b         # func=0x0b with VPR29
c2 0x83e8cb
c2 0x84e90b
c2 0x85e94b
c2 0x62f0a3         # func=0x23 with VPR30
c2 0x63f0e3
c2 0x64f123
c2 0x65f163
```

## Data Type Conversion

The MXU performs automatic data type conversion during MAC operations:

### Int8 to Float32 Conversion
After the MAC sequence, certain VPR registers contain float32 values:
- **VPR9**: Contains floats from input bytes 17-24
- **VPR11**: Contains floats from input bytes 33-40
- **VPR13**: Contains floats from input bytes 49-56

### Int8 to Int16 Conversion
- **VPR0**: Contains int16 values from input bytes 33-40
- **VPR1**: Contains int16 values from input bytes 1-8

### Example
```
Input: 1, 2, 3, 4, 5, 6, 7, 8, ..., 64 (int8)

After MAC sequence:
VPR9 (float32):  17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0
VPR11 (float32): 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0
VPR13 (float32): 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0
VPR0 (int16):    33, 34, 35, 36, 37, 38, 39, 40
VPR1 (int16):    1, 2, 3, 4, 5, 6, 7, 8
```

## Store Instructions

### func=0x2f - Store (uses $k0/$26 as base)
```c
// 0x734200af: rs=26, rt=2, rd=0, sa=2
// 0x734300ef: rs=26, rt=3, rd=0, sa=3
// 0x7344012f: rs=26, rt=4, rd=0, sa=4
// 0x7345016f: rs=26, rt=5, rd=0, sa=5
```

### func=0x34 - Store (uses $t6/$14 as base)
```c
// 0x71c110b4: rs=14, rt=1, rd=2, sa=2
// 0x71c118f4: rs=14, rt=1, rd=3, sa=3
// 0x71c12134: rs=14, rt=1, rd=4, sa=4
// 0x71c12974: rs=14, rt=1, rd=5, sa=5
```

### func=0x35 - Store
```c
// 0x700010b5: rs=0, rt=0, rd=2, sa=2
// 0x700118b5: rs=0, rt=1, rd=3, sa=2
// 0x700020f5: rs=0, rt=0, rd=4, sa=3
// 0x700128f5: rs=0, rt=1, rd=5, sa=3
```

## Notes

1. **Memory Requirements**: Tests with stack memory produce partial results; NNA-allocated memory via `IOCTL_SOC_NNA_MALLOC` is required for full functionality.

2. **Register Usage**:
   - rs=13 ($t5) is commonly used as base for weight loading (func=0x31)
   - rs=26 ($k0) is used for store (func=0x2f)
   - rs=14 ($t6) is used for store (func=0x34)

3. **VPR29/VPR30**: These appear to be special source registers for MAC operations.

4. **VPR6**: The rd=6 field in c2 instructions suggests VPR6 may be a special control register.

5. **Data Layout**: The MXU processes data in specific patterns:
   - VPR9/11/13 contain float32 conversions of specific input byte ranges
   - VPR0/1 contain int16 conversions of specific input byte ranges

## Verified Use Cases

### 1. Fast Memory Copy (64 bytes at a time)
Using LA0 and SA0 to copy 64 bytes per operation:

```c
register void *base __asm__("t0");

/* Load 64 bytes from src to VPR0 */
base = src;
__asm__ __volatile__(
    ".word 0x71001811\n"  /* LA0 VPR0 low half */
    ".word 0x71015811\n"  /* LA0 VPR0 high half */
    :: "r"(base) : "memory"
);

/* Store 64 bytes from VPR0 to dst */
base = dst;
__asm__ __volatile__(
    ".word 0x710000d5\n"  /* SA0 VPR0 low half */
    ".word 0x710102d5\n"  /* SA0 VPR0 high half */
    :: "r"(base) : "memory"
);
```

### 2. Fast Memory Set (64 bytes at a time)
Set 64 bytes to a pattern value:

```c
/* First load pattern to VPR0, then store to multiple destinations */
for (size_t i = 0; i < size; i += 64) {
    base = dst + i;
    __asm__ __volatile__(
        ".word 0x710000d5\n .word 0x710102d5\n"
        :: "r"(base) : "memory"
    );
}
```

### 3. Int8 to Float32 Conversion
The MAC sequence converts int8 input to float32:
- Input bytes 17-32 → VPR9 as float32 (16 values)
- Input bytes 33-48 → VPR11 as float32 (16 values)
- Input bytes 49-64 → VPR13 as float32 (16 values)

## Limitations

1. **MAC Operations Not Fully Decoded**: The exact multiply-accumulate formula for the c2 `func=0x0b/0x23` pair is not yet understood. These instructions clearly perform data conversion (int8→int16/float32) and some accumulation, but the final MAC results appear to live in internal accumulators rather than any VPR that we can load/store.

2. **Store Instructions**: The func=0x2f, 0x34, 0x35 store instructions do not appear to write to the expected output buffer in our standalone tests. In Venus they are typically paired with NNA‑managed buffers and descriptor setup; they likely depend on proper NNDMA/AIP configuration.

3. **NNA Hardware Dependency**: Full functionality requires NNA hardware initialization and memory allocation via `/dev/soc-nna`. Using stack/heap buffers outside the NNA allocator typically gives only partial or no results.

## COP2 Vector Arithmetic (ADD / SUB / MUL)

In addition to the special MAC microcode above, libvenus makes heavy use of a much more conventional COP2 vector arithmetic class that operates on full 512‑bit VPR registers as 16× `float32` lanes.

### Encoding

- Opcode: **COP2** (`0x12`)
- Fields:
  - `rs` – operation class selector
  - `rt` – source vector 2
  - `rd` – destination vector
  - `sa` – source vector 1 (often must equal `rd`)
  - `fn` – function within the operation class

From libvenus disassembly we see two main classes:

- **rs = 20** – Add/Sub class
  - `fn = 3`  → **ADD**  — `VPR[rd] = VPR[sa] + VPR[rt]`
  - `fn = 11` → **SUB**  — `VPR[rd] = VPR[sa] - VPR[rt]`
- **rs = 19** – Multiply class
  - `fn = 35` → **MUL**  — `VPR[rd] = VPR[sa] * VPR[rt]`

With the hardware constraint that for in‑place operations we use `rd = sa` so the instruction behaves like:

- `ADD`: `VPR[dst] = VPR[dst] + VPR[src]`
- `SUB`: `VPR[dst] = VPR[dst] - VPR[src]`
- `MUL`: `VPR[dst] = VPR[dst] * VPR[src]`

These encodings are wrapped in `include/mxuv3.h` as:

- `VPR_ADD(dst, src)` – 16‑lane float vector add
- `VPR_SUB(dst, src)` – 16‑lane float vector subtract
- `VPR_MUL(dst, src)` – 16‑lane float vector multiply

and are used by the Mars runtime in `src/mars/mxu_conv.c` to implement MXU‑accelerated elementwise ops and convolutions.

### Unary Variants

By setting all three registers equal (`rt = rd = sa`), we get useful unary variants:

- `VPR_SQR(reg)`  — `VPR[reg] = VPR[reg] * VPR[reg]`
- `VPR_DBL(reg)`  — `VPR[reg] = VPR[reg] + VPR[reg]`
- `VPR_ZERO(reg)` — `VPR[reg] = VPR[reg] - VPR[reg] = 0`

These are exposed in `mxuv3.h` and are handy building blocks for activation functions and simple fused arithmetic in Mars.

## Missing Vector Instructions (DIV / SQRT / RSQRT)

A negative but important discovery from libvenus is that **MXUv3 does not provide vector division or square‑root instructions**:

- There are **hundreds** of scalar FPU `div.s` and `sqrt.s` instructions in libvenus.
- There are **zero** MXU opcodes that look like reciprocal, rsqrt, or divide operations.

In practice Venus (and Mars) handle these operations by:

1. **Precomputing reciprocals** during training or model conversion (e.g., batch‑norm fusion stores `1 / sqrt(variance)` directly in the weights).
2. Using MXUv3 **MUL** for the runtime part (`y = (x - mean) * inv_std` instead of `y = (x - mean) / std`).
3. Falling back to the scalar FPU for any rare true divide/sqrt that cannot be folded into weights.

For Mars, the guideline is:

- Prefer **MXU MUL/ADD/SUB** with pre‑baked constants.
- Use scalar FPU only when absolutely necessary.

## Performance Results

Even without full MAC instruction support, optimized scalar code with proper loop unrolling provides significant speedups:

| Input Size | Kernel | Speedup |
|------------|--------|---------|
| 160x160x3  | 3x3 s2 | 1.06x   |
| 80x80x16   | 3x3 s2 | 1.76x   |
| 40x40x32   | 3x3 s2 | 2.04x   |
| 20x20x64   | 3x3 s1 | 2.33x   |
| 20x20x128  | 1x1 s1 | 2.14x   |

Speedup increases with channel depth due to better cache utilization with unrolled inner loops.

