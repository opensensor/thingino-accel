# MXUv3 Test Suite (XBurst2 / T41)

This directory contains **small, single-purpose test programs** that validate
our reverse-engineered MXUv3 (VPR/COP2) findings and provide concrete numbers
we can map onto FFmpeg hotspots.

## Build (cross compile)

From `thingino-accel/`:

> Note: if your shell is already in `.../thingino-accel/`, run `make mxuv3-tests`
> (do **not** use `make -C thingino-accel ...`, which would descend into the
> vendor subtree `thingino-accel/thingino-accel/`).

- Set your toolchain prefix (example for the smart_nvr toolchain wrapper):

  - `export CROSS_COMPILE=/home/matteius/output/master/smart_nvr_a1n_eth-4.4/host/bin/mipsel-linux-`

- Build just the MXUv3 tests:

  - `make clean`
  - `make mxuv3-tests`

From the repository root, the equivalent is:

- `make -C thingino-accel clean`
- `make -C thingino-accel mxuv3-tests`

Output binaries land in: `build/bin/`.

## Deploy to device

Copy the binaries to the target device (example path):

- `scp build/bin/mxu_test build/bin/mxuv3_* root@CAMERA:/tmp/`

## Run on device

Most tests assume:

- **MIPS/XBurst2** userspace
- `soc-nna` driver present (some tests need `/dev/soc-nna` and `/dev/mem`)
- run as **root** for the HW-init path

Suggested run order:

1. `./mxu_test`
   - Broad smoke test: MIR/MCSR, VPR SA0/LA0, bandwidth copy microbench.

2. `./mxuv3_sum_test`
   - Validates VSR SUMZ / MTSUM / MFSUM round-trips.

3. `./mxuv3_sa0_offset_test`
   - Confirms the **SA0 offset 0/1** behavior.
   - Probes offsets 2/3 (informational): FFmpeg currently assumes only 0/1 are reliable.

4. `./mxuv3_s4mac_test [iters]`
   - Validates `S4MACSSB` semantics (4x16B dot products) + a simple timing loop.

5. `./mxuv3_blockclear_bench [blocks] [iters]`
   - Benchmarks a FFmpeg-relevant primitive: **clearing 8x8 DCT blocks** using `VPR_ZERO + SA0`.

6. `./mxuv3_vpr_zero_test`
   - Validates whether `VPR_ZERO` actually clears VPR0 on this kernel.
   - If the first `VPR_ZERO` is swallowed while CU2 is being enabled, the second should still succeed.

7. `./mxuv3_f32_vec_test [n]`
   - Exercises Mars' `mxu_add_f32()` and documents whether **MXU compute requires NNA init** on this kernel.
   - Requires `/dev/mem` + `/dev/soc-nna` (root).

## How this maps to FFmpeg optimizations

- **Block clear / coefficient reset**
  - Directly corresponds to FFmpeg's `libavcodec/mips/blockdsp_mxu.c` (VPR_ZERO + SA0).
  - `mxuv3_blockclear_bench` helps justify enabling/expanding this path.

- **Dot-product style kernels**
  - `S4MACSSB` validates 8-bit MAC throughput for patterns that resemble filter taps / transforms.
  - If stable, we can look at FFmpeg primitives that boil down to repeated 8-bit MAC + horizontal reductions.

- **Alignment constraints**
  - These tests help confirm the operational boundary that FFmpeg must respect:
    **64-byte alignment** for fast/defined VPR load/store.
