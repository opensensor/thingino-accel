# MXUv3: Organic Findings vs Later Discoveries

**Status:** Internal comparison note. This document contrasts what we learned empirically on T41 ("organic findings") with later discoveries that refined or corrected that picture.

## 1. Scope

This document focuses on MXUv3 as used on Ingenic T41 / XBurst2, and compares:

- **Organic findings** – what we inferred from RE of Venus, experiments, and Mars bring‑up.
- **Later discoveries** – clarifications that arrived afterwards (e.g. from broader ISA context, new experiments, or better reading of the existing binaries).

The goal is to capture the *delta* so that other docs (e.g. `mxuv3_instructions.md`, Mars code) can be interpreted with the right mental model.

---

## 2. Programming Model: Registers & Data Paths

### 2.1 Organic findings

- MXU has **32 VPR registers**, each **512 bits** (64 bytes).
- VPRs can be treated as:
  - 64×int8, 32×int16, 16×int32 or 16×float32 lanes.
- Special roles were observed:
  - **VPR29 / VPR30** appear as weight sources for MAC microcode.
  - **VPR6** looks special in some encodings (possible control usage).
- Only the VPR file was considered when looking for MAC results.

### 2.2 Later discoveries (delta)

- There is a **separate sum register cluster**, **VSR0–VSR3**, used as accumulators for MAC‑style instructions.
- There is a **vector write register file (VWR)** which is effectively a word‑level alias of **VPR31** (`vw0`, `vw1`, ... inside `vr31`).
- MAC results are expected to live in **VSR**, not in ordinary VPRs; VPR29/30 are better understood as *weight sources* rather than the place where accumulated sums end up.

**Impact:**

- The earlier statement that “MAC results live in internal accumulators not visible via VPR load/store” was incomplete.
- We now explain MAC behavior in terms of **VPR (data)** + **VSR (accumulators)** + **VWR (write‑port view of VPR31)**, which matches how Venus actually uses the hardware.

---

## 3. MAC Operations and Sum Access

### 3.1 Organic findings

- Identified a paired MAC sequence using c2 `func=0x0b` and `func=0x23` with VPR29/VPR30 as weight sources.
- Observed:
  - Int8→int16 and int8→float32 conversions into specific VPRs (e.g. VPR9/11/13).
  - No obvious “final sums” in any VPR after the MAC sequence.
- Concluded that MAC results must be in hidden, inaccessible accumulators.

### 3.2 Later discoveries (delta)

- The “hidden accumulators” are concretely modeled as **VSR0–VSR3**.
- There is a family of **sum access instructions** (e.g. `SUM*`, `MFSUM*`, `MXSUM*`) that move values between **VSR** and **VPR/GPR**.

**Impact:**

- The MAC pair (`0x0b/0x23`) can be described as:
  - Reading inputs/weights from VPRs (notably VPR29/VPR30) and accumulating into VSR.
  - Final outputs are obtained only if software explicitly reads from the **sum registers**, not by re‑loading VPRs.
- This clarifies why pure LA/SA probing around the MAC sequence never showed the full accumulated results.

---

## 4. Load/Store and Memory Behaviour

### 4.1 Organic findings

- Reverse‑engineered **LA0/SA0** encodings:
  - LA0: 256‑bit half‑loads; two instructions per 512‑bit VPR.
  - SA0: 256‑bit half‑stores; two instructions per 512‑bit VPR.
- Determined that:
  - Memory must be **64‑byte aligned**.
  - In early tests, **stack memory often gave partial results**, whereas buffers allocated via the NNA driver worked reliably.
- Initial docs said that **NNA‑allocated memory was required** for full functionality.

### 4.2 Later discoveries (delta)

- The load/store model is a **family** of LA/SA/LU/SU variants; LA0/SA0 are just specific encodings within that family.
- The “NNA‑allocated only” requirement is better understood as a **T41 platform quirk**:
  - The MXU3 ISA itself does not mandate NNA‑allocated buffers.
  - On T41, cache‑coherency / aliasing / driver integration can make plain stack/heap buffers behave unpredictably, while NNA‑managed buffers happen to do the right thing.

**Impact:**

- `mxuv3_instructions.md` has been updated to describe:
  - 64‑byte alignment as a requirement we have empirically validated.
  - NNA‑managed buffers as *recommended* for the current T41 platform, without treating them as an architectural requirement.

---

## 5. COP2 Vector Arithmetic (ADD / SUB / MUL)

### 5.1 Organic findings

- From libvenus disassembly we inferred a COP2 encoding providing:
  - **rs = 20, fn = 3/11** → vector **ADD/SUB** on 16×float32 lanes.
  - **rs = 19, fn = 35** → vector **MUL** on 16×float32 lanes.
- Implemented this as macros and helpers in `include/mxuv3.h`:
  - `VPR_ADD`, `VPR_SUB`, `VPR_MUL`, plus unary `VPR_SQR`, `VPR_DBL`, `VPR_ZERO`.

### 5.2 Later discoveries (delta)

- These encodings line up with a **larger, structured FP vector arithmetic set**, including extra ops like complex mul/add and max/min.
- The earlier interpretation (ADD/SUB/MUL over 16×float32) was correct, but we now understand it as part of a **broader MXU3 FP class**, not ad‑hoc encodings.

**Impact:**

- Our `mxuv3.h` API already matches the real semantics well; the main delta is that we can now reason about these ops in terms of a more complete FP opcode class and extend the API if needed (e.g. for max/min, complex mul).

---

## 6. Floating‑Point Division / Sqrt

### 6.1 Organic findings

- Scanning libvenus showed:
  - Many scalar FPU `div.s` and `sqrt.s` instructions.
  - No MXU opcodes resembling reciprocal/divide/rsqrt.
- Concluded that MXUv3 **does not offer vector div/sqrt/rsqrt**, and that models work around this by:
  - Precomputing reciprocals / inverse std‑dev during conversion.
  - Using MXU MUL for runtime work.

### 6.2 Later discoveries (delta)

- The broader MXU3 instruction set confirms that there are **vector add/sub/mul and friends**, but **no vector div/sqrt/rsqrt**.

**Impact:**

- Our earlier guidance for Mars remains valid:
  - Use MXU for MUL/ADD/SUB with pre‑baked constants.
  - Rely on scalar FPU or offline precomputation for division‑like work.

---

## 7. Neural‑Network Specific Instructions

### 7.1 Organic findings

- Treated the NNA as a separate block, driven via:
  - NNDMA/AIP MMIO ranges (`__aie_mmap`, driver ioctls, etc.).
  - Venus’ host‑side code writing descriptors and polling status.
- Kernel tracer (`nna-aip-trace.ko`) observed NNDMA and AIP IO activity but there was no explicit ISA‑level naming.

### 7.2 Later discoveries (delta)

- MXU3 includes **neural‑network oriented instructions** (e.g. NNR/NNW/NNMAC‑style ops) that conceptually tie the vector unit to the NNA hardware.
- These provide a more direct ISA view of what we previously thought of purely as "driver + MMIO" behaviour.

**Impact:**

- For Mars and future RE work, it makes sense to think of NNA control as an extension of the MXU3 execution model, not only as raw MMIO writes.
- Mapping these NN instructions to the concrete NNDMA/AIP behaviour seen by the tracer is an open follow‑up task.

---

## 8. Summary

- The **organic findings** around VPR layout, basic LA/SA encodings, and COP2 ADD/SUB/MUL were largely correct and directly usable.
- The **deltas** from later discoveries mainly refine the mental model:
  - Introduce **VSR** and **VWR** as first‑class pieces of the state.
  - Explain where MAC results live and how to read them back.
  - Clarify that some quirks (like “NNA‑allocated only” memory) are platform‑specific rather than architectural.
  - Recognize NN‑specific instructions and a richer FP vector class around the pieces we already use.

Going forward, `mxuv3_instructions.md` and the Mars MXU code should be interpreted with this richer model in mind, and any new RE should be written in terms of VPR/VSR/VWR plus the NN‑oriented instruction set, not just raw hex patterns.

