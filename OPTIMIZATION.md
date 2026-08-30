# Optimizing the CfC cell on a Cortex-M4

How `cfc_step_backbone` went from 270,658 cycles to 169,912 on a NUCLEO-F401RE, a 37% reduction, and, more usefully, what was tried that didn't work and why.

Short version: the first three optimization rounds targeted 6% of the runtime, because a broken measurement setup pointed at the wrong thing. Everything below is measured on hardware via DWT cycle counting, never estimated.

---

## The workload

`cfc_step_backbone` is five dense matrix-vector products plus activations:

| layer | in | out | MACs |
|---|---|---|---|
| backbone | 38 | 128 | 4,864 |
| ff1, ff2, time_a, time_b | 128 | 32 | 4,096 each |
| total | | | 21,248 |

At a naive 1 cycle per multiply-accumulate that's about 21k cycles. The first measurement came back at 244k, 12x off. That gap is the whole story.

---

## Results

All measured on hardware, caches enabled, N=1000, `cfc_step_backbone`:

| `dense_layer` / `cfc_math` variant | cycles | cyc/MAC | vs. libm |
|---|---|---|---|
| newlib `tanhf`/`expf` activations | 270,658 | 12.74 | -- |
| Pade[4/3] activations | 248,012 | 11.67 | -8.4% |
| FMA contraction | 228,634 | 10.76 | -15.5% |
| 4 partial accumulators | 210,953 | 9.93 | -22.1% |
| FMA + 4x unrolling | 202,043 | 9.51 | -25.4% |
| partial accumulators + FMA (shipped) | 169,912 | 8.00 | -37.2% |

At the ~64MHz the debugger configures, 169,912 cycles is 2.65ms/step, about 377Hz. Every variant passes the golden-vector test against PyTorch at 1e-5 per timestep across both model configurations.

---

## What went wrong first

### 1. Every early measurement was invalid

The firmware never configured `FLASH_ACR`. It ran under whatever OpenOCD's `reset-init` handler left behind, which was `0x102`: prefetch on, instruction and data cache off. Confirmed by reading the register live on the target (`mdw 0x40023C00`).

So every cycle count for the first several rounds was measured with flash caching disabled, on a workload that streams 85KB of weights out of flash at 2 wait states. Enabling the caches (`FLASH_ACR = 0x702`) was worth about 76k cycles, far more than any code change made before or since.

This also explained an anomaly that had been sitting unexplained: every run reported `min == max == avg` exactly across 1000 calls. That looked like a broken benchmark (loop hoisting, misplaced DWT reads). It wasn't. With caches off, execution was fully dominated by deterministic flash stalls, so every iteration genuinely took identical time. With caches on, normal variation returned (a 58-cycle spread). The harness had been correct all along.

Lesson: verify the machine before optimizing the code.

### 2. Then three rounds aimed at the wrong 6%

With 224 transcendental calls per step (128 `lecun_tanh` + 64 `tanh` + 32 `sigmoid`), activations looked like the obvious bottleneck. They weren't.

**Round 1, continued fraction.** Replaced newlib `tanhf`/`expf` with a 7-term continued fraction. Passed the golden test cleanly. Measured 30% slower: 244k to 324k. Cause, confirmed by counting instructions in the disassembly: the expansion needs about 8 `vdiv.f32` per call, and Cortex-M4 single-precision divide is non-pipelined at about 14 cycles. The library call overhead it eliminated was cheaper than the divisions it added.

**Round 2, Pade[4/3].** Rational approximation, algebraically restructured to need exactly one division, verified by counting `vdiv.f32` in the binary. Coefficients derived from tanh's exact Taylor series via exact rational arithmetic, fit to the measured operating range (|x| <= 2.9 across all golden sequences, fit to 3.5 for margin, max error 3.2e-7). Removing 7 of 8 divisions recovered only about 16k cycles, the first hard evidence that divide latency was never the dominant cost.

**Round 3, the measurement that should have come first.** A deliberately wrong build with all three activations replaced by `return x;`. It fails the golden test by design; it exists only to measure everything that is not activation math.

```
stub (activations = return x;)   233,054
Pade[4/3]                        248,012
newlib tanhf/expf                270,658
```

Total activation cost: 14,964 cycles, 6.0% of runtime. Three rounds of work on six percent of the problem. The Pade work was real (it beats libm by 8.6%), but it was never going to matter much.

Lesson: measure the floor before optimizing a component. A stub build is cheap and would have redirected the entire effort on day one.

---

## Where the time actually went

Per-call DWT instrumentation around each of the five `dense_layer` calls, with the DWT read overhead itself calibrated and subtracted:

| call | MACs | cycles | cyc/MAC |
|---|---|---|---|
| backbone | 4,864 | 52,998 | 10.9 |
| ff1 | 4,096 | 47,627 | 11.6 |
| ff2 | 4,096 | 43,531 | 10.6 |
| time_a | 4,096 | 47,626 | 11.6 |
| time_b | 4,096 | 47,629 | 11.6 |
| total | 21,248 | 239,411 | |

96.5% of runtime is inside `dense_layer`, the one function nobody had profiled. For comparison: activations 6.0%, and moving all 85KB of weights from flash to SRAM was worth only 4.6% (not taken: it costs 84.5KB of a 96KB RAM budget, and it confirmed the data cache was already doing its job).

At about 11 cycles/MAC while executing only 6 instructions/MAC, the core was stalling, not instruction-bound.

---

## The fix

Two independent problems in one loop:

```c
for (int i = 0; i < in_dim; i++)
    acc += in[i] * w[o * in_dim + i];   // one accumulator, no FMA
```

**Problem 1, serial dependency chain.** Every iteration's add waits on the previous iteration's add to retire. Per the Cortex-M4 TRM, float arithmetic takes an extra cycle when its result is consumed by the next instruction, and multiply-accumulate consumes its addend a cycle later still. With one accumulator that penalty is paid on every MAC. Fix: four independent accumulators, so three FMAs can issue while one resolves.

**Problem 2, no fused multiply-add.** GCC emitted separate `vmul.f32` and `vadd.f32`, never `vfma.f32`, zero fused instructions in the whole binary. Not an algorithmic issue: `-std=c99` requests strict conformance, and FMA rounds once where a separate multiply and add round twice, so GCC won't fuse without `-ffp-contract=fast`.

The two are genuinely independent, which the isolated builds confirmed: partial accumulators alone (-22.1%) beat FMA alone (-15.5%) despite a worse instruction count, because breaking the stall mattered more than saving instructions. Combining them lands at -37.2%.

### The trap in between

The partial-accumulator variant silently stopped being inlined. Its larger body crossed GCC's inlining threshold, so instead of five inlined copies it became five real function calls, invisible in the source, invisible in the instruction-per-MAC count, and it made that build a confounded comparison (partial accumulators plus call overhead vs. an inlined baseline). Caught with `arm-none-eabi-nm`, fixed with `__attribute__((always_inline))` on the shipped version, and verified by symbol check rather than assumed.

Lesson: an optimization can change something you weren't measuring.

---

## Where the 8 cycles per MAC go

169,912 cycles is 8.00 cycles/MAC against a "1 cycle/MAC" naive ideal. That gap is fully accountable from documented instruction timings — no mystery cycles. Timings below are from the Cortex-M4 TRM (ARM DDI 0439, FPU instruction set table and instruction timing chapter), read from the document, not recalled:

| instruction | cycles |
|---|---|
| `VFMA.F32` (also `VMLA.F32`) | **3** |
| `VMUL.F32`, `VADD.F32` | 1 |
| `VLDR.32` | 2 (consecutive loads pipeline like integer `LDR`, ~1/cycle after the first) |
| `VDIV.F32` | 14 (retroactively confirms the continued-fraction post-mortem exactly) |
| integer ALU op | 1 |
| taken branch | 1 + P, pipeline refill P = 1–3 |

Plus two dependency rules: an FP result consumed by the *immediately following* instruction costs +1, and MAC instructions consume their addend one cycle late (which is what lets a dependent MAC chain partially off the hook).

**The first correction is to the ideal itself: 1 cycle/MAC was never achievable on this core.** `VFMA.F32` takes 3 cycles on a single-issue, in-order pipeline with a non-pipelined FPU multiply-accumulate. The arithmetic alone — before a single operand is loaded — costs 21,248 × 3 = **63,744 cycles**. That is 37% of the entire measured runtime, and it is documented, irreducible FMA execution time.

**Modeling the shipped inner loop.** One iteration (4 MACs, 17 instructions, from the disassembly): 8 `vldr` + 4 `vfma` into four independent accumulators + 4 integer ops + 1 `bne`.

| component | cycles |
|---|---|
| 8 loads (two 4-long bursts, pipelined) | ~9–10 |
| 4 × `VFMA` @ 3 | 12 |
| 4 integer loop-control ops | 4 |
| taken branch (1 + P) | 2–4 |
| **per iteration (4 MACs)** | **27–30** |

The step executes 5,248 of these iterations (4,096 across ff1/ff2/time_a/time_b + 1,152 in the backbone layer), plus 128 two-MAC remainder passes (in_dim=38 isn't a multiple of 4), 256 output-row prologues/epilogues (~18 cycles each: accumulator init, pairwise reduction + bias, store, outer loop), the measured 14,964-cycle activation cost, and memcpy/call glue:

| ledger | optimistic (P=1, loads 9) | middle (P=2, loads 10) | measured |
|---|---|---|---|
| inner iterations (5,248) | 141,696 | 152,192 | |
| remainders + row overhead + glue | ~6,300 | ~6,500 | |
| activations (measured) | 14,964 | 14,964 | |
| **total** | **163,204** | **173,700** | **169,912** |

The measured number sits inside the bracket, ~2% from either end. Working backwards, the real hardware averages ~28.3 cycles per iteration — branch refill averaging under 2 and loads pipelining well.

**The model postdicts the other variants, which is the strong evidence it's right:**

- **fma variant** (single accumulator, 1 MAC/iteration: 2 loads ~3 + `cmp` 1 + `vfma` 3 + `bne` 3 = 10 cycles/MAC): predicted 232,452, measured 228,634 — 1.7% error.
- **fma → shipped delta**: model predicts 10 − 29/4 = 2.75 cycles/MAC saved by amortizing loop control over 4-MAC iterations; measured (228,634 − 169,912)/21,248 = **2.76**. Agreement to 0.4%.
- **fma-unroll → shipped delta**: same instruction mix as shipped minus the independent chains; measured difference (202,043 − 169,912)/21,248 = **1.51 cycles/MAC**. That number *is* the per-MAC cost of the serial accumulator dependency, measured directly — the stall the partial accumulators removed.

**The floor, computed rather than guessed:** VFMA execution (63,744) + one weight load per MAC that is used exactly once and can never be amortized (≥21,248 pipelined, ~26k realistic) ≈ **85–90k cycles ≈ 4.0–4.2 cycles/MAC** for float32 with this data layout. The shipped 8.00 sits ~3.8 cycles/MAC above that floor, and every one of those cycles is attributed: input loads (~1.0), loop control + branch refill (~1.7), row overhead + remainders (~0.4), activations (~0.7).

Getting below ~90k therefore requires changing the arithmetic, not the loop — which is exactly what the next section concludes. (`SMLAD` does two int16 MACs in a single cycle: the arithmetic floor alone drops 6x.)

## What's left

The load count looked like the obvious next lever. ff1/ff2/time_a/time_b all read the same 128-element input, so that side isn't actually irreducible the way weight loads are (each weight is used exactly once, no way around that). It was tried and it didn't work. `src/experiments/cfc_fused.c` fuses those four layers into one pass: input loaded once per element, 4 MACs against it. Confirmed by disassembly to actually cut loads (2.0 to 1.25 per MAC, 17 to 12 instructions per 4 MACs), and measured identical to the shipped version on hardware, 169,912 cycles. Fusing across 4 layers stretches each layer's own accumulation from a 32-long chain (shipped) to 128-long, and that cost canceled the saved loads exactly. The binding constraint is FPU latency and the one unavoidable weight load per MAC, not load count in general. Reducing loads elsewhere doesn't help if it lengthens the dependency chain by a matching amount.

The floor for float32 matrix-vector on this core is ~4.0–4.2 cycles/MAC (computed above: 3.0 of documented VFMA execution + ~1.0–1.2 of irreducible weight loads), not 1. Two structural changes were tried near that floor — partial accumulators (won) and input fusion (wash) — which is most of what loop-level restructuring can offer here. Going further means changing the arithmetic, not the loop:

- Smaller data: int8/int16 with the M4's DSP extension (`SMLAD` does two MACs per instruction). Requires quantization and re-verification against the golden vectors at a looser but justified threshold.
- Fewer MACs: smaller `hidden_dim`/`backbone_units`. The cheapest option by far, and a model question rather than a code question.

---

## Verification

Every variant, including the rejected ones, was checked against golden vectors generated from the PyTorch reference (`ncps.torch.CfCCell`): 5 input sequences x 500 timesteps, compared at every timestep rather than only at the end, since recurrent error compounds. Threshold 1e-5 max absolute error; the shipped implementation runs about 3.8e-7, roughly 26x margin.

Both the partial accumulators (reassociation) and FMA (single vs. double rounding) change float rounding, so this mattered. "It's faster" is not a result until the numbers are still right.

One subtlety worth recording: `-ffp-contract=fast` alone does nothing on an x86-64 host, because the SSE2 baseline has no FMA instruction. The host test was verified by disassembly to contain zero fused instructions until `-mfma` was added, at which point the reported errors shifted slightly, proof it was finally exercising the same arithmetic the ARM target runs. `test/Makefile` now probes for FMA support and warns if the host lacks it.

---

## Summary

| | |
|---|---|
| Starting point | 270,658 cycles, 12.74 cyc/MAC |
| Shipped | 169,912 cycles, 8.00 cyc/MAC (-37.2%) |
| Wall clock at 64MHz | 4.23ms to 2.65ms (237Hz to 377Hz) |
| Time spent on 6% of the problem | 3 rounds |
| Time spent on 96.5% of the problem | 1 round |
| Optimizations that made things slower | 1 (30% slower, reverted) |
| Bugs found in the measurement setup | 2 (caches off; silent inlining loss) |

Reusable lessons, in the order they'd have saved the most time:

1. Verify the measurement environment before trusting any number. The caches-off bug invalidated every early measurement.
2. Measure the floor before optimizing a component. One stub build would have redirected three rounds of misdirected work.
3. Instruction counts are not cycle counts. The variant with worse instructions-per-MAC won, because stalls dominated.
4. A correctness test is not a performance test. Every rejected variant passed the golden test perfectly, including the one that was 30% slower.