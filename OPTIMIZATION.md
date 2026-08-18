# Optimizing the CfC cell on a Cortex-M4

How `cfc_step_backbone` went from 270,658 cycles to 169,912 on a
NUCLEO-F401RE — a 37% reduction — and, more usefully, what was tried that
didn't work and why.

The short version: **the first three optimization rounds targeted 6% of the
runtime, because a broken measurement setup pointed at the wrong thing.**
Everything below is measured on hardware via DWT cycle counting, never
estimated.

---

## The workload

`cfc_step_backbone` is five dense matrix-vector products plus activations:

| layer | in | out | MACs |
|---|---|---|---|
| backbone | 38 | 128 | 4,864 |
| ff1, ff2, time_a, time_b | 128 | 32 | 4,096 each |
| **total** | | | **21,248** |

At a naive 1 cycle per multiply-accumulate that's ~21k cycles. The first
measurement came back at 244k — **12x** off. That gap is the whole story.

---

## Results

All measured on hardware, caches enabled, N=1000, `cfc_step_backbone`:

| `dense_layer` / `cfc_math` variant | cycles | cyc/MAC | vs. libm |
|---|---|---|---|
| newlib `tanhf`/`expf` activations | 270,658 | 12.74 | — |
| Padé[4/3] activations | 248,012 | 11.67 | −8.4% |
| FMA contraction | 228,634 | 10.76 | −15.5% |
| 4 partial accumulators | 210,953 | 9.93 | −22.1% |
| FMA + 4x unrolling | 202,043 | 9.51 | −25.4% |
| **partial accumulators + FMA (shipped)** | **169,912** | **8.00** | **−37.2%** |

At the ~64MHz the debugger configures, 169,912 cycles is **2.65ms/step, or
~377Hz**. Every variant passes the golden-vector test against PyTorch at
1e-5 per timestep across both model configurations.

---

## What went wrong first

### 1. Every early measurement was invalid

The firmware never configured `FLASH_ACR`. It ran under whatever OpenOCD's
`reset-init` handler left behind, which was `0x102`: prefetch on,
**instruction and data cache off**. Confirmed by reading the register live
on the target (`mdw 0x40023C00`).

So every cycle count for the first several rounds was measured with flash
caching disabled, on a workload that streams 85KB of weights out of flash
at 2 wait states. Enabling the caches (`FLASH_ACR = 0x702`) was worth ~76k
cycles — far more than any code change made before or since.

This also explained an anomaly that had been sitting unexplained: every run
reported `min == max == avg` **exactly** across 1000 calls. That looked like
a broken benchmark (loop hoisting, misplaced DWT reads). It wasn't — with
caches off, execution was fully dominated by deterministic flash stalls, so
every iteration genuinely took identical time. With caches on, normal
variation returned (a 58-cycle spread). The harness had been correct all
along.

**Lesson: verify the machine before optimizing the code.**

### 2. Then three rounds aimed at the wrong 6%

With 224 transcendental calls per step (128 `lecun_tanh` + 64 `tanh` + 32
`sigmoid`), activations looked like the obvious bottleneck. They weren't.

**Round 1 — continued fraction.** Replaced newlib `tanhf`/`expf` with a
7-term continued fraction. Passed the golden test cleanly. Measured **30%
slower**: 244k → 324k. Cause, confirmed by counting instructions in the
disassembly: the expansion needs ~8 `vdiv.f32` per call, and Cortex-M4
single-precision divide is *non-pipelined* at ~14 cycles. The library call
overhead it eliminated was cheaper than the divisions it added.

**Round 2 — Padé[4/3].** Rational approximation, algebraically restructured
to need exactly **one** division, verified by counting `vdiv.f32` in the
binary. Coefficients derived from tanh's exact Taylor series via exact
rational arithmetic, fit to the measured operating range (|x| ≤ 2.9 across
all golden sequences, fit to 3.5 for margin, max error 3.2e-7). Removing 7
of 8 divisions recovered only ~16k cycles — the first hard evidence that
divide latency was never the dominant cost.

**Round 3 — the measurement that should have come first.** A deliberately
*wrong* build with all three activations replaced by `return x;`. It fails
the golden test by design; it exists only to measure everything that is
**not** activation math.

```
stub (activations = return x;)   233,054
Padé[4/3]                        248,012
newlib tanhf/expf                270,658
```

**Total activation cost: 14,964 cycles — 6.0% of runtime.** Three rounds of
work on six percent of the problem. The Padé work was real (it beats libm by
8.6%), but it was never going to matter much.

**Lesson: measure the floor before optimizing a component.** A stub build is
cheap and would have redirected the entire effort on day one.

---

## Where the time actually went

Per-call DWT instrumentation around each of the five `dense_layer` calls,
with the DWT read overhead itself calibrated and subtracted:

| call | MACs | cycles | cyc/MAC |
|---|---|---|---|
| backbone | 4,864 | 52,998 | 10.9 |
| ff1 | 4,096 | 47,627 | 11.6 |
| ff2 | 4,096 | 43,531 | 10.6 |
| time_a | 4,096 | 47,626 | 11.6 |
| time_b | 4,096 | 47,629 | 11.6 |
| **total** | 21,248 | **239,411** | |

**96.5% of runtime is inside `dense_layer`** — the one function nobody had
profiled. For comparison: activations 6.0%, and moving all 85KB of weights
from flash to SRAM was worth only 4.6% (not taken: it costs 84.5KB of a
96KB RAM budget, and it confirmed the data cache was already doing its job).

At ~11 cycles/MAC while executing only 6 instructions/MAC, the core was
**stalling, not instruction-bound**.

---

## The fix

Two independent problems in one loop:

```c
for (int i = 0; i < in_dim; i++)
    acc += in[i] * w[o * in_dim + i];   // one accumulator, no FMA
```

**Problem 1 — serial dependency chain.** Every iteration's add waits on the
previous iteration's add to retire. Per the Cortex-M4 TRM, float arithmetic
takes an extra cycle when its result is consumed by the next instruction,
and multiply-accumulate consumes its addend a cycle later still. With one
accumulator that penalty is paid on *every* MAC. Fix: four independent
accumulators, so three FMAs can issue while one resolves.

**Problem 2 — no fused multiply-add.** GCC emitted separate `vmul.f32` +
`vadd.f32`, never `vfma.f32` — zero fused instructions in the whole binary.
Not an algorithmic issue: `-std=c99` requests strict conformance, and FMA
rounds once where a separate multiply and add round twice, so GCC won't fuse
without `-ffp-contract=fast`.

The two are genuinely independent, which the isolated builds confirmed:
partial accumulators alone (−22.1%) **beat** FMA alone (−15.5%) despite a
*worse* instruction count, because breaking the stall mattered more than
saving instructions. Combining them lands at −37.2%.

### The trap in between

The partial-accumulator variant silently **stopped being inlined**. Its
larger body crossed GCC's inlining threshold, so instead of five inlined
copies it became five real function calls — invisible in the source,
invisible in the instruction-per-MAC count, and it made that build a
confounded comparison (partial accumulators *plus* call overhead vs. an
inlined baseline). Caught with `arm-none-eabi-nm`, fixed with
`__attribute__((always_inline))` on the shipped version, and verified by
symbol check rather than assumed.

**Lesson: an optimization can change something you weren't measuring.**

---

## What's left

8.0 cycles/MAC against a naive 1.0 ideal still looks bad, but that ideal
assumes a fully pipelined FMA issuing every cycle with operands already in
registers. Per MAC, the real cost is:

- **Two loads.** A matrix-vector product touches each weight exactly once,
  so nothing is reusable across iterations. Eight of ~16 instructions in the
  inner loop are `vldr`. On a single-issue core, a load cycle is a cycle not
  spent computing. This is a **load-bandwidth bound**.
- **FPU latency not fully hidden**, even across four chains.
- **Loop control**, ~4 of 16 instructions doing no arithmetic.

A realistic floor for float32 matrix-vector on this core is roughly **3–4
cycles/MAC**, not 1. Most of the gap that loop-level work can close has been
closed. Going meaningfully further means changing the problem, not the loop:

- **Fewer loads per MAC** — block the computation to reuse a loaded
  activation across several output rows. The current row-major weight layout
  doesn't allow it; this is a data-layout change.
- **Smaller data** — int8/int16 with the M4's DSP extension (`SMLAD` does
  two MACs per instruction). Requires quantization and re-verification
  against the golden vectors at a looser but justified threshold.
- **Fewer MACs** — smaller `hidden_dim`/`backbone_units`. The cheapest
  option by far, and a model question rather than a code question.

---

## Verification

Every variant, including the rejected ones, was checked against golden
vectors generated from the PyTorch reference (`ncps.torch.CfCCell`): 5
input sequences × 500 timesteps, compared **at every timestep** rather than
only at the end, since recurrent error compounds. Threshold 1e-5 max
absolute error; the shipped implementation runs ~3.8e-7, about 26x margin.

Both the partial accumulators (reassociation) and FMA (single vs. double
rounding) change float rounding, so this mattered — "it's faster" is not a
result until the numbers are still right.

One subtlety worth recording: `-ffp-contract=fast` alone does **nothing** on
an x86-64 host, because the SSE2 baseline has no FMA instruction. The host
test was verified by disassembly to contain zero fused instructions until
`-mfma` was added, at which point the reported errors shifted slightly —
proof it was finally exercising the same arithmetic the ARM target runs.
`test/Makefile` now probes for FMA support and warns if the host lacks it.

---

## Summary

| | |
|---|---|
| Starting point | 270,658 cycles, 12.74 cyc/MAC |
| Shipped | 169,912 cycles, 8.00 cyc/MAC (−37.2%) |
| Wall clock @64MHz | 4.23ms → 2.65ms (237Hz → 377Hz) |
| Time spent on 6% of the problem | 3 rounds |
| Time spent on 96.5% of the problem | 1 round |
| Optimizations that made things slower | 1 (30% slower, reverted) |
| Bugs found in the measurement setup | 2 (caches off; silent inlining loss) |

The reusable lessons, in the order they'd have saved the most time:

1. **Verify the measurement environment before trusting any number.** The
   caches-off bug invalidated every early measurement.
2. **Measure the floor before optimizing a component.** One stub build
   would have redirected three rounds of misdirected work.
3. **Instruction counts are not cycle counts.** The variant with worse
   instructions-per-MAC won, because stalls dominated.
4. **A correctness test is not a performance test.** Every rejected variant
   passed the golden test perfectly, including the one that was 30% slower.
