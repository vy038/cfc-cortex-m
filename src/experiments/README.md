# Archived experiment sources

**None of this is built by default.** The shipping implementation is
`src/cfc.c` + `src/cfc_math.c`. These files exist so the measurements in
[`../../OPTIMIZATION.md`](../../OPTIMIZATION.md) can be reproduced, and so
the rejected approaches are visible rather than lost.

Each file is a full drop-in replacement for either `cfc.c` or `cfc_math.c`
(same API, so exactly one links into any binary — linking two is a
duplicate-symbol error, which is a deliberate safety property). Build them
via `make experiments` in `bench/`.

## `cfc.c` replacements — the dense_layer optimization sequence

| file | what it is | measured |
|---|---|---|
| `cfc_baseline.c.bak` | the original single-accumulator `dense_layer` | 248012 cyc |
| `cfc_partial_acc.c` | 4 partial accumulators, no FMA flag | 210953 cyc |
| `cfc_unroll.c` | `#pragma GCC unroll 4`, single accumulator | 202043 cyc (with FMA) |
| `cfc_fused.c` | ff1/ff2/time_a/time_b share one pass over the input: each element loaded once, 4 MACs against it (`dense_layer4`). Input-side loads drop 4→1 per 4 MACs; 12 instr/4 MACs vs shipped 17. | 169912 cyc — **identical to shipped, no gain** |

`.bak` on the baseline is deliberate: it keeps the old implementation from
reading as live source. `bench/Makefile` passes `-x c` to compile it anyway.

`cfc_fused.c` is a documented **negative result**: the load-count reduction
it targeted was real (confirmed by disassembly) but didn't move the cycle
count at all. Fusing across 4 layers stretches each layer's own
accumulation from a 32-long chain to 128-long, and that cost canceled the
saved loads exactly. See NOTES.md's "Fused-input variant" entry for the
full reasoning — worth reading before trying a similar fusion idea again.

The winner — partial accumulators **+** FMA **+** `always_inline`, at
169912 cycles — is not here. It was promoted to `src/cfc.c`.

## `cfc_math.c` replacements — the activation study

| file | what it is | measured |
|---|---|---|
| `cfc_math_pade.c.bak` | snapshot of the Padé[4/3] implementation | (same as current `cfc_math.c`) |
| `cfc_math_libm.c` | original newlib `tanhf`/`expf` | 270658 cyc |
| `cfc_math_stub.c` | **deliberately wrong**: all activations `return x;` | 233054 cyc |

`cfc_math_stub.c` is a timing floor, not an implementation: it measures
everything that is *not* activation math. It fails `test_golden` by design
and emits a `#warning` on every build so it can never be mistaken for a
real one. Its number is what proved activations were only 6% of runtime.

## Not archived here

The continued-fraction `tanh` (7 terms, ~8 divisions per call) was measured
at 323712 cycles — 30% *slower* than the library it replaced — and deleted
rather than kept. Its story is in `OPTIMIZATION.md`; the code isn't worth
preserving.
