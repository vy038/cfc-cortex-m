# Engineering notes

Working notes for this repo: build gotchas, hardware quirks, and rules that
exist because the corresponding mistake already happened once. Read this
before changing `dense_layer`, the build flags, or anything that produces a
number. [OPTIMIZATION.md](OPTIMIZATION.md) has the narrative; this file is
the operational version.

## What this is

C99 port of the CfC (closed-form continuous-time) recurrent cell from
`ncps.torch.CfCCell` (ncps 1.0.1, `mode="default"`), bit-verified against
PyTorch, benchmarked on a **NUCLEO-F401RE** (STM32F401RE, Cortex-M4F — not
an F407, see board note under Rules). Two configurations are ported as
**fully independent code paths**, not variants of one function:

- **backbone**: `backbone_layers=1`, `backbone_units=128`, `lecun_tanh` activation
- **no-backbone**: `backbone_layers=0`, ff1/ff2/time_a/time_b read straight from `concat(input, hx)`

Both use `input_dim=6`, `hidden_dim=32`, seed 42.

Current measured performance: **169,912 cycles/step** (backbone, caches on,
N=1000) = 8.00 cycles/MAC over 21,248 MACs, ~2.65ms / ~377Hz at 64MHz.
**[OPTIMIZATION.md](OPTIMIZATION.md) is the full story** — read it before
touching `dense_layer` or proposing performance work. It records what was
tried, what regressed, and why.

## Commands

```sh
# Python: generate weights + golden vectors (run from python/)
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python export.py              # backbone   -> ../weights/generated/{weights.h, golden.json}
python export_nobackbone.py   # no-backbone -> ../weights/generated/{weights_nobackbone.h, golden_nobackbone.json}

# Host verification (run from test/) — per-timestep, all 5 sequences, both configs
make run-all            # both
make run                # backbone only
make run-nobackbone     # no-backbone only

# NUCLEO-F401RE bench (run from bench/, needs arm-none-eabi-gcc)
make all-configs        # bench.elf/.bin + bench-nobackbone.elf/.bin
arm-none-eabi-size bench.elf bench-nobackbone.elf
make experiments        # optional: archived optimization/diagnostic builds
```

There is no single-test-case runner — `test_golden`/`test_golden_nobackbone`
always walk all 5 sequences × 500 timesteps and exit nonzero if any
sequence's max absolute error exceeds `1e-5`. To isolate one sequence, grep
the printed per-sequence line (`[step_a] max_abs_err=...`).

### Flashing and measuring

```
reset halt
flash write_image erase /path/to/bench.bin 0x08000000
reset init
arm semihosting enable
resume
```

Cycle counts require **real hardware with a debug probe attached**. QEMU
(`-M olimex-stm32-h405`) boots the binary and semihosting works — useful for
link/boot verification — but its Cortex-M4 model does not implement
`DWT->CYCCNT` at all (confirmed with `-d unimp`: PPB offsets 0x1000/0x1004
log as unassigned). Do not retry that path for timing.

## Architecture

**Weight/golden generation is one Python module per config**
(`python/export.py`, `python/export_nobackbone.py`), both importing shared
plumbing (`gen_sequences`, `run_golden`, `write_golden_json`, `flatten`,
`c_array`) from `export.py`. Regenerate whenever `python/train.py`'s
`build_model`/dims change — nothing downstream hardcodes dimensions.

**Generated headers are namespaced so both can be included in one
translation unit.** `weights.h` uses `CFC_*`, `weights_nobackbone.h` uses
`CFC_NB_*`. `src/cfc.h` includes both unconditionally, which is why the two
configs share one `cfc_state_t` (`CFC_HIDDEN_DIM == CFC_NB_HIDDEN_DIM`) and
live in one `src/cfc.c`.

**`src/cfc.c` has two independent step functions**, `cfc_step_backbone` and
`cfc_step_nobackbone`, deliberately not merged with an `#ifdef` or runtime
branch — mirroring the actual branch in `CfCCell.forward`. They share only
the `dense_layer` matrix-vector helper.

**`dense_layer` is the hot path — 96.5% of runtime, measured.** Its shape is
load-bearing; see the Rules section before editing it.

**`cfc_math.c` keeps `cfc_lecun_tanh` (backbone-only) and `cfc_tanh`
(ff1/ff2, both configs) as distinctly named functions**, not one function
with a mode flag — mixing them up is the easiest way to silently break
bit-exactness, since both are "a tanh" but not the same formula. All three
activations derive from one Padé[4/3] `fast_tanhf` core; `cfc_sigmoid` uses
`0.5 + 0.5*tanh(x/2)` so there is only one approximation to validate.

**Test harnesses are near-identical duplicates, not a shared parser**,
matching the fully-separate-code-path design of the two `cfc_step_*`
functions. Each is a hand-rolled scanner tailored to its own `golden*.json`
(fixed key order: name → inputs → ts → hidden_states → outputs; no general
JSON parsing, no external dependency) — this works only because `export*.py`
fully controls field order and byte format. The whitespace-independence
claim has been validated incidentally: the harness produced bit-identical
errors against both a compact and a pretty-printed copy of the same data.
It anchors on key names and reads a known float count, so indentation is
irrelevant — but it would equally not notice trailing garbage or a
duplicated payload, so `export*.py` remaining the only writer matters.

Both exports are **deterministic** — re-running `export.py` /
`export_nobackbone.py` reproduces `weights.h` and `golden*.json`
byte-identically. If a regenerated file differs, something changed upstream
(ncps version, seed, dims); investigate rather than committing the diff.

## Rules for this repo

### Correctness

- Whatever config Python exports is exactly what the matching C function
  ports. Never mix backbone-on weights with the no-backbone code path or
  vice versa — that's why the weight arrays are namespaced differently.
- Any equation claim about `CfCCell` must be checked against actual ncps
  source, never taken on faith from a prior summary (including this file):
  `python3 -c "import ncps.torch.cfc_cell as m; import inspect; print(inspect.getsource(m.CfCCell.forward))"`
  (inside `python/.venv`). An earlier pass trusted a paraphrased summary of
  this exact function; it contained an error, caught only by reading source.
- `test_golden*.c` must compare EVERY timestep, not just final output —
  recurrent error compounds, and the first-divergence timestep (printed when
  any value exceeds `1e-6`) is what localizes a bug.
- Before trusting any new golden/test harness, deliberately corrupt a known
  value in the golden JSON and confirm the test catches it and reports the
  correct sequence/timestep/field. A passing test is not a correct test.
- **Host test builds must match the firmware's float behavior.**
  `test/Makefile` carries `-ffp-contract=fast` *and* `-mfma`. On x86-64 the
  first flag alone emits **zero** fused instructions (SSE2 baseline has no
  FMA), so without `-mfma` the test silently verifies arithmetic the target
  never runs — confirmed by objdump. The Makefile probes `/proc/cpuinfo` and
  warns if the host can't fuse.

### Editing `dense_layer` — two things are load-bearing

1. **Four partial accumulators, not one.** A single accumulator is a serial
   dependency chain; the M4 pays an extra cycle when a float result is
   consumed by the next instruction. Four independent chains were worth ~15%
   on hardware — more than the instructions they cost. Collapsing them back
   silently costs that.
2. **`__attribute__((always_inline))`.** Without it the body is large enough
   that GCC stops inlining at `-O2` and emits 5 real calls per step. This
   happened silently once and confounded a measurement. After ANY edit,
   check `arm-none-eabi-nm bench.elf | grep dense_layer` — no symbol means
   still inlined. `always_inline` is a strong hint, not a guarantee.

Both this structure and `-ffp-contract=fast` change float rounding.
`test_golden` is the arbiter; current margin is ~26x under the 1e-5 threshold.

### Measurement discipline (learned the hard way — see OPTIMIZATION.md)

- **Get the floor before optimizing a component.** The stub build
  (`src/experiments/cfc_math_stub.c`, activations = `return x;`) proved
  activations were only 6% of runtime — after three rounds had already been
  spent on them. One stub build first would have redirected all of it.
- **Instruction count is not cycle count.** The variant with *worse*
  instructions-per-MAC won on hardware, because stalls dominated.
- **A correctness test is not a performance test.** Every rejected variant
  passed `test_golden` perfectly, including one that was 30% slower.
- Sanity-check any cycle count against a rough MAC-count estimate before
  trusting it. `DWT->CYCCNT` only increments while a debug probe is attached
  (the debug power domain gates it), and a frozen counter reads as a
  plausible small number, not an obvious error.

### Hardware / build environment

- **Target board is NUCLEO-F401RE (STM32F401RE)**, not an F407 as early
  planning assumed: 512K flash / **96K SRAM** (not 128K — corrected in
  `bench/stm32f4.ld`'s `RAM LENGTH`). Still Cortex-M4F with FPU, so the
  `-mfloat-abi=hard -mfpu=fpv4-sp-d16` build is unaffected. Any clock-tree
  or peripheral code added later must not assume F407 behavior (F401 has no
  Ethernet/camera and different timers).
- **The firmware sets `FLASH_ACR = 0x00000702`** (2 wait states + PRFTEN +
  ICEN + DCEN) as the first statement in `main()`. Previously nothing
  touched it and the chip ran under OpenOCD's reset-init value `0x102` —
  **both caches off** — which invalidated every early measurement. After
  flashing, confirm with telnet `mdw 0x40023C00`; expect `0x00000702`. If it
  reads `0x102` the write was clobbered and the measurement is invalid.
- **RCC is still not configured by firmware.** The clock is whatever OpenOCD
  sets (~64MHz); standalone without a debugger this boots to the 16MHz HSI
  reset default, ~4x slower than anything measured. Explicit RCC setup is
  outstanding and is a hard prerequisite for Stride integration.
- `bench/Makefile` uses `-ffunction-sections -fdata-sections` +
  `-Wl,--gc-sections` so the linker drops the unused `cfc_step_*` and its
  weight arrays. Without them both configs pull in both weight sets and
  `.text` is meaningless for comparison (this bit the project once). With
  them, flash deltas are valid. `test/Makefile` doesn't need them.
- `bench/Makefile` keeps `CFLAGS_BASE` (pre-FMA flags) separate from
  `CFLAGS` **only** so archived experiments reproduce their recorded
  numbers. Don't collapse them.

## Layout

```
python/    train.py (model def, both configs) + export.py + export_nobackbone.py
src/       cfc.c/h (both cell variants), cfc_math.c/h (Padé[4/3] activations)
           experiments/  archived variants + README (not built by default)
test/      test_golden.c + test_golden_nobackbone.c, plain gcc, hand-rolled JSON scanners
bench/     NUCLEO-F401RE: main.c/main_nobackbone.c, startup.s, stm32f4.ld
           experiments/  diagnostic harnesses + README (not built by default)
weights/generated/   weights.h, weights_nobackbone.h, golden.json, golden_nobackbone.json
OPTIMIZATION.md      the measurement/optimization story — read before perf work
```

## Measurement history (don't redo these)

All on hardware, caches on, N=1000, `cfc_step_backbone`, 21,248 MACs:

| variant | cycles | cyc/MAC | note |
|---|---|---|---|
| newlib `tanhf`/`expf` | 270,658 | 12.74 | original activations |
| Padé[4/3] activations | 248,012 | 11.67 | |
| stub (activations = `return x;`) | 233,054 | 10.97 | floor: proved activations are 6% |
| FMA contraction only | 228,634 | 10.76 | |
| 4 partial accumulators only | 210,953 | 9.93 | beat FMA despite worse instr/MAC |
| FMA + 4x unroll | 202,043 | 9.51 | |
| **partial acc + FMA (shipped)** | **169,912** | **8.00** | |

Rejected: continued-fraction `tanh` (~8 `vdiv.f32`/call) measured 323,712 —
**30% slower** than the library it replaced; M4 float divide is
non-pipelined at ~14 cycles. Deleted, not archived.

Also settled: moving all 85KB of weights to SRAM is worth only 4.6% and
costs 84.5KB of a 96KB budget — not taken. Measurement is reproducible to
~50 cycles (999/1000 calls identical; only the first call differs, +49,
cold cache).

**Remaining headroom is load bandwidth, not loop structure.** Two loads per
MAC, nothing reusable in a matrix-vector product; a realistic floor here is
~3-4 cycles/MAC, not 1. Going further means blocking for data reuse,
int8/SMLAD quantization, or fewer MACs — different projects, see
OPTIMIZATION.md's closing section.
