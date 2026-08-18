# cfc-cortex-m

![C](https://img.shields.io/badge/C99-bare_metal-555?style=flat-square&logo=c&logoColor=white)
![Cortex-M4](https://img.shields.io/badge/Cortex--M4-STM32F401-555?style=flat-square)
![Verified](https://img.shields.io/badge/verified-bit--exact_vs_PyTorch-555?style=flat-square)
![Status](https://img.shields.io/badge/status-done-green?style=flat-square)

**MIT's CfC liquid neural network, hand-ported to C99 and running on a $15
microcontroller — bit-verified against PyTorch and profiled down to the
instruction.**

| | |
|---|---|
| **Cycles/step** | **169,912** (2.65ms → ~377Hz @ 64MHz) |
| **Speedup vs. first working version** | **37%** (12.74 → 8.00 cycles/MAC) |
| **Accuracy vs. PyTorch** | **3.8e-7** max abs error, every timestep (threshold 1e-5) |
| **Profiled hot path** | `dense_layer` — 96.5% of runtime, measured per-call |
| **Notable rejected approach** | continued-fraction `tanh` — **30% slower**, reverted |

Runs bare metal: no dynamic allocation, no RTOS, `math.h` only. Every number
above was measured on real hardware with the DWT cycle counter, not estimated.

📄 [**OPTIMIZATION.md**](OPTIMIZATION.md) — how it got 37% faster, what
failed, and the measurement bug that sent three rounds of work at 6% of the
problem
📋 [**NOTES.md**](NOTES.md) — build gotchas, hardware quirks, and measurement
rules learned the hard way

---

A C99 port of the CfC (closed-form continuous-time) recurrent cell from
[ncps](https://github.com/mlech26l/ncps) (`ncps.torch.CfCCell`, mode
`"default"`), verified against PyTorch and benchmarked on a
NUCLEO-F401RE (STM32F401RE, Cortex-M4F) target. Two configurations are ported as fully
independent code paths: **backbone** (`backbone_layers=1`, 128 units,
lecun_tanh) and **no-backbone** (`backbone_layers=0`).

## CfC equations (as implemented)

Extracted directly from `ncps/torch/cfc_cell.py` (ncps 1.0.1),
`CfCCell.forward`, `mode="default"`:

```python
x = torch.cat([input, hx], 1)
if self.backbone_layers > 0:
    x = self.backbone(x)
ff1 = self.tanh(self.ff1(x))
ff2 = self.tanh(self.ff2(x))
t_interp = self.sigmoid(self.time_a(x) * ts + self.time_b(x))
new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2
return new_hidden, new_hidden
```

### Backbone config (`backbone_layers=1`, `backbone_units=128`, `backbone_activation="lecun_tanh"`)

```
x          = concat(input, hx)                       # (input_dim + hidden_dim)
backbone   = lecun_tanh(backbone_W @ x + backbone_b)  # lecun_tanh(z) = 1.7159 * tanh(0.666 * z)
ff1        = tanh(ff1_W @ backbone + ff1_b)           # standard tanh, NOT lecun_tanh
ff2        = tanh(ff2_W @ backbone + ff2_b)           # standard tanh, NOT lecun_tanh
t_a        = time_a_W @ backbone + time_a_b
t_b        = time_b_W @ backbone + time_b_b
t_interp   = sigmoid(t_a * ts + t_b)
new_hidden = ff1 * (1 - t_interp) + t_interp * ff2
output     = new_hidden   # CfCCell returns (new_hidden, new_hidden)
```

`lecun_tanh` is used **only** inside the backbone activation. `ff1`/`ff2`
and the time-gate sigmoid use standard `tanh`/`sigmoid`. `ts` is a scalar
timespan for the step (broadcast across the hidden dimension), not a
per-unit value.

### No-backbone config (`backbone_layers=0`)

The `if self.backbone_layers > 0: x = self.backbone(x)` branch in
`CfCCell.forward` is skipped entirely, so `x` stays the raw
`concat(input, hx)` and feeds directly into ff1/ff2/time_a/time_b
(`cat_dim = input_dim + hidden_dim`, no 128-wide intermediate stage):

```
x          = concat(input, hx)          # fed DIRECTLY into ff1/ff2/time_a/time_b
ff1        = tanh(ff1_W @ x + ff1_b)
ff2        = tanh(ff2_W @ x + ff2_b)
t_a        = time_a_W @ x + time_a_b
t_b        = time_b_W @ x + time_b_b
t_interp   = sigmoid(t_a * ts + t_b)
new_hidden = ff1 * (1 - t_interp) + t_interp * ff2
output     = new_hidden
```

Dimensions used in this project (both configs): `input_dim=6`, `hidden_dim=32`;
backbone config additionally uses `backbone_units=128`.

## Layout

```
python/    ncps/PyTorch model definitions + export to C headers + golden vectors
           export.py            -> backbone config
           export_nobackbone.py -> no-backbone config
src/       the C99 cell (cfc.c/h, both configs) and math helpers (cfc_math.c/h)
           experiments/  archived optimization variants, not built by default
test/      host-side verification against PyTorch golden vectors (plain gcc)
bench/     NUCLEO-F401RE cycle-count benchmarks (arm-none-eabi-gcc)
           experiments/  diagnostic/profiling harnesses, not built by default
weights/generated/   generated headers/JSON, produced by python/export*.py
```

The two `experiments/` directories hold the rejected and superseded variants
behind the numbers in [OPTIMIZATION.md](OPTIMIZATION.md), kept so the results
can be reproduced. Neither is part of a normal build; each has a README
explaining what its files measured.

`weights/generated/weights.h` (backbone, `CFC_*` names) and
`weights/generated/weights_nobackbone.h` (no-backbone, `CFC_NB_*` names) are
both included unconditionally by `src/cfc.h`, so the two weight sets never
collide in the same translation unit. They are the single source of truth
for dimensions — nothing in `src/` or `test/` hardcodes them. `cfc_state_t`
is shared by both configs since `CFC_HIDDEN_DIM == CFC_NB_HIDDEN_DIM`.

`src/cfc.c` implements `cfc_step_backbone` and `cfc_step_nobackbone` as two
fully independent functions (no shared body / `#ifdef` branching) — the
only thing they share is the generic `dense_layer` matrix-vector helper,
same as the original single-config version did internally.

## Usage

### 1. Generate weights + golden vectors

```sh
cd python
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python export.py              # backbone config -> weights.h, golden.json
python export_nobackbone.py   # no-backbone config -> weights_nobackbone.h, golden_nobackbone.json
```

### 2. Host verification (compares every timestep, not just the final one)

```sh
cd test
make run-all          # both configs
make run              # backbone only
make run-nobackbone   # no-backbone only
```

Reports max absolute error and max relative error per sequence, and the
first timestep where any value diverges by more than `1e-6` (recurrent
error compounds, so the first divergence point is what localizes a bug).
Exits nonzero if any sequence's max absolute error exceeds `1e-5`.

These builds deliberately match the firmware's float behavior
(`-ffp-contract=fast`, plus `-mfma` on x86 hosts — without it GCC emits no
fused instructions at all and the test would silently verify arithmetic the
target never runs). The Makefile probes for FMA support and warns if the
host lacks it.

### 3. NUCLEO-F401RE benchmark

```sh
cd bench
make all-configs      # both configs
arm-none-eabi-size bench.elf bench-nobackbone.elf
make experiments      # optional: the archived optimization/diagnostic builds
```

`make` alone builds only `bench.elf`/`.bin` (backbone); `make bench-nobackbone`
builds `bench-nobackbone.elf`/`.bin`. Each `main*.c` uses the Cortex-M4 DWT
cycle counter to report min/max/avg cycles for 1000 calls to its
`cfc_step_*` function, printed over ARM semihosting
(`initialise_monitor_handles()` + `printf`), so no UART driver is needed.
Targets the actual board's memory map — NUCLEO-F401RE / STM32F401RE
(512K flash / **96K RAM**, not the 128K an F407 would have) — in
`stm32f4.ld`.

Confirmed working under QEMU for boot/link verification (semihosting
output, correct execution):

```sh
qemu-system-arm -M olimex-stm32-h405 -cpu cortex-m4 -kernel bench.elf -semihosting -nographic
```

**QEMU's Cortex-M4 model does not implement `DWT->CYCCNT`** on this
machine (confirmed via `-d unimp`), so no cycle counts are obtainable
under emulation — this is a hard blocker, not a config issue. Cycle
counts require real hardware:

```sh
st-flash write bench.bin 0x08000000
```

Note: on real Cortex-M4 silicon, `DWT->CYCCNT` only increments while a
debug probe (SWD/JTAG) is attached, because the debug power domain gates
it — keep the probe connected when reading cycle counts off the board.

## Test vector generation

`python/export.py` and `python/export_nobackbone.py` each generate the same
5 sequences of 500 timesteps (`input_dim=6`), run through the respective
PyTorch cell timestep-by-timestep starting from a zero hidden state
(`export_nobackbone.py` reuses `export.py`'s `gen_sequences`/`run_golden`
helpers directly, so both golden sets are directly comparable):

- `step_a` / `step_b` — one/two step changes at fixed timesteps, constant `ts=1.0`
- `chirp_a` / `chirp_b` — per-channel linear-sweep sine waves, constant `ts=1.0`
- `noise` — Gaussian noise, with `ts` varying per-timestep in `[0.8, 1.2]` to exercise the time-gating math

`golden.json` / `golden_nobackbone.json` record `inputs`, `ts`,
`hidden_states`, and `outputs` for every timestep of every sequence
(`hidden_states` and `outputs` are numerically identical for a bare
`CfCCell`, and both are checked in the C tests for robustness).

`test/test_golden.c` and `test/test_golden_nobackbone.c` each parse their
JSON file with a small hand-rolled scanner tailored exactly to this schema
(no JSON library dependency) — it works because export.py fully controls
the field order and doesn't need to survive arbitrary JSON. The two test
files are near-identical duplicates rather than a shared parser, matching
the fully-separate-code-path design of the two `cfc_step_*` functions they
exercise.
