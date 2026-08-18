# Diagnostic bench harnesses

**Not built by default** (`make` in `bench/` builds production only). Build
these with `make bench-instrumented`, `make bench-ram-weights`,
`make bench-repro`, or all archived targets with `make experiments`.

Unlike the source variants in `../../src/experiments/`, these are **tools,
not frozen historical measurements** — they link the *current* `src/cfc.c`,
so re-running them after a change measures the change.

| harness | what it answers |
|---|---|
| `main_instrumented.c` | Where do the cycles go? DWT timestamps around each of the 5 `dense_layer` calls individually, with the DWT read overhead calibrated and subtracted. This is what proved `dense_layer` is 96.5% of runtime. |
| `main_ram_weights.c` | Are flash-resident weights the bottleneck? Copies all ~84.5KB of weights to SRAM at startup and re-measures. Answer was no: 4.6%, not worth 84.5KB of a 96KB budget. |
| `main_repro.c` | Is the measurement trustworthy? Records all 1000 individual per-call cycle counts, not just min/max/avg, plus first-vs-last-50 warm/cold comparison and an executed-iteration counter. Established reproducibility to ~50 cycles. |

## Maintenance hazard

`main_instrumented.c` and `main_ram_weights.c` each contain a **copy** of
`dense_layer`, because the real one is `static` in `cfc.c` and not exposed
via the header. They are copies so that they profile exactly the shipping
code rather than a lookalike — but that means **they go stale silently** if
`src/cfc.c`'s `dense_layer` changes and these aren't updated to match.

Both are currently synced to the partial-accumulator + `always_inline`
version. If you change `dense_layer`, re-sync both or their numbers become
fiction.
