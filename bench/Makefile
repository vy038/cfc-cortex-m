CC = arm-none-eabi-gcc
OBJCOPY = arm-none-eabi-objcopy
CPU = -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16

# CFLAGS_BASE is the historical flag set, kept separate ONLY so the archived
# experiments below can reproduce the numbers recorded in NOTES.md, which
# were measured before -ffp-contract=fast was adopted. Production builds use
# CFLAGS.
CFLAGS_BASE = $(CPU) -std=c99 -Wall -Wextra -O2 -ffreestanding \
              -ffunction-sections -fdata-sections -fstack-usage \
              -I../src -I../weights/generated
# -ffp-contract=fast lets GCC fuse multiply-add into vfma.f32. Required for
# src/cfc.c's dense_layer to hit its measured performance; see OPTIMIZATION.md.
# test/Makefile carries the same flag so host verification matches the target.
CFLAGS = $(CFLAGS_BASE) -ffp-contract=fast
LDFLAGS = $(CPU) -T stm32f4.ld -specs=rdimon.specs -nostartfiles -Wl,--gc-sections
LDLIBS = -lm
CFC_SRCS = ../src/cfc.c ../src/cfc_math.c

BIN = bench
BIN_NB = bench-nobackbone

# ---------------------------------------------------------------------------
# Production
# ---------------------------------------------------------------------------

all: $(BIN).elf $(BIN).bin

$(BIN).elf: main.c startup.s $(CFC_SRCS) stm32f4.ld
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ main.c startup.s $(CFC_SRCS) $(LDLIBS)

$(BIN).bin: $(BIN).elf
	$(OBJCOPY) -O binary $< $@

bench-nobackbone: $(BIN_NB).elf $(BIN_NB).bin

$(BIN_NB).elf: main_nobackbone.c startup.s $(CFC_SRCS) stm32f4.ld
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ main_nobackbone.c startup.s $(CFC_SRCS) $(LDLIBS)

$(BIN_NB).bin: $(BIN_NB).elf
	$(OBJCOPY) -O binary $< $@

all-configs: all bench-nobackbone

clean:
	rm -f *.elf *.bin *.su

.PHONY: all bench-nobackbone all-configs clean

# ===========================================================================
# ARCHIVED EXPERIMENTS -- not part of the build, kept for reproducibility.
#
# These reproduce the measurements recorded in NOTES.md and OPTIMIZATION.md.
# Sources live in ../src/experiments/ and ./experiments/. Nothing below is
# needed to build or ship the firmware; `make` alone builds production only.
#
# Flag fidelity matters here: targets measured before -ffp-contract=fast was
# adopted deliberately use CFLAGS_BASE, so they still reproduce their recorded
# numbers rather than silently picking up the new flag.
#
# The old single-accumulator dense_layer lives in cfc_baseline.c.bak. The .bak
# extension keeps it from reading as live source, so it needs `-x c` to tell
# GCC it is C despite the extension.
# ===========================================================================

EXP_SRC = ../src/experiments
EXP_MAIN = experiments
BASELINE_CFC = -x c $(EXP_SRC)/cfc_baseline.c.bak -x none

# --- activation study: which cfc_math implementation (all vs baseline cfc.c)
bench-pade.elf: main.c startup.s $(EXP_SRC)/cfc_baseline.c.bak ../src/cfc_math.c stm32f4.ld
	$(CC) $(CFLAGS_BASE) $(LDFLAGS) -o $@ main.c startup.s $(BASELINE_CFC) ../src/cfc_math.c $(LDLIBS)

bench-libm.elf: main.c startup.s $(EXP_SRC)/cfc_baseline.c.bak $(EXP_SRC)/cfc_math_libm.c stm32f4.ld
	$(CC) $(CFLAGS_BASE) $(LDFLAGS) -o $@ main.c startup.s $(BASELINE_CFC) $(EXP_SRC)/cfc_math_libm.c $(LDLIBS)

# stub is DELIBERATELY numerically wrong (identity activations): timing floor
# only, fails test_golden by design, never a shippable build.
bench-stub.elf: main.c startup.s $(EXP_SRC)/cfc_baseline.c.bak $(EXP_SRC)/cfc_math_stub.c stm32f4.ld
	$(CC) $(CFLAGS_BASE) $(LDFLAGS) -o $@ main.c startup.s $(BASELINE_CFC) $(EXP_SRC)/cfc_math_stub.c $(LDLIBS)

# --- dense_layer study: the optimization sequence
bench-partial-acc.elf: main.c startup.s $(EXP_SRC)/cfc_partial_acc.c ../src/cfc_math.c stm32f4.ld
	$(CC) $(CFLAGS_BASE) $(LDFLAGS) -o $@ main.c startup.s $(EXP_SRC)/cfc_partial_acc.c ../src/cfc_math.c $(LDLIBS)

bench-fma.elf: main.c startup.s $(EXP_SRC)/cfc_baseline.c.bak ../src/cfc_math.c stm32f4.ld
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ main.c startup.s $(BASELINE_CFC) ../src/cfc_math.c $(LDLIBS)

bench-fma-unroll.elf: main.c startup.s $(EXP_SRC)/cfc_unroll.c ../src/cfc_math.c stm32f4.ld
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ main.c startup.s $(EXP_SRC)/cfc_unroll.c ../src/cfc_math.c $(LDLIBS)

# NOTE: there is no bench-partial-acc-fma target -- that variant WON and was
# promoted to src/cfc.c, so plain `make` builds it. bench.bin IS that binary.

# --- diagnostic harnesses (tools, not fixed historical measurements: these
# track the CURRENT src/cfc.c). instrumented/ram-weights carry their own copy
# of dense_layer and must be re-synced when src/cfc.c's changes.
bench-instrumented.elf: $(EXP_MAIN)/main_instrumented.c startup.s $(CFC_SRCS) stm32f4.ld
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(EXP_MAIN)/main_instrumented.c startup.s $(CFC_SRCS) $(LDLIBS)

bench-ram-weights.elf: $(EXP_MAIN)/main_ram_weights.c startup.s $(CFC_SRCS) stm32f4.ld
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(EXP_MAIN)/main_ram_weights.c startup.s $(CFC_SRCS) $(LDLIBS)

bench-repro.elf: $(EXP_MAIN)/main_repro.c startup.s $(CFC_SRCS) stm32f4.ld
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(EXP_MAIN)/main_repro.c startup.s $(CFC_SRCS) $(LDLIBS)

%.bin: %.elf
	$(OBJCOPY) -O binary $< $@

EXPERIMENTS = bench-pade bench-libm bench-stub \
              bench-partial-acc bench-fma bench-fma-unroll \
              bench-instrumented bench-ram-weights bench-repro

$(EXPERIMENTS): %: %.elf %.bin

experiments: $(EXPERIMENTS)

.PHONY: experiments $(EXPERIMENTS)
