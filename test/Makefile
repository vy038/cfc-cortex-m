CC = gcc

# Host verification must compile src/cfc.c the same way the firmware does,
# or it validates arithmetic the target never runs. Two flags matter:
#
#   -ffp-contract=fast  lets GCC fuse multiply-add (single rounding instead
#                       of two). The ARM build uses this; see bench/Makefile.
#   -mfma               on x86-64 the flag above is NOT enough: the SSE2
#                       baseline has no FMA instruction, so GCC emits zero
#                       fused ops and the test silently exercises unfused
#                       arithmetic. Verified by objdump: 0 fused instructions
#                       without -mfma, 34 with it. Needs a Haswell-era (2013+)
#                       CPU, so it is probed rather than assumed below.
#
# Both FMA fusion and dense_layer's partial accumulators change float rounding
# vs. a naive implementation, and both ship in the firmware, so both belong in
# what the golden test covers.
HAS_FMA := $(shell grep -qm1 ' fma ' /proc/cpuinfo 2>/dev/null && echo yes)
ifeq ($(HAS_FMA),yes)
FMA_FLAGS = -ffp-contract=fast -mfma
else
FMA_FLAGS = -ffp-contract=fast
$(warning This CPU lacks FMA: host tests will run UNFUSED arithmetic and will \
not fully match the ARM target's rounding. The 1e-5 threshold has ~20x margin \
so this should still pass, but it is weaker verification than on an FMA host.)
endif

CFLAGS = -std=c99 -Wall -Wextra -O2 $(FMA_FLAGS) -I../src -I../weights/generated
LDLIBS = -lm
CFC_SRCS = ../src/cfc.c ../src/cfc_math.c

BIN = test_golden
BIN_NB = test_golden_nobackbone

all: $(BIN) $(BIN_NB)

$(BIN): test_golden.c $(CFC_SRCS)
	$(CC) $(CFLAGS) -o $@ test_golden.c $(CFC_SRCS) $(LDLIBS)

$(BIN_NB): test_golden_nobackbone.c $(CFC_SRCS)
	$(CC) $(CFLAGS) -o $@ test_golden_nobackbone.c $(CFC_SRCS) $(LDLIBS)

run: $(BIN)
	./$(BIN) ../weights/generated/golden.json

run-nobackbone: $(BIN_NB)
	./$(BIN_NB) ../weights/generated/golden_nobackbone.json

run-all: run run-nobackbone

clean:
	rm -f $(BIN) $(BIN_NB)

.PHONY: all run run-nobackbone run-all clean
