/*
 * Optimization experiment #3 (isolated from #1, stacked on top of #2):
 * dense_layer's inner loop unrolled via #pragma GCC unroll, to test
 * whether reducing loop-control overhead (cmp/branch, ~2 of the 6
 * instructions/MAC found in the disassembly) helps beyond FMA alone.
 * Meant to be built WITH -ffp-contract=fast (see bench/Makefile's
 * bench-fma-unroll target) -- that answers "does unrolling help beyond
 * FMA," per the investigation's step 3.
 *
 * Accumulation order is UNCHANGED from src/cfc.c (still one running
 * `acc`, left to right) -- unrolling here only replicates the loop body,
 * it does not reassociate the sum. Bit-identical output to src/cfc.c is
 * expected when built with the same flags; any difference from baseline
 * when built with -ffp-contract=fast comes from FMA's single-rounding
 * behavior, same as experiment #2, not from this file's unrolling.
 *
 * Everything else is byte-for-byte identical to src/cfc.c. Full-file copy
 * rather than #ifdef, matching this project's established pattern for
 * measurable variants (cfc_math_libm.c, cfc_math_stub.c).
 */
#include <string.h>

#include "cfc.h"
#include "cfc_math.h"

static void dense_layer(const float *in, int in_dim, const float *w, const float *b,
                         int out_dim, float *out) {
    for (int o = 0; o < out_dim; o++) {
        float acc = b[o];
#pragma GCC unroll 4
        for (int i = 0; i < in_dim; i++)
            acc += in[i] * w[o * in_dim + i];
        out[o] = acc;
    }
}

void cfc_init(cfc_state_t *s) {
    memset(s->h, 0, sizeof(s->h));
}

void cfc_step_backbone(cfc_state_t *s, const float *input, float ts, float *output) {
    float cat[CFC_CAT_DIM];
    memcpy(cat, input, CFC_INPUT_DIM * sizeof(float));
    memcpy(cat + CFC_INPUT_DIM, s->h, CFC_HIDDEN_DIM * sizeof(float));

    float backbone[CFC_BACKBONE_UNITS];
    dense_layer(cat, CFC_CAT_DIM, CFC_BACKBONE_W, CFC_BACKBONE_B, CFC_BACKBONE_UNITS, backbone);
    for (int i = 0; i < CFC_BACKBONE_UNITS; i++)
        backbone[i] = cfc_lecun_tanh(backbone[i]);

    float ff1[CFC_HIDDEN_DIM], ff2[CFC_HIDDEN_DIM], ta[CFC_HIDDEN_DIM], tb[CFC_HIDDEN_DIM];
    dense_layer(backbone, CFC_BACKBONE_UNITS, CFC_FF1_W, CFC_FF1_B, CFC_HIDDEN_DIM, ff1);
    dense_layer(backbone, CFC_BACKBONE_UNITS, CFC_FF2_W, CFC_FF2_B, CFC_HIDDEN_DIM, ff2);
    dense_layer(backbone, CFC_BACKBONE_UNITS, CFC_TIME_A_W, CFC_TIME_A_B, CFC_HIDDEN_DIM, ta);
    dense_layer(backbone, CFC_BACKBONE_UNITS, CFC_TIME_B_W, CFC_TIME_B_B, CFC_HIDDEN_DIM, tb);

    for (int i = 0; i < CFC_HIDDEN_DIM; i++) {
        float f1 = cfc_tanh(ff1[i]);
        float f2 = cfc_tanh(ff2[i]);
        float t_interp = cfc_sigmoid(ta[i] * ts + tb[i]);
        float new_h = f1 * (1.0f - t_interp) + t_interp * f2;
        s->h[i] = new_h;
        output[i] = new_h;
    }
}

void cfc_step_nobackbone(cfc_state_t *s, const float *input, float ts, float *output) {
    float cat[CFC_NB_CAT_DIM];
    memcpy(cat, input, CFC_NB_INPUT_DIM * sizeof(float));
    memcpy(cat + CFC_NB_INPUT_DIM, s->h, CFC_NB_HIDDEN_DIM * sizeof(float));

    float ff1[CFC_NB_HIDDEN_DIM], ff2[CFC_NB_HIDDEN_DIM], ta[CFC_NB_HIDDEN_DIM], tb[CFC_NB_HIDDEN_DIM];
    dense_layer(cat, CFC_NB_CAT_DIM, CFC_NB_FF1_W, CFC_NB_FF1_B, CFC_NB_HIDDEN_DIM, ff1);
    dense_layer(cat, CFC_NB_CAT_DIM, CFC_NB_FF2_W, CFC_NB_FF2_B, CFC_NB_HIDDEN_DIM, ff2);
    dense_layer(cat, CFC_NB_CAT_DIM, CFC_NB_TIME_A_W, CFC_NB_TIME_A_B, CFC_NB_HIDDEN_DIM, ta);
    dense_layer(cat, CFC_NB_CAT_DIM, CFC_NB_TIME_B_W, CFC_NB_TIME_B_B, CFC_NB_HIDDEN_DIM, tb);

    for (int i = 0; i < CFC_NB_HIDDEN_DIM; i++) {
        float f1 = cfc_tanh(ff1[i]);
        float f2 = cfc_tanh(ff2[i]);
        float t_interp = cfc_sigmoid(ta[i] * ts + tb[i]);
        float new_h = f1 * (1.0f - t_interp) + t_interp * f2;
        s->h[i] = new_h;
        output[i] = new_h;
    }
}
