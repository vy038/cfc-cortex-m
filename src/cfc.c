/*
 * The CfC cell. dense_layer's inner loop is the hot path -- 96.5% of
 * cfc_step_backbone's runtime, measured per-call on hardware. Its shape
 * here is the product of a measured optimization sequence; see
 * OPTIMIZATION.md for the full story and NOTES.md for the raw numbers.
 * Two things in this loop are load-bearing and must not be "cleaned up":
 *
 * 1. FOUR PARTIAL ACCUMULATORS (acc0..acc3), not one running total.
 *    A single accumulator is a strict serial dependency chain: each MAC
 *    waits on the previous MAC's FPU add to retire, and the Cortex-M4
 *    TRM notes float arithmetic takes an extra cycle when its result is
 *    consumed by the next instruction. Four independent chains let three
 *    FMAs issue while one resolves. Worth ~15% on its own -- more than
 *    the instruction count it costs. Collapsing these back into one
 *    accumulator will silently cost that back.
 *
 * 2. __attribute__((always_inline)). Without it, this function's body is
 *    large enough that GCC stops inlining it at -O2 and emits 5 real
 *    calls per step instead. That happened silently once already and
 *    confounded a measurement. After ANY edit here, re-check with
 *    `arm-none-eabi-nm bench.elf | grep dense_layer` -- no symbol means
 *    still inlined. always_inline is a strong hint, not a guarantee.
 *
 * Built with -ffp-contract=fast (see bench/Makefile and test/Makefile),
 * which lets GCC fuse the multiply-add into a single vfma.f32. Without
 * that flag GCC emits separate vmul.f32 + vadd.f32 -- strict C99
 * conformance forbids the fusion, since FMA rounds once where the
 * separate pair rounds twice. Both flag and accumulator structure change
 * float rounding; test_golden is the arbiter, and both pass at 1e-5 with
 * roughly 20x margin.
 */
#include <string.h>

#include "cfc.h"
#include "cfc_math.h"

__attribute__((always_inline)) static inline void
dense_layer(const float *in, int in_dim, const float *w, const float *b, int out_dim,
            float *out) {
    for (int o = 0; o < out_dim; o++) {
        const float *wrow = w + o * in_dim;
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

        int n4 = in_dim - (in_dim % 4);
        int i = 0;
        for (; i < n4; i += 4) {
            acc0 += in[i] * wrow[i];
            acc1 += in[i + 1] * wrow[i + 1];
            acc2 += in[i + 2] * wrow[i + 2];
            acc3 += in[i + 3] * wrow[i + 3];
        }

        float acc = b[o] + (acc0 + acc1) + (acc2 + acc3);
        /* remainder: in_dim=38 (backbone call) leaves 2 here, in_dim=128
           (the other four calls) leaves 0 */
        for (; i < in_dim; i++)
            acc += in[i] * wrow[i];

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
