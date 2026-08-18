/*
 * Fused-input experiment: ff1/ff2/time_a/time_b all consume the SAME
 * 128-element vector (the backbone output; `cat` in the no-backbone
 * config). The shipped cfc.c computes them as 4 separate dense_layer
 * calls, so every input element is loaded 4 times -- once per call.
 * dense_layer4 below walks the input once and does 4 MACs per element
 * (one against each layer's weight row), loading each input element once
 * and reusing it 4 times.
 *
 * Load arithmetic per 4 MACs: shipped = 8 loads (4x input + 4x weight),
 * fused = 5 loads (1x input + 4x weight). Weight loads are irreducible --
 * each weight is used exactly once per step -- so 5/4 loads per MAC is
 * the floor for this data layout.
 *
 * Chain structure: the four layers' accumulators ARE four independent
 * FMA chains (a0..a3 belong to different output values), so this keeps
 * the shipped version's stall-breaking property by construction -- the
 * chains are interleaved per input element rather than per unrolled
 *4-element block. Built WITH -ffp-contract=fast, same as shipped.
 *
 * Applied in BOTH configs (cfc_step_backbone and cfc_step_nobackbone):
 * the no-backbone config's four layers share `cat` the same way. The
 * backbone's own 38->128 layer has no fusion partner and keeps the
 * shipped dense_layer.
 *
 * Rounding changes vs. shipped: each layer's sum is now one left-to-right
 * chain instead of 4 partial sums combined pairwise. test_golden is the
 * arbiter, as with every prior structural change.
 *
 * Both helpers force-inlined; verify with arm-none-eabi-nm after any
 * change (no dense_layer/dense_layer4 symbol = still inlined). That
 * check exists because a restructured dense_layer silently stopped
 * inlining once before.
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
        for (; i < in_dim; i++)
            acc += in[i] * wrow[i];

        out[o] = acc;
    }
}

/* Four dense layers over one shared input, input loaded once per element.
   a0..a3 are independent accumulator chains (one per layer). */
__attribute__((always_inline)) static inline void
dense_layer4(const float *in, int in_dim,
             const float *w0, const float *b0, float *out0,
             const float *w1, const float *b1, float *out1,
             const float *w2, const float *b2, float *out2,
             const float *w3, const float *b3, float *out3,
             int out_dim) {
    for (int o = 0; o < out_dim; o++) {
        const float *r0 = w0 + o * in_dim;
        const float *r1 = w1 + o * in_dim;
        const float *r2 = w2 + o * in_dim;
        const float *r3 = w3 + o * in_dim;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

        for (int i = 0; i < in_dim; i++) {
            float x = in[i];
            a0 += x * r0[i];
            a1 += x * r1[i];
            a2 += x * r2[i];
            a3 += x * r3[i];
        }

        out0[o] = b0[o] + a0;
        out1[o] = b1[o] + a1;
        out2[o] = b2[o] + a2;
        out3[o] = b3[o] + a3;
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
    dense_layer4(backbone, CFC_BACKBONE_UNITS,
                 CFC_FF1_W, CFC_FF1_B, ff1,
                 CFC_FF2_W, CFC_FF2_B, ff2,
                 CFC_TIME_A_W, CFC_TIME_A_B, ta,
                 CFC_TIME_B_W, CFC_TIME_B_B, tb,
                 CFC_HIDDEN_DIM);

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
    dense_layer4(cat, CFC_NB_CAT_DIM,
                 CFC_NB_FF1_W, CFC_NB_FF1_B, ff1,
                 CFC_NB_FF2_W, CFC_NB_FF2_B, ff2,
                 CFC_NB_TIME_A_W, CFC_NB_TIME_A_B, ta,
                 CFC_NB_TIME_B_W, CFC_NB_TIME_B_B, tb,
                 CFC_NB_HIDDEN_DIM);

    for (int i = 0; i < CFC_NB_HIDDEN_DIM; i++) {
        float f1 = cfc_tanh(ff1[i]);
        float f2 = cfc_tanh(ff2[i]);
        float t_interp = cfc_sigmoid(ta[i] * ts + tb[i]);
        float new_h = f1 * (1.0f - t_interp) + t_interp * f2;
        s->h[i] = new_h;
        output[i] = new_h;
    }
}
