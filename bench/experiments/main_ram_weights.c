#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "cfc.h"
#include "cfc_math.h"

extern void initialise_monitor_handles(void);

#define DEMCR (*(volatile uint32_t *)0xE000EDFCu)
#define DWT_CTRL (*(volatile uint32_t *)0xE0001000u)
#define DWT_CYCCNT (*(volatile uint32_t *)0xE0001004u)
#define FLASH_ACR (*(volatile uint32_t *)0x40023C00)

/*
 * RAM copies of every weight/bias array cfc_step_backbone reads, to test
 * whether flash-resident (ART-cached, 2 wait states) weights vs
 * SRAM-resident weights changes dense_layer's cost -- see NOTES.md's
 * "check where the weights live" diagnostic. Total ~84KB, fits in the
 * F401's 96KB SRAM alongside the ~2.3KB the rest of the firmware uses
 * (confirmed by successful link -- the linker script would error with
 * "region RAM overflowed" otherwise).
 */
static float ram_backbone_w[CFC_BACKBONE_UNITS * CFC_CAT_DIM];
static float ram_backbone_b[CFC_BACKBONE_UNITS];
static float ram_ff1_w[CFC_HIDDEN_DIM * CFC_BACKBONE_UNITS];
static float ram_ff1_b[CFC_HIDDEN_DIM];
static float ram_ff2_w[CFC_HIDDEN_DIM * CFC_BACKBONE_UNITS];
static float ram_ff2_b[CFC_HIDDEN_DIM];
static float ram_time_a_w[CFC_HIDDEN_DIM * CFC_BACKBONE_UNITS];
static float ram_time_a_b[CFC_HIDDEN_DIM];
static float ram_time_b_w[CFC_HIDDEN_DIM * CFC_BACKBONE_UNITS];
static float ram_time_b_b[CFC_HIDDEN_DIM];

/* Byte-for-byte copy of src/cfc.c's static dense_layer -- see
   main_instrumented.c for why this is duplicated rather than shared, and
   for the "must be re-synced when cfc.c changes" warning. Last synced:
   the partial-accumulator + always_inline version. */
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

/* cfc_step_backbone's logic, reading RAM weight copies instead of the
   flash-resident CFC_*_W/_B arrays. Everything else identical. */
static void cfc_step_backbone_ram(cfc_state_t *s, const float *input, float ts, float *output) {
    float cat[CFC_CAT_DIM];
    memcpy(cat, input, CFC_INPUT_DIM * sizeof(float));
    memcpy(cat + CFC_INPUT_DIM, s->h, CFC_HIDDEN_DIM * sizeof(float));

    float backbone[CFC_BACKBONE_UNITS];
    dense_layer(cat, CFC_CAT_DIM, ram_backbone_w, ram_backbone_b, CFC_BACKBONE_UNITS, backbone);
    for (int i = 0; i < CFC_BACKBONE_UNITS; i++)
        backbone[i] = cfc_lecun_tanh(backbone[i]);

    float ff1[CFC_HIDDEN_DIM], ff2[CFC_HIDDEN_DIM], ta[CFC_HIDDEN_DIM], tb[CFC_HIDDEN_DIM];
    dense_layer(backbone, CFC_BACKBONE_UNITS, ram_ff1_w, ram_ff1_b, CFC_HIDDEN_DIM, ff1);
    dense_layer(backbone, CFC_BACKBONE_UNITS, ram_ff2_w, ram_ff2_b, CFC_HIDDEN_DIM, ff2);
    dense_layer(backbone, CFC_BACKBONE_UNITS, ram_time_a_w, ram_time_a_b, CFC_HIDDEN_DIM, ta);
    dense_layer(backbone, CFC_BACKBONE_UNITS, ram_time_b_w, ram_time_b_b, CFC_HIDDEN_DIM, tb);

    for (int i = 0; i < CFC_HIDDEN_DIM; i++) {
        float f1 = cfc_tanh(ff1[i]);
        float f2 = cfc_tanh(ff2[i]);
        float t_interp = cfc_sigmoid(ta[i] * ts + tb[i]);
        float new_h = f1 * (1.0f - t_interp) + t_interp * f2;
        s->h[i] = new_h;
        output[i] = new_h;
    }
}

int main(void) {
    /* 2 wait states + prefetch + instruction cache + data cache.
       Do NOT rely on OpenOCD's reset-init to set this. */
    FLASH_ACR = 0x00000702;

    memcpy(ram_backbone_w, CFC_BACKBONE_W, sizeof(ram_backbone_w));
    memcpy(ram_backbone_b, CFC_BACKBONE_B, sizeof(ram_backbone_b));
    memcpy(ram_ff1_w, CFC_FF1_W, sizeof(ram_ff1_w));
    memcpy(ram_ff1_b, CFC_FF1_B, sizeof(ram_ff1_b));
    memcpy(ram_ff2_w, CFC_FF2_W, sizeof(ram_ff2_w));
    memcpy(ram_ff2_b, CFC_FF2_B, sizeof(ram_ff2_b));
    memcpy(ram_time_a_w, CFC_TIME_A_W, sizeof(ram_time_a_w));
    memcpy(ram_time_a_b, CFC_TIME_A_B, sizeof(ram_time_a_b));
    memcpy(ram_time_b_w, CFC_TIME_B_W, sizeof(ram_time_b_w));
    memcpy(ram_time_b_b, CFC_TIME_B_B, sizeof(ram_time_b_b));

    initialise_monitor_handles();
    DEMCR |= (1u << 24);  /* TRCENA */
    DWT_CTRL |= 1u;       /* CYCCNTENA */
    DWT_CYCCNT = 0;

    cfc_state_t state;
    cfc_init(&state);
    float input[CFC_INPUT_DIM] = {0.1f, -0.2f, 0.3f, 0.0f, 0.5f, -0.1f};
    float output[CFC_HIDDEN_DIM];

    const int N = 1000;
    uint32_t min_c = 0xFFFFFFFFu, max_c = 0;
    uint64_t sum_c = 0;

    for (int i = 0; i < N; i++) {
        uint32_t t0 = DWT_CYCCNT;
        cfc_step_backbone_ram(&state, input, 1.0f, output);
        uint32_t d = DWT_CYCCNT - t0;
        if (d < min_c)
            min_c = d;
        if (d > max_c)
            max_c = d;
        sum_c += d;
    }

    printf("cfc_step_backbone_ram cycles: min=%lu max=%lu avg=%lu (N=%d)\n", (unsigned long)min_c,
           (unsigned long)max_c, (unsigned long)(sum_c / N), N);

    while (1) {
    }
}
