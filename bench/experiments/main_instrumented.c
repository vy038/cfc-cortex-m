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
 * Byte-for-byte copy of src/cfc.c's static dense_layer -- duplicated here
 * (not called via a shared header) so this diagnostic measures the exact
 * same code the production build runs, not a reimplementation that could
 * accidentally differ. IF src/cfc.c's dense_layer CHANGES, THIS MUST BE
 * UPDATED TO MATCH, or this harness silently profiles code that no longer
 * exists. Last synced: the partial-accumulator + always_inline version.
 */
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

/*
 * Byte-for-byte copy of cfc_step_backbone's logic (src/cfc.c), with DWT
 * timestamps inserted around each of the 5 dense_layer calls. t[] is
 * accumulated across N calls by the caller so the reported numbers are
 * summed/averaged the same way as the existing aggregate benchmark.
 */
static void cfc_step_backbone_instrumented(cfc_state_t *s, const float *input, float ts,
                                            float *output, uint32_t *t) {
    uint32_t t0, t1;

    float cat[CFC_CAT_DIM];
    memcpy(cat, input, CFC_INPUT_DIM * sizeof(float));
    memcpy(cat + CFC_INPUT_DIM, s->h, CFC_HIDDEN_DIM * sizeof(float));

    float backbone[CFC_BACKBONE_UNITS];
    t0 = DWT_CYCCNT;
    dense_layer(cat, CFC_CAT_DIM, CFC_BACKBONE_W, CFC_BACKBONE_B, CFC_BACKBONE_UNITS, backbone);
    t1 = DWT_CYCCNT;
    t[0] += (t1 - t0);
    for (int i = 0; i < CFC_BACKBONE_UNITS; i++)
        backbone[i] = cfc_lecun_tanh(backbone[i]);

    float ff1[CFC_HIDDEN_DIM], ff2[CFC_HIDDEN_DIM], ta[CFC_HIDDEN_DIM], tb[CFC_HIDDEN_DIM];

    t0 = DWT_CYCCNT;
    dense_layer(backbone, CFC_BACKBONE_UNITS, CFC_FF1_W, CFC_FF1_B, CFC_HIDDEN_DIM, ff1);
    t1 = DWT_CYCCNT;
    t[1] += (t1 - t0);

    t0 = DWT_CYCCNT;
    dense_layer(backbone, CFC_BACKBONE_UNITS, CFC_FF2_W, CFC_FF2_B, CFC_HIDDEN_DIM, ff2);
    t1 = DWT_CYCCNT;
    t[2] += (t1 - t0);

    t0 = DWT_CYCCNT;
    dense_layer(backbone, CFC_BACKBONE_UNITS, CFC_TIME_A_W, CFC_TIME_A_B, CFC_HIDDEN_DIM, ta);
    t1 = DWT_CYCCNT;
    t[3] += (t1 - t0);

    t0 = DWT_CYCCNT;
    dense_layer(backbone, CFC_BACKBONE_UNITS, CFC_TIME_B_W, CFC_TIME_B_B, CFC_HIDDEN_DIM, tb);
    t1 = DWT_CYCCNT;
    t[4] += (t1 - t0);

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

    initialise_monitor_handles();
    DEMCR |= (1u << 24);  /* TRCENA */
    DWT_CTRL |= 1u;       /* CYCCNTENA */

    /* Calibrate DWT_CYCCNT read overhead itself: back-to-back reads with
       nothing in between, averaged over many samples. Subtract this from
       each per-call measurement below, since every t0/t1 pair pays it. */
    const int CAL_N = 10000;
    uint64_t cal_sum = 0;
    for (int i = 0; i < CAL_N; i++) {
        uint32_t a = DWT_CYCCNT;
        uint32_t b = DWT_CYCCNT;
        cal_sum += (b - a);
    }
    uint32_t read_overhead = (uint32_t)(cal_sum / CAL_N);

    cfc_state_t state;
    cfc_init(&state);
    float input[CFC_INPUT_DIM] = {0.1f, -0.2f, 0.3f, 0.0f, 0.5f, -0.1f};
    float output[CFC_HIDDEN_DIM];

    const int N = 1000;
    uint32_t per_call[5] = {0, 0, 0, 0, 0};

    DWT_CYCCNT = 0;
    uint32_t agg_t0 = DWT_CYCCNT;
    for (int i = 0; i < N; i++)
        cfc_step_backbone_instrumented(&state, input, 1.0f, output, per_call);
    uint32_t agg_t1 = DWT_CYCCNT;
    uint32_t aggregate = agg_t1 - agg_t0;

    const char *names[5] = {"backbone", "ff1", "ff2", "time_a", "time_b"};
    printf("DWT read overhead (avg of %d back-to-back reads): %lu cycles\n", CAL_N,
           (unsigned long)read_overhead);
    printf("Instrumented aggregate (N=%d): %lu cycles (%lu/call)\n", N, (unsigned long)aggregate,
           (unsigned long)(aggregate / N));

    uint32_t sum_per_call_avg = 0;
    for (int k = 0; k < 5; k++) {
        uint32_t raw_avg = per_call[k] / N;
        /* each per-call measurement pays read_overhead once (one t0/t1 pair) */
        uint32_t corrected_avg = (raw_avg > read_overhead) ? (raw_avg - read_overhead) : 0;
        sum_per_call_avg += corrected_avg;
        printf("  %-9s raw_sum=%-10lu raw_avg=%-6lu corrected_avg=%lu\n", names[k],
               (unsigned long)per_call[k], (unsigned long)raw_avg, (unsigned long)corrected_avg);
    }
    printf("Sum of 5 corrected per-call averages: %lu cycles\n", (unsigned long)sum_per_call_avg);
    printf("Instrumentation tax (instrumented aggregate - sum of corrected dense_layer calls): "
           "%lu cycles\n",
           (unsigned long)((aggregate / N) - sum_per_call_avg));

    while (1) {
    }
}
