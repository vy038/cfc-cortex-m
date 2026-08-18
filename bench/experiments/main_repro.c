#include <stdint.h>
#include <stdio.h>

#include "cfc.h"

extern void initialise_monitor_handles(void);

#define DEMCR (*(volatile uint32_t *)0xE000EDFCu)
#define DWT_CTRL (*(volatile uint32_t *)0xE0001000u)
#define DWT_CYCCNT (*(volatile uint32_t *)0xE0001004u)
#define FLASH_ACR (*(volatile uint32_t *)0x40023C00)

#define N 1000

/*
 * Reproducibility diagnostic: records every individual call's cycle count
 * (not just min/max/avg) plus an explicit iteration counter, to directly
 * answer:
 *   - is N actually executed, or is something being hoisted/skipped?
 *   - does the first call (or first few) differ systematically from later
 *     ones (a warm/cold effect)?
 * Run this same binary across separate reset/resume cycles (power cycle
 * or OpenOCD reset+run, not just re-running without resetting) and diff
 * the printed output -- the 231497-vs-248018 discrepancy on the identical
 * pade binary needs to be explained by comparing multiple such runs.
 */
static uint32_t cycles[N];

int main(void) {
    /* 2 wait states + prefetch + instruction cache + data cache.
       Do NOT rely on OpenOCD's reset-init to set this. */
    FLASH_ACR = 0x00000702;

    initialise_monitor_handles();
    DEMCR |= (1u << 24);  /* TRCENA */
    DWT_CTRL |= 1u;       /* CYCCNTENA */
    DWT_CYCCNT = 0;

    cfc_state_t state;
    cfc_init(&state);
    float input[CFC_INPUT_DIM] = {0.1f, -0.2f, 0.3f, 0.0f, 0.5f, -0.1f};
    float output[CFC_HIDDEN_DIM];

    int executed = 0;
    for (int i = 0; i < N; i++) {
        uint32_t t0 = DWT_CYCCNT;
        cfc_step_backbone(&state, input, 1.0f, output);
        uint32_t t1 = DWT_CYCCNT;
        cycles[i] = t1 - t0;
        executed++;
    }

    /* sanity check: did the loop actually run N times? */
    printf("iterations executed: %d (expected %d)\n", executed, N);

    uint32_t min_c = 0xFFFFFFFFu, max_c = 0;
    uint64_t sum_c = 0;
    for (int i = 0; i < N; i++) {
        if (cycles[i] < min_c)
            min_c = cycles[i];
        if (cycles[i] > max_c)
            max_c = cycles[i];
        sum_c += cycles[i];
    }
    printf("overall: min=%lu max=%lu avg=%lu (N=%d)\n", (unsigned long)min_c, (unsigned long)max_c,
           (unsigned long)(sum_c / N), N);

    printf("first 20 calls:");
    for (int i = 0; i < 20; i++)
        printf(" %lu", (unsigned long)cycles[i]);
    printf("\n");

    printf("last 20 calls:");
    for (int i = N - 20; i < N; i++)
        printf(" %lu", (unsigned long)cycles[i]);
    printf("\n");

    /* warm/cold check: average of first 50 vs average of last 50 */
    uint64_t first50 = 0, last50 = 0;
    for (int i = 0; i < 50; i++)
        first50 += cycles[i];
    for (int i = N - 50; i < N; i++)
        last50 += cycles[i];
    printf("avg(first 50)=%lu avg(last 50)=%lu delta=%ld\n", (unsigned long)(first50 / 50),
           (unsigned long)(last50 / 50), (long)(first50 / 50) - (long)(last50 / 50));

    /* single-call outlier check: how many calls differ from the modal
       (min) value, and by how much -- a real warm/cold or jitter effect
       should show up as a cluster of outliers, not uniform noise */
    int num_outliers = 0;
    uint32_t max_outlier = 0;
    for (int i = 0; i < N; i++) {
        if (cycles[i] != min_c) {
            num_outliers++;
            uint32_t d = cycles[i] - min_c;
            if (d > max_outlier)
                max_outlier = d;
        }
    }
    printf("calls != min: %d / %d, max deviation from min: %lu\n", num_outliers, N,
           (unsigned long)max_outlier);

    while (1) {
    }
}
