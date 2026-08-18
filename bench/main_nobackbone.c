#include <stdint.h>
#include <stdio.h>

#include "cfc.h"

extern void initialise_monitor_handles(void);

#define DEMCR (*(volatile uint32_t *)0xE000EDFCu)
#define DWT_CTRL (*(volatile uint32_t *)0xE0001000u)
#define DWT_CYCCNT (*(volatile uint32_t *)0xE0001004u)

#define FLASH_ACR (*(volatile uint32_t *)0x40023C00)

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
    float input[CFC_NB_INPUT_DIM] = {0.1f, -0.2f, 0.3f, 0.0f, 0.5f, -0.1f};
    float output[CFC_NB_HIDDEN_DIM];

    const int N = 1000;
    uint32_t min_c = 0xFFFFFFFFu, max_c = 0;
    uint64_t sum_c = 0;

    for (int i = 0; i < N; i++) {
        uint32_t t0 = DWT_CYCCNT;
        cfc_step_nobackbone(&state, input, 1.0f, output);
        uint32_t d = DWT_CYCCNT - t0;
        if (d < min_c)
            min_c = d;
        if (d > max_c)
            max_c = d;
        sum_c += d;
    }

    printf("cfc_step_nobackbone cycles: min=%lu max=%lu avg=%lu (N=%d)\n", (unsigned long)min_c,
           (unsigned long)max_c, (unsigned long)(sum_c / N), N);

    while (1) {
    }
}
