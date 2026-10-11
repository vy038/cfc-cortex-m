/* HSI -> PLL -> 84MHz SYSCLK (F401 max), APB1 42MHz. Replaces reliance on
 * OpenOCD's reset-init clock (~64MHz). Caller sets FLASH_ACR (2 wait states)
 * first: 84MHz at 2.7-3.6V needs 2WS, and the wait states must be in place
 * before the switch. Core voltage scale 1 is the F401 reset default. */
#ifndef CLOCK_H
#define CLOCK_H
#include <stdint.h>

#define RCC_CR   (*(volatile uint32_t *)0x40023800u)
#define RCC_PLL  (*(volatile uint32_t *)0x40023804u)
#define RCC_CFGR (*(volatile uint32_t *)0x40023808u)

static inline void clock_init_84mhz(void) {
    RCC_CR |= 1u;                                /* HSION */
    while (!(RCC_CR & 2u)) {}                    /* HSIRDY */
    RCC_CFGR &= ~3u;                             /* SYSCLK = HSI */
    while (RCC_CFGR & 0xCu) {}
    RCC_CR &= ~(1u << 24);                       /* PLL off to reconfigure */
    while (RCC_CR & (1u << 25)) {}
    /* 16MHz / M16 = 1MHz; x N336 = 336MHz; / P4 = 84MHz; Q7 = 48MHz */
    RCC_PLL = 16u | (336u << 6) | (1u << 16) | (7u << 24);
    RCC_CR |= 1u << 24;                          /* PLLON */
    while (!(RCC_CR & (1u << 25))) {}            /* PLLRDY */
    RCC_CFGR = 4u << 10;                         /* APB1 /2 (42MHz max) */
    RCC_CFGR |= 2u;                              /* SYSCLK = PLL */
    while ((RCC_CFGR & 0xCu) != 0x8u) {}
}
#endif
