.syntax unified
.cpu cortex-m4
.thumb

.section .isr_vector,"a"
.word _estack
.word Reset_Handler
.word Default_Handler /* NMI */
.word Default_Handler /* HardFault */
.word Default_Handler /* MemManage */
.word Default_Handler /* BusFault */
.word Default_Handler /* UsageFault */
.word 0
.word 0
.word 0
.word 0
.word Default_Handler /* SVCall */
.word Default_Handler /* DebugMon */
.word 0
.word Default_Handler /* PendSV */
.word Default_Handler /* SysTick */

.section .text.Reset_Handler
.type Reset_Handler, %function
Reset_Handler:
    /* enable FPU (CP10/CP11 full access) before any float code runs */
    ldr r0, =0xE000ED88
    ldr r1, [r0]
    ldr r2, =0x00F00000
    orr r1, r1, r2
    str r1, [r0]
    dsb
    isb

    ldr r0, =_sdata
    ldr r1, =_edata
    ldr r2, =_sidata
copy_data:
    cmp r0, r1
    bge copy_done
    ldr r3, [r2], #4
    str r3, [r0], #4
    b copy_data
copy_done:

    ldr r0, =_sbss
    ldr r1, =_ebss
    movs r2, #0
zero_bss:
    cmp r0, r1
    bge zero_done
    str r2, [r0], #4
    b zero_bss
zero_done:

    bl main
hang:
    b hang

.type Default_Handler, %function
Default_Handler:
    b Default_Handler
