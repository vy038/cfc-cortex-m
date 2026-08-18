/* NUCLEO-F401RE (STM32F401RE, STM32F401xD/xE line) memory map.
 * NOT an F407: 96K SRAM vs F407's 128K, hence RAM LENGTH below. Flash size
 * (512K) happens to match the F407VG figure used in earlier planning, but
 * that was coincidence, not confirmation the board was an F407. */
MEMORY
{
  FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 512K
  RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 96K
}

_estack = ORIGIN(RAM) + LENGTH(RAM);

SECTIONS
{
  .isr_vector : { KEEP(*(.isr_vector)) } > FLASH
  .text   : { *(.text*) } > FLASH
  .rodata : { *(.rodata*) } > FLASH

  _sidata = LOADADDR(.data);
  .data : {
    _sdata = .;
    *(.data*)
    _edata = .;
  } > RAM AT> FLASH

  .bss : {
    _sbss = .;
    *(.bss*)
    *(COMMON)
    _ebss = .;
  } > RAM

  PROVIDE(end = .);
  PROVIDE(_end = .);
}
