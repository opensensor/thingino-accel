/*
 * nna_init.c - Simple kernel module to initialize NNA system register
 * 
 * This module writes the required value to 0x13012038 which is needed
 * for the AIP hardware to function correctly.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/io.h>

#define SYS_REG_ADDR 0x13012038
#define SYS_REG_VALUE 0x88404002

static int __init nna_init_init(void)
{
    volatile unsigned int *reg;
    unsigned int old_val, new_val;
    
    /* Use kseg1 address for uncached access */
    reg = (volatile unsigned int *)(0xa0000000 + SYS_REG_ADDR);
    
    /* Sync before read */
    __asm__ volatile(
        ".set push\n\t"
        ".set mips32r2\n\t"
        "sync\n\t"
        "lw $0, 0(%0)\n\t"
        ".set pop\n\t"
        ::"r" (0xa0000000));
    
    old_val = *reg;
    
    /* Write the required value */
    *reg = SYS_REG_VALUE;
    
    /* Sync after write */
    __asm__ volatile(
        ".set push\n\t"
        ".set mips32r2\n\t"
        "sync\n\t"
        "lw $0, 0(%0)\n\t"
        ".set pop\n\t"
        ::"r" (0xa0000000));
    
    new_val = *reg;
    
    printk(KERN_INFO "nna_init: Wrote 0x%08x to 0x%08x (was 0x%08x, now 0x%08x)\n",
           SYS_REG_VALUE, SYS_REG_ADDR, old_val, new_val);
    
    return 0;
}

static void __exit nna_init_exit(void)
{
    printk(KERN_INFO "nna_init: Module unloaded\n");
}

module_init(nna_init_init);
module_exit(nna_init_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("thingino-accel");
MODULE_DESCRIPTION("NNA System Register Initializer");
MODULE_VERSION("1.0");

