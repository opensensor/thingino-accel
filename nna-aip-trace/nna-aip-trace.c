#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/io.h>
#include <linux/delay.h>
#include <linux/workqueue.h>
#include <linux/slab.h>

#define NNA_AIP_TRACE_VERSION "0.1"

/*
 * Lightweight NNA/AIP register & descriptor tracer, modeled after tx-isp-trace.c
 *
 * Usage (on the camera kernel tree):
 *   - Build this as an out-of-tree module against the running kernel.
 *   - insmod nna-aip-trace.ko
 *   - Run stock Venus / libdrivers workloads.
 *   - Watch dmesg and/or capture it to a file for offline analysis.
 *
 * This intentionally keeps policy simple: it logs every 32-bit word change in
 * the small I/O regions and sampled changes in descriptor RAM. If this is too
 * noisy we can add smarter classification/filters later.
 */

/* Hardware regions of interest (from soc-nna + AIP RE work) */
#define NNDMA_IO_PADDR       0x12508000
#define NNDMA_IO_SIZE        0x1000
#define NNDMA_DESRAM_PADDR   0x12500000
#define NNDMA_DESRAM_SIZE    0x8000
#define AIP_IO_PADDR         0x12b00000
#define AIP_IO_SIZE          0x1000

/* Optional ORAM tracing (can be enabled later if needed) */
#define TRACE_ORAM 0
#define L2CACHE_SIZE_PADDR   0x12200000
#define L2CACHE_SIZE_OFFSET  0x60
#define ORAM_BASE_ADDR       0x12600000

struct trace_region {
    phys_addr_t phys;
    void __iomem *virt;
    size_t size;           /* bytes, multiple of 4 */
    const char *name;
    u32 *last_vals;        /* previous 32-bit values */
};

static struct trace_region regions[] = {
    { NNDMA_IO_PADDR,     NULL, NNDMA_IO_SIZE,     "nndma-io",   NULL },
    { NNDMA_DESRAM_PADDR, NULL, NNDMA_DESRAM_SIZE, "nndma-desram", NULL },
    { AIP_IO_PADDR,       NULL, AIP_IO_SIZE,       "aip-io",     NULL },
};

#define NUM_REGIONS (sizeof(regions) / sizeof(regions[0]))

static struct delayed_work trace_work;
static bool tracing = false;

static int init_region(struct trace_region *r)
{
    size_t words, i;

    r->virt = ioremap(r->phys, r->size);
    if (!r->virt) {
        pr_err("nna-aip-trace: failed to ioremap %s at 0x%pap size 0x%zx\n",
               r->name, &r->phys, r->size);
        return -ENOMEM;
    }

    words = r->size / sizeof(u32);
    r->last_vals = kcalloc(words, sizeof(u32), GFP_KERNEL);
    if (!r->last_vals) {
        iounmap(r->virt);
        r->virt = NULL;
        return -ENOMEM;
    }

    /* Initial snapshot */
    for (i = 0; i < words; i++)
        r->last_vals[i] = readl(r->virt + i * sizeof(u32));

    pr_info("nna-aip-trace: region %s init phys=0x%pap size=0x%zx words=%zu\n",
            r->name, &r->phys, r->size, words);
    return 0;
}

static void cleanup_region(struct trace_region *r)
{
    if (!r)
        return;
    if (r->last_vals) {
        kfree(r->last_vals);
        r->last_vals = NULL;
    }
    if (r->virt) {
        iounmap(r->virt);
        r->virt = NULL;
    }
}

static void trace_worker(struct work_struct *work)
{
    int ri;

    if (!tracing)
        return;

    for (ri = 0; ri < NUM_REGIONS; ri++) {
        struct trace_region *r = &regions[ri];
        size_t words = r->size / sizeof(u32);
        size_t i;
        int changes = 0;

        if (!r->virt || !r->last_vals)
            continue;

        for (i = 0; i < words; i++) {
            u32 off = i * sizeof(u32);
            u32 cur = readl(r->virt + off);
            u32 prev = r->last_vals[i];

            if (cur != prev) {
                /* Limit spam per region per tick */
                if (changes < 64) {
                    pr_info("nna-aip-trace: %s off 0x%04x: 0x%08x -> 0x%08x\n",
                            r->name, off, prev, cur);
                }
                r->last_vals[i] = cur;
                changes++;
            }
        }

        if (changes)
            pr_info("nna-aip-trace: %s changed %d words this tick\n",
                    r->name, changes);
    }

    /* Re-arm at ~20ms interval */
    schedule_delayed_work(&trace_work, HZ / 50);
}

static int __init nna_aip_trace_init(void)
{
    int i, ret;

    pr_info("nna-aip-trace v%s initializing\n", NNA_AIP_TRACE_VERSION);

    for (i = 0; i < NUM_REGIONS; i++) {
        ret = init_region(&regions[i]);
        if (ret) {
            while (--i >= 0)
                cleanup_region(&regions[i]);
            return ret;
        }
    }

    tracing = true;
    INIT_DELAYED_WORK(&trace_work, trace_worker);
    schedule_delayed_work(&trace_work, HZ / 50);

    pr_info("nna-aip-trace: started background polling\n");
    return 0;
}

static void __exit nna_aip_trace_exit(void)
{
    int i;

    tracing = false;
    cancel_delayed_work_sync(&trace_work);

    for (i = 0; i < NUM_REGIONS; i++)
        cleanup_region(&regions[i]);

    pr_info("nna-aip-trace: unloaded\n");
}

module_init(nna_aip_trace_init);
module_exit(nna_aip_trace_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("thingino-accel");
MODULE_DESCRIPTION("NNA/AIP register and descriptor tracer for T41 Venus reverse engineering");
MODULE_VERSION(NNA_AIP_TRACE_VERSION);

