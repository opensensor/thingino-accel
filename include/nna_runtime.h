/*
 * thingino-accel - Runtime Environment for .mgk Models
 * 
 * Provides runtime symbols and environment for .mgk model execution
 */

#ifndef THINGINO_ACCEL_NNA_RUNTIME_H
#define THINGINO_ACCEL_NNA_RUNTIME_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize runtime environment (sets up oram_base, etc.) */
int nna_runtime_init(void);

/* Cache flush function */
void __aie_flushcache_dir(void *addr, size_t size, int direction);

/* Exported symbols for .mgk models */
extern void *oram_base;
extern void *__oram_vbase;
extern void *__ddr_pbase;
extern void *__ddr_vbase;
extern void *__nndma_io_vbase;
extern void *__nndma_desram_vbase;
extern void *__nndma_fastio_vbase;

#ifdef __cplusplus
}
#endif

#endif /* THINGINO_ACCEL_NNA_RUNTIME_H */

