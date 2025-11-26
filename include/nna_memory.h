/*
 * thingino-accel - Open Source NNA Library for Ingenic T41/T31
 * 
 * Memory management API - ORAM and DDR allocation
 */

#ifndef THINGINO_ACCEL_NNA_MEMORY_H
#define THINGINO_ACCEL_NNA_MEMORY_H

#include "nna_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Allocate aligned memory for NNA operations
 * 
 * Memory is allocated from DDR and aligned to 64-byte boundary
 * (required for DMA operations).
 * 
 * Args:
 *   size: Number of bytes to allocate
 * 
 * Returns:
 *   Pointer to allocated memory, or NULL on failure
 */
void* nna_malloc(size_t size);

/*
 * Allocate aligned memory with specific alignment
 * 
 * Args:
 *   alignment: Alignment in bytes (must be power of 2)
 *   size: Number of bytes to allocate
 * 
 * Returns:
 *   Pointer to allocated memory, or NULL on failure
 */
void* nna_memalign(size_t alignment, size_t size);

/*
 * Allocate and zero-initialize memory
 * 
 * Args:
 *   nmemb: Number of elements
 *   size: Size of each element
 * 
 * Returns:
 *   Pointer to allocated memory, or NULL on failure
 */
void* nna_calloc(size_t nmemb, size_t size);

/*
 * Free memory allocated by nna_malloc/nna_memalign/nna_calloc
 * 
 * Args:
 *   ptr: Pointer to memory to free
 */
void nna_free(void *ptr);

/*
 * Allocate memory from ORAM (on-chip accelerator RAM)
 * 
 * ORAM is limited (384KB on T41) but very fast. Use for:
 * - Intermediate feature maps
 * - Small tensors
 * - Frequently accessed data
 * 
 * Args:
 *   size: Number of bytes to allocate
 * 
 * Returns:
 *   Pointer to ORAM memory, or NULL if ORAM is full
 */
void* nna_oram_malloc(size_t size);

/*
 * Free ORAM memory
 * 
 * Args:
 *   ptr: Pointer to ORAM memory to free
 */
void nna_oram_free(void *ptr);

/*
 * Get ORAM memory usage statistics
 * 
 * Args:
 *   total: Output - total ORAM size in bytes
 *   used: Output - used ORAM size in bytes
 *   free: Output - free ORAM size in bytes
 * 
 * Returns:
 *   NNA_SUCCESS on success
 */
int nna_oram_get_stats(size_t *total, size_t *used, size_t *free);

/*
 * Flush CPU cache for DMA operations
 * 
 * Call this before NNA reads from memory written by CPU.
 * 
 * Args:
 *   ptr: Pointer to memory region
 *   size: Size of memory region in bytes
 */
void nna_cache_flush(void *ptr, size_t size);

/*
 * Invalidate CPU cache for DMA operations
 * 
 * Call this before CPU reads from memory written by NNA.
 * 
 * Args:
 *   ptr: Pointer to memory region
 *   size: Size of memory region in bytes
 */
void nna_cache_invalidate(void *ptr, size_t size);

#ifdef __cplusplus
}
#endif

#endif /* THINGINO_ACCEL_NNA_MEMORY_H */

