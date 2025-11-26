/*
 * thingino-accel - Open Source NNA Library for Ingenic T41/T31
 * 
 * Main NNA API - Hardware initialization and control
 */

#ifndef THINGINO_ACCEL_NNA_H
#define THINGINO_ACCEL_NNA_H

#include "nna_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Initialize NNA hardware
 * 
 * Opens /dev/soc-nna, initializes ORAM, and prepares hardware for inference.
 * 
 * Returns:
 *   NNA_SUCCESS on success
 *   NNA_ERROR_DEVICE if /dev/soc-nna cannot be opened
 *   NNA_ERROR_INIT if hardware initialization fails
 */
int nna_init(void);

/*
 * Deinitialize NNA hardware
 * 
 * Releases all resources and closes device.
 */
void nna_deinit(void);

/*
 * Get NNA hardware information
 * 
 * Args:
 *   info: Pointer to structure to fill with hardware info
 * 
 * Returns:
 *   NNA_SUCCESS on success
 *   NNA_ERROR_INVALID if info is NULL
 */
int nna_get_hw_info(nna_hw_info_t *info);

/*
 * Check if NNA is initialized and ready
 * 
 * Returns:
 *   1 if initialized, 0 otherwise
 */
int nna_is_ready(void);

/*
 * Get library version string
 * 
 * Returns:
 *   Version string (e.g., "0.1.0")
 */
const char* nna_get_version(void);

/*
 * Lock NNA for exclusive access
 * 
 * Use this when multiple processes might access NNA.
 * 
 * Returns:
 *   NNA_SUCCESS on success
 *   NNA_ERROR_TIMEOUT if lock cannot be acquired
 */
int nna_lock(void);

/*
 * Unlock NNA
 * 
 * Returns:
 *   NNA_SUCCESS on success
 */
int nna_unlock(void);

#ifdef __cplusplus
}
#endif

#endif /* THINGINO_ACCEL_NNA_H */

