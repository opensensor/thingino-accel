/*
 * thingino-accel - Internal device interface
 * 
 * Internal functions for device access (not part of public API)
 */

#ifndef THINGINO_ACCEL_DEVICE_INTERNAL_H
#define THINGINO_ACCEL_DEVICE_INTERNAL_H

#include <stdint.h>

/* Get device file descriptor */
int nna_device_get_fd(void);

/* Get /dev/mem file descriptor */
int nna_device_get_memfd(void);

/* Get ORAM mapped pointer */
void* nna_device_get_oram(void);

#endif /* THINGINO_ACCEL_DEVICE_INTERNAL_H */

