/*
 * Mars NN command abstraction
 *
 * This is an abstract, PDF-agnostic representation of NN commands
 * (load/compute/store/barrier). Later we can map these to the
 * concrete NNR/NNMAC encodings used by the T41 NNA.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MARS_NN_CMD_NOP = 0,
    MARS_NN_CMD_LOAD,    /* ORAM -> NNA tile */
    MARS_NN_CMD_STORE,   /* NNA tile -> ORAM */
    MARS_NN_CMD_MAC,     /* MAC / conv / GEMM-style op */
    MARS_NN_CMD_BARRIER, /* Synchronization point */
} mars_nn_cmd_type_t;

typedef struct {
    mars_nn_cmd_type_t type;
    uint16_t flags;
    uint16_t reserved;
    uint32_t arg0;
    uint32_t arg1;
    uint32_t arg2;
    uint32_t arg3;
} mars_nn_cmd_t;

typedef struct {
    mars_nn_cmd_t *cmds;
    size_t count;
    size_t capacity;
} mars_nn_program_t;

void mars_nn_program_init(mars_nn_program_t *p,
                          mars_nn_cmd_t *storage,
                          size_t capacity);

/* Simple helpers to append commands; return 0 on success, -1 on overflow. */
int mars_nn_emit_load(mars_nn_program_t *p,
                      uint32_t oram_offset,
                      uint32_t length_bytes,
                      uint32_t dst_tile);

int mars_nn_emit_mac(mars_nn_program_t *p,
                     uint32_t dst_tile,
                     uint32_t src_a_tile,
                     uint32_t src_b_tile,
                     uint32_t length);

int mars_nn_emit_store(mars_nn_program_t *p,
                       uint32_t oram_offset,
                       uint32_t length_bytes,
                       uint32_t src_tile);

#ifdef __cplusplus
}
#endif

