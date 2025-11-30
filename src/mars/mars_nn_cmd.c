/*
 * Mars NN command abstraction implementation
 */

#include "mars_nn_cmd.h"

static int mars_nn_push_cmd(mars_nn_program_t *p,
                            mars_nn_cmd_type_t type,
                            uint32_t a,
                            uint32_t b,
                            uint32_t c,
                            uint32_t d)
{
    if (!p || !p->cmds || p->count >= p->capacity) {
        return -1;
    }

    mars_nn_cmd_t *cmd = &p->cmds[p->count++];
    cmd->type = type;
    cmd->flags = 0;
    cmd->reserved = 0;
    cmd->arg0 = a;
    cmd->arg1 = b;
    cmd->arg2 = c;
    cmd->arg3 = d;
    return 0;
}

void mars_nn_program_init(mars_nn_program_t *p,
                          mars_nn_cmd_t *storage,
                          size_t capacity)
{
    if (!p) return;
    p->cmds = storage;
    p->count = 0;
    p->capacity = capacity;
}

int mars_nn_emit_load(mars_nn_program_t *p,
                      uint32_t oram_offset,
                      uint32_t length_bytes,
                      uint32_t dst_tile)
{
    return mars_nn_push_cmd(p, MARS_NN_CMD_LOAD,
                             oram_offset, length_bytes,
                             dst_tile, 0);
}

int mars_nn_emit_mac(mars_nn_program_t *p,
                     uint32_t dst_tile,
                     uint32_t src_a_tile,
                     uint32_t src_b_tile,
                     uint32_t length)
{
    return mars_nn_push_cmd(p, MARS_NN_CMD_MAC,
                             dst_tile, src_a_tile,
                             src_b_tile, length);
}

int mars_nn_emit_store(mars_nn_program_t *p,
                       uint32_t oram_offset,
                       uint32_t length_bytes,
                       uint32_t src_tile)
{
    return mars_nn_push_cmd(p, MARS_NN_CMD_STORE,
                             oram_offset, length_bytes,
                             src_tile, 0);
}

