/*
 * Mars Math Test
 *
 * Simple vector and matrix math tests using Mars helpers. This runs on
 * CPU (and MXU on MIPS where initialized) and also builds a tiny
 * abstract NN command program describing an equivalent operation.
 */

#include <stdio.h>
#include <math.h>

#include "mars_math.h"
#include "mars_nn_cmd.h"

static int almost_equal(float a, float b, float eps)
{
    float diff = a - b;
    if (diff < 0.0f) diff = -diff;
    return diff <= eps;
}

int main(void)
{
    printf("Mars Math Test (vector/matrix)\n");

    /* Vector add test */
    float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float c[4] = {0.0f};

    mars_vec_add_f32(c, a, b, 4);

    printf("vec_add: [");
    for (int i = 0; i < 4; i++) {
        printf("%g%s", c[i], (i < 3) ? ", " : "]\n");
    }

    float expected_add[4] = {6.0f, 8.0f, 10.0f, 12.0f};
    int ok_add = 1;
    for (int i = 0; i < 4; i++) {
        if (!almost_equal(c[i], expected_add[i], 1e-5f)) {
            ok_add = 0;
        }
    }
    printf("vec_add %s\n", ok_add ? "OK" : "FAIL");

    /* Dot-product test */
    float dot = mars_vec_dot_f32(a, b, 4);
    printf("dot(a,b) = %g (expected 70)\n", dot);
    printf("dot %s\n", almost_equal(dot, 70.0f, 1e-4f) ? "OK" : "FAIL");

    /* Small matrix multiply: A[2x3], B[3x2] */
    float A[6] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    float Bm[6] = {
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
    };
    float C[4] = {0.0f};

    mars_matmul_f32(C, A, Bm, 2, 3, 2);

    printf("matmul C: [");
    for (int i = 0; i < 4; i++) {
        printf("%g%s", C[i], (i < 3) ? ", " : "]\n");
    }

    /* Expected C = A * Bm */
    float C_exp[4] = {
        58.0f, 64.0f,
        139.0f, 154.0f
    };
    int ok_mm = 1;
    for (int i = 0; i < 4; i++) {
        if (!almost_equal(C[i], C_exp[i], 1e-3f)) {
            ok_mm = 0;
        }
    }
    printf("matmul %s\n", ok_mm ? "OK" : "FAIL");

    /* Build a tiny conceptual NN program for the dot product.
     * This does not execute on hardware yet; it is just a structured
     * description of what we expect an NNA sequence to look like. */
    mars_nn_cmd_t buf[8];
    mars_nn_program_t prog;
    mars_nn_program_init(&prog, buf, 8);

    /* Pretend that a and b each live in ORAM at different offsets. */
    (void)mars_nn_emit_load(&prog, /*oram_offset=*/0,       /*len=*/sizeof(a), /*dst_tile=*/0);
    (void)mars_nn_emit_load(&prog, /*oram_offset=*/0x1000,  /*len=*/sizeof(b), /*dst_tile=*/1);
    (void)mars_nn_emit_mac(&prog,  /*dst_tile=*/2,
                           /*src_a_tile=*/0,
                           /*src_b_tile=*/1,
                           /*length=*/4);
    (void)mars_nn_emit_store(&prog,/*oram_offset=*/0x2000,
                             /*len=*/sizeof(float),
                             /*src_tile=*/2);

    printf("NN program length: %zu commands\n", prog.count);
    for (size_t i = 0; i < prog.count; i++) {
        const mars_nn_cmd_t *cmd = &prog.cmds[i];
        printf("  cmd[%zu]: type=%d a0=0x%08x a1=0x%08x a2=0x%08x a3=0x%08x\n",
               i, (int)cmd->type,
               cmd->arg0, cmd->arg1, cmd->arg2, cmd->arg3);
    }

    return (ok_add && almost_equal(dot, 70.0f, 1e-4f) && ok_mm) ? 0 : 1;
}

