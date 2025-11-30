/*
 * Mars Runtime Test
 *
 * Load and run a .mars model file
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mars.h"
#include "mars_runtime.h"
#include "nna.h"

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s <model.mars>\n", prog);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char *model_path = argv[1];

    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║  Mars Runtime Test                                       ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    /* Initialize NNA */
    printf("Initializing NNA...\n");
    int ret = nna_init();
    if (ret != NNA_SUCCESS) {
        fprintf(stderr, "NNA init failed: %d\n", ret);
        return 1;
    }

    /* Load model */
    printf("Loading model: %s\n", model_path);
    mars_model_t *model = NULL;
    mars_error_t err = mars_load_file(model_path, &model);
    if (err != MARS_OK) {
        fprintf(stderr, "Failed to load model: %s\n", mars_get_error_string(err));
        nna_deinit();
        return 1;
    }

    /* Print summary */
    mars_print_summary(model);

    /* Get input tensor */
    mars_runtime_tensor_t *input = mars_get_input(model, 0);
    if (!input) {
        fprintf(stderr, "Failed to get input tensor\n");
        mars_free(model);
        nna_deinit();
        return 1;
    }

    printf("Input tensor:\n");
    printf("  Name: %s\n", input->desc.name);
    printf("  Shape: [");
    for (uint32_t i = 0; i < input->desc.ndims; i++) {
        printf("%d%s", input->desc.shape[i], i < input->desc.ndims - 1 ? ", " : "");
    }
    printf("]\n");
    printf("  Data type: %s\n", input->desc.dtype == MARS_DTYPE_FLOAT32 ? "float32" : "int8");
    printf("  Data: vaddr=%p paddr=%p\n", input->vaddr, input->paddr);

    /* Fill input with test pattern */
    if (input->vaddr) {
        size_t input_size = input->alloc_size;
        printf("  Filling %zu bytes with test pattern...\n", input_size);
        if (input->desc.dtype == MARS_DTYPE_FLOAT32) {
            float *p = (float *)input->vaddr;
            size_t num_floats = input_size / sizeof(float);
            for (size_t i = 0; i < num_floats; i++) {
                p[i] = (float)(i % 256) / 255.0f;  /* 0.0-1.0 normalized pattern */
            }
        } else {
            int8_t *p = (int8_t *)input->vaddr;
            for (size_t i = 0; i < input_size; i++) {
                p[i] = (int8_t)(i % 127);  /* 0-126 incrementing pattern */
            }
        }
    }

    /* Run inference */
    printf("\nRunning inference...\n");
    err = mars_run(model);
    if (err != MARS_OK) {
        fprintf(stderr, "Inference failed: %s\n", mars_get_error_string(err));
        mars_free(model);
        nna_deinit();
        return 1;
    }

    /* Get output */
    mars_runtime_tensor_t *output = mars_get_output(model, 0);
    if (output && output->vaddr) {
        printf("\nOutput tensor:\n");
        printf("  Name: %s\n", output->desc.name);
        printf("  Shape: [");
        for (uint32_t i = 0; i < output->desc.ndims; i++) {
            printf("%d%s", output->desc.shape[i], i < output->desc.ndims - 1 ? ", " : "");
        }
        printf("]\n");
        printf("  Data type: %s\n", output->desc.dtype == MARS_DTYPE_FLOAT32 ? "float32" : "int8");

        /* Print first few values based on data type */
        if (output->desc.dtype == MARS_DTYPE_FLOAT32) {
            float *data = (float *)output->vaddr;
            size_t num_floats = output->alloc_size / sizeof(float);
            float min_val = data[0], max_val = data[0], sum = 0;
            int nonzero = 0;
            for (size_t i = 0; i < num_floats; i++) {
                if (data[i] < min_val) min_val = data[i];
                if (data[i] > max_val) max_val = data[i];
                sum += data[i];
                if (data[i] != 0.0f) nonzero++;
            }
            printf("  Stats: min=%.4f max=%.4f mean=%.4f nonzero=%d/%zu\n",
                   min_val, max_val, sum / num_floats, nonzero, num_floats);
            printf("  First 16: ");
            for (int i = 0; i < 16; i++) printf("%.3f ", data[i]);
            printf("\n  Sample @1000: ");
            for (int i = 1000; i < 1016 && i < (int)num_floats; i++) printf("%.3f ", data[i]);
            printf("\n");
        } else {
            int8_t *data = (int8_t *)output->vaddr;
            printf("  First 16 values: ");
            for (int i = 0; i < 16 && i < (int)output->alloc_size; i++) {
                printf("%d ", data[i]);
            }
            printf("\n");
        }
    }

    /* Cleanup */
    printf("\nCleaning up...\n");
    mars_free(model);
    nna_deinit();

    printf("Done!\n");
    return 0;
}

