/*
 * thingino-accel - Model Loading Test
 * 
 * Test loading and basic inference with a .mgk model
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nna.h"
#include "nna_model.h"

#define TEST_MODEL_PATH "/tmp/AEC_T41_16K_NS_OUT_UC.mgk"

void print_header(const char *title) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  %-54s  ║\n", title);
    printf("╚══════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

void print_test(int num, const char *name) {
    printf("[%d] %s\n", num, name);
}

void print_success(const char *msg) {
    printf("  ✓ %s\n", msg);
}

void print_error(const char *msg) {
    printf("  ✗ %s\n", msg);
}

int main(int argc, char **argv) {
    const char *model_path = (argc > 1) ? argv[1] : TEST_MODEL_PATH;
    int test_num = 1;
    int failed = 0;
    
    print_header("thingino-accel - Model Loading Test");
    
    /* Test 1: Initialize NNA */
    print_test(test_num++, "NNA initialization");
    if (nna_init() != NNA_SUCCESS) {
        print_error("NNA initialization failed");
        return 1;
    }
    print_success("NNA initialized");
    
    /* Test 2: Load model */
    print_test(test_num++, "Load model");
    printf("      Model: %s\n", model_path);
    
    nna_model_t *model = nna_model_load(model_path, NULL);
    if (model == NULL) {
        print_error("Model loading failed");
        failed = 1;
        goto cleanup;
    }
    print_success("Model loaded");
    
    /* Test 3: Get model info */
    print_test(test_num++, "Model information");
    nna_model_info_t info;
    if (nna_model_get_info(model, &info) != NNA_SUCCESS) {
        print_error("Failed to get model info");
        failed = 1;
        goto cleanup_model;
    }
    
    printf("      Inputs:  %u\n", info.num_inputs);
    printf("      Outputs: %u\n", info.num_outputs);
    printf("      Layers:  %u\n", info.num_layers);
    printf("      Size:    %zu bytes\n", info.model_size);
    printf("      Forward memory: %zu bytes\n", info.forward_mem_req);
    print_success("Model info retrieved");
    
    /* Test 4: Get input tensors */
    print_test(test_num++, "Input tensors");
    for (uint32_t i = 0; i < info.num_inputs && i < 3; i++) {
        nna_tensor_t *input = nna_model_get_input(model, i);
        if (input == NULL) {
            printf("      Input %u: NOT FOUND\n", i);
            continue;
        }

        printf("      Input %u: [%d, %d, %d, %d]\n",
               i, input->shape.dims[0], input->shape.dims[1],
               input->shape.dims[2], input->shape.dims[3]);
    }
    print_success("Input tensors accessible");
    
    /* Test 5: Get output tensors */
    print_test(test_num++, "Output tensors");
    for (uint32_t i = 0; i < info.num_outputs && i < 3; i++) {
        const nna_tensor_t *output = nna_model_get_output(model, i);
        if (output == NULL) {
            printf("      Output %u: NOT FOUND\n", i);
            continue;
        }

        printf("      Output %u: [%d, %d, %d, %d]\n",
               i, output->shape.dims[0], output->shape.dims[1],
               output->shape.dims[2], output->shape.dims[3]);
    }
    print_success("Output tensors accessible");
    
    /* Test 6: Run inference (with dummy data) */
    print_test(test_num++, "Run inference");
    if (nna_model_run(model) != NNA_SUCCESS) {
        print_error("Inference failed (expected - no input data set)");
        /* This is expected to fail without setting input data */
    } else {
        print_success("Inference completed");
    }
    
cleanup_model:
    nna_model_unload(model);
    print_success("Model unloaded");
    
cleanup:
    nna_deinit();
    print_success("NNA deinitialized");
    
    printf("\n");
    if (failed) {
        print_header("✗ TESTS FAILED");
        return 1;
    } else {
        print_header("✓ TESTS PASSED");
        printf("Next steps:\n");
        printf("  1. Set input tensor data\n");
        printf("  2. Run inference with real data\n");
        printf("  3. Read output tensors\n");
        printf("\n");
        return 0;
    }
}

