/*
 * thingino-accel - Basic initialization test
 * 
 * Tests:
 * 1. NNA device initialization
 * 2. Hardware info retrieval
 * 3. Memory allocation (DDR and ORAM)
 * 4. Tensor creation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nna.h"
#include "nna_memory.h"
#include "nna_tensor.h"

#define TEST_PASS(msg) printf("  ✓ %s\n", msg)
#define TEST_FAIL(msg) printf("  ✗ %s\n", msg)

int main(int argc, char **argv) {
    int result;
    
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║         thingino-accel - Initialization Test            ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");
    printf("\n");
    
    /* Test 1: Library version */
    printf("[1/6] Library version\n");
    const char *version = nna_get_version();
    printf("      Version: %s\n", version);
    TEST_PASS("Version retrieved");
    
    /* Test 2: NNA initialization */
    printf("\n[2/6] NNA initialization\n");
    result = nna_init();
    if (result != NNA_SUCCESS) {
        TEST_FAIL("NNA initialization failed");
        printf("      Error code: %d\n", result);
        printf("      Make sure /dev/soc-nna exists and soc-nna.ko is loaded\n");
        return 1;
    }
    TEST_PASS("NNA initialized");
    
    /* Test 3: Hardware info */
    printf("\n[3/6] Hardware information\n");
    nna_hw_info_t hw_info;
    result = nna_get_hw_info(&hw_info);
    if (result != NNA_SUCCESS) {
        TEST_FAIL("Failed to get hardware info");
        nna_deinit();
        return 1;
    }
    printf("      ORAM Physical: 0x%08x\n", hw_info.oram_pbase);
    printf("      ORAM Virtual:  0x%08x\n", hw_info.oram_vbase);
    printf("      ORAM Size:     %u KB\n", hw_info.oram_size / 1024);
    printf("      NNA Version:   0x%02x\n", hw_info.version);
    TEST_PASS("Hardware info retrieved");
    
    /* Test 4: DDR memory allocation */
    printf("\n[4/6] DDR memory allocation\n");
    size_t test_size = 1024 * 1024; /* 1 MB */
    void *ddr_mem = nna_malloc(test_size);
    if (ddr_mem == NULL) {
        TEST_FAIL("DDR allocation failed");
        nna_deinit();
        return 1;
    }
    printf("      Allocated: %zu bytes @ %p\n", test_size, ddr_mem);
    
    /* Write and verify */
    memset(ddr_mem, 0xAA, test_size);
    if (((unsigned char*)ddr_mem)[0] == 0xAA) {
        TEST_PASS("DDR memory is writable");
    } else {
        TEST_FAIL("DDR memory write failed");
    }
    
    nna_free(ddr_mem);
    TEST_PASS("DDR memory freed");
    
    /* Test 5: ORAM allocation */
    printf("\n[5/6] ORAM allocation\n");
    size_t oram_size = 4096; /* 4 KB */
    void *oram_mem = nna_oram_malloc(oram_size);
    if (oram_mem == NULL) {
        TEST_FAIL("ORAM allocation failed");
        nna_deinit();
        return 1;
    }
    printf("      Allocated: %zu bytes @ %p\n", oram_size, oram_mem);
    
    /* Get ORAM stats */
    size_t total, used, free;
    nna_oram_get_stats(&total, &used, &free);
    printf("      ORAM usage: %zu / %zu KB (%.1f%%)\n",
           used / 1024, total / 1024, (used * 100.0) / total);
    TEST_PASS("ORAM allocation successful");
    
    /* Test 6: Tensor creation */
    printf("\n[6/6] Tensor operations\n");
    nna_shape_t shape = nna_shape_make(1, 224, 224, 3); /* 1x224x224x3 RGB image */
    nna_tensor_t *tensor = nna_tensor_create(&shape, NNA_DTYPE_UINT8, NNA_FORMAT_NHWC);
    if (tensor == NULL) {
        TEST_FAIL("Tensor creation failed");
        nna_deinit();
        return 1;
    }
    
    printf("      Shape: [%d, %d, %d, %d]\n",
           shape.dims[0], shape.dims[1], shape.dims[2], shape.dims[3]);
    printf("      Elements: %zu\n", nna_tensor_numel(tensor));
    printf("      Bytes: %zu\n", nna_tensor_bytes(tensor));
    TEST_PASS("Tensor created");
    
    /* Fill tensor with test data */
    unsigned char *data = (unsigned char*)nna_tensor_data(tensor);
    for (size_t i = 0; i < nna_tensor_bytes(tensor); i++) {
        data[i] = i % 256;
    }
    TEST_PASS("Tensor data written");
    
    nna_tensor_destroy(tensor);
    TEST_PASS("Tensor destroyed");
    
    /* Cleanup */
    printf("\n[CLEANUP] Shutting down\n");
    nna_deinit();
    TEST_PASS("NNA deinitialized");
    
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║              ✓ ALL TESTS PASSED!                        ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");
    printf("\n");
    printf("Next steps:\n");
    printf("  1. Implement model loading (.mgk format)\n");
    printf("  2. Add inference execution\n");
    printf("  3. Test with real models\n");
    printf("\n");
    
    return 0;
}

