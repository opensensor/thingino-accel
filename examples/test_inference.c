/*
 * thingino-accel - Inference Test
 * Tests actual model inference execution
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <dlfcn.h>
#include "nna.h"
#include "nna_model.h"
#include "nna_tensor.h"

/* Minimal copies of kernel signal context structs for MIPS so we can
 * extract the faulting PC from the ucontext pointer passed to the
 * SIGBUS handler. These mirror the layout in
 * thingino-linux-opensensor/arch/mips/include/uapi/asm/sigcontext.h
 * and asm-generic/ucontext.h, but are kept local to avoid pulling in
 * kernel headers into userspace ABI.
 */
#if defined(__mips__)
typedef struct {
    unsigned int       sc_regmask;
    unsigned int       sc_status;
    unsigned long long sc_pc;
    unsigned long long sc_regs[32];
    unsigned long long sc_fpregs[32];
    unsigned int       sc_acx;
    unsigned int       sc_fpc_csr;
    unsigned int       sc_fpc_eir;
    unsigned int       sc_used_math;
    unsigned int       sc_dsp;
    unsigned long long sc_mdhi;
    unsigned long long sc_mdlo;
    unsigned long      sc_hi1;
    unsigned long      sc_lo1;
    unsigned long      sc_hi2;
    unsigned long      sc_lo2;
    unsigned long      sc_hi3;
    unsigned long      sc_lo3;
} debug_sigcontext_t;

typedef struct debug_ucontext {
    unsigned long        uc_flags;
    struct debug_ucontext *uc_link;
    stack_t              uc_stack;
    debug_sigcontext_t   uc_mcontext;
    sigset_t             uc_sigmask;
} debug_ucontext_t;
#endif

static void sigbus_handler(int sig, siginfo_t *info, void *ucontext) {
    void *pc = NULL;

#if defined(__mips__)
    debug_ucontext_t *uc = (debug_ucontext_t *)ucontext;
    if (uc) {
        pc = (void *)(uintptr_t)uc->uc_mcontext.sc_pc;
    }
#else
    (void)ucontext;
#endif

    Dl_info di;
    const char *obj = "?";
    const char *sym = "?";
    unsigned long off = 0;
    if (pc && dladdr(pc, &di)) {
        obj = di.dli_fname ? di.dli_fname : "?";
        sym = di.dli_sname ? di.dli_sname : "?";
        off = (unsigned long)((char *)pc - (char *)di.dli_saddr);
    }

    fprintf(stderr,
            "[SIGBUS] Caught SIGBUS (code=%d) at address %p, pc=%p\n"
            "        in %s: %s+0x%lx\n",
            info ? info->si_code : 0,
            info ? info->si_addr : NULL,
            pc,
            obj,
            sym,
            off);
    fflush(stderr);

    exit(128 + sig);
}

static void print_header(const char *msg) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  %-54s  ║\n", msg);
    printf("╚══════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

static void print_test(int num, const char *name) {
    printf("[%d] %s\n", num, name);
}

static void print_success(const char *msg) {
    printf("  ✓ %s\n", msg);
}

static void print_error(const char *msg) {
    printf("  ✗ %s\n", msg);
}
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.mgk>\n", argv[0]);
        return 1;
    }

    /* Install a signal handler to help debug bus/segfault errors in the .mgk path */
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = sigbus_handler;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGBUS, &sa, NULL);
    sigaction(SIGSEGV, &sa, NULL);

    const char *model_path = argv[1];
    int test_num = 1;
    int failed = 0;

    print_header("thingino-accel - Inference Test");

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
        print_error("Model load failed");
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
    print_success("Model info retrieved");

    /* Test 4: Prepare input data */
    print_test(test_num++, "Prepare input data");

    nna_tensor_t *input = nna_model_get_input(model, 0);
    if (input == NULL) {
        print_error("Failed to get input tensor");
        failed = 1;
        goto cleanup_model;
    }

    printf("      Input shape: [%d, %d, %d, %d]\n",
           input->shape.dims[0], input->shape.dims[1],
           input->shape.dims[2], input->shape.dims[3]);

    /* Calculate input size */
    size_t input_size = input->shape.dims[0] * input->shape.dims[1] *
                        input->shape.dims[2] * input->shape.dims[3];

    /* Get input data pointer and fill with test pattern */
    int8_t *input_data = (int8_t*)nna_tensor_data(input);
    if (input_data == NULL) {
        print_error("Failed to get input data pointer");
        failed = 1;
        goto cleanup_model;
    }

    /* Fill with simple test pattern */
    for (size_t i = 0; i < input_size; i++) {
        input_data[i] = (int8_t)(i % 128);
    }

    print_success("Input data prepared");

    /* Test 5: Run inference */
    print_test(test_num++, "Run inference");

    if (nna_model_run(model) != NNA_SUCCESS) {
        print_error("Inference failed");
        failed = 1;
        goto cleanup_model;
    }

    print_success("Inference completed");

    /* Test 6: Get output */
    print_test(test_num++, "Get output data");

    const nna_tensor_t *output = nna_model_get_output(model, 0);
    if (output == NULL) {
        print_error("Failed to get output tensor");
        failed = 1;
        goto cleanup_model;
    }

    printf("      Output shape: [%d, %d, %d, %d]\n",
           output->shape.dims[0], output->shape.dims[1],
           output->shape.dims[2], output->shape.dims[3]);

    print_success("Output data retrieved");

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
        print_header("✓ ALL TESTS PASSED");
        return 0;
    }
}

