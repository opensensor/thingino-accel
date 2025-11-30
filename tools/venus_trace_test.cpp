/*
 * Venus baseline test: run an OEM .mgk model via the real Magik Venus API
 * so that the NNA/AIP tracer can capture what stock firmware does.
 *
 * IMPORTANT: we load the OEM libs with dlopen() at runtime instead of
 * linking against them directly. This avoids running their static
 * initializers before main(), and lets us see exactly where things go
 * wrong (or right) on this Franken-rootfs.
 *
 * Build (from repo root, with Buildroot toolchain on PATH):
 *   PATH=$$PATH:$$HOME/output-stable/wyze_camv4_t41nq_gc4653_atbm6062/host/bin \
 *   CROSS_COMPILE=mipsel-linux- \
 *   $${CROSS_COMPILE}g++ -Wall -Wextra -O2 -std=c++14 -D_GLIBCXX_USE_CXX11_ABI=1 \
 *     -Iinclude -Isrc \
 *     -Imagik-toolkit/magik-toolkit-2.0/InferenceKit/nna2/mips720-glibc229/T41/include \
 *     tools/venus_trace_test.cpp tools/libassert_shim.c -o build/bin/venus_trace_test \
 *     -ldl -lstdc++
 *
 * Run on the T41 uClibc-ish rootfs (from /mnt/nfs):
 *   export LD_LIBRARY_PATH=/opt:/mnt/nfs/build/lib
 *   export LD_PRELOAD=/opt/libassert_shim.so
 *   ./build/bin/venus_trace_test [/mnt/nfs/AEC_T41_16K_NS_OUT_UC.mgk]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>
#include <exception>
#include <string>

#include "venus.h"  // OEM Magik Venus C++ API (types only, no direct linking)

using magik_venus_init_t   = int (*)(void);
using magik_venus_deinit_t = int (*)(void);

using namespace magik::venus;

// Function pointer types for C++ API entrypoints. We will resolve these via
// dlsym() on the OEM libvenus.so.
using net_create_fn_t = std::unique_ptr<BaseNet> (*)(TensorFormat, ShareMemoryMode);
using load_model_fn_t = int (*)(BaseNet *net, const void *model_path,
                                int memory_model, int start_off, AddressDesc *addr_desc);
using net_simple_fn_t = int (*)(BaseNet *net);

// Global stage tracker so we can see roughly where std::terminate() fires.
static const char *g_stage = "<pre-main>";

static void custom_terminate() {
    fprintf(stderr,
            "\n[venus_trace_test] std::terminate called at stage %s (likely from OEM libs)\n",
            g_stage ? g_stage : "(null)");
    // We could try to introspect std::current_exception() here, but in this
    // extremely constrained environment it's safer to just exit.
    _Exit(1);
}

struct TerminateGuard {
    TerminateGuard() { std::set_terminate(custom_terminate); }
} g_terminate_guard;

static void *must_dlopen(const char *path) {
    g_stage = "dlopen";
    void *h = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (!h) {
        fprintf(stderr, "[venus_trace_test] dlopen(%s) failed: %s\n", path, dlerror());
        exit(1);
    }
    return h;
}

template <typename T>
static T must_dlsym(void *handle, const char *symbol) {
    dlerror();
    T fn = reinterpret_cast<T>(dlsym(handle, symbol));
    const char *err = dlerror();
    if (err || !fn) {
        fprintf(stderr, "[venus_trace_test] dlsym(%s) failed: %s\n",
                symbol, err ? err : "(null)");
        exit(1);
    }
    return fn;
}

int main(int argc, char **argv) {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  Venus Trace Test (dlopen-based)                         ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    const char *default_model = "/mnt/nfs/AEC_T41_16K_NS_OUT_UC.mgk";
    const char *model_path = (argc > 1) ? argv[1] : default_model;
    printf("Using model: %s\n", model_path);

    // 1) Bring in OEM libs explicitly in a controlled order so that any
    // unresolved driver/AIP symbols in libvenus.so can bind to them.
    printf("[venus_trace_test] dlopen(/opt/libdrivers.so) ...\n");
    void *h_drivers = must_dlopen("/opt/libdrivers.so");
    (void)h_drivers;

    printf("[venus_trace_test] dlopen(/opt/libaip.so) ...\n");
    void *h_aip = must_dlopen("/opt/libaip.so");
    (void)h_aip;

    printf("[venus_trace_test] dlopen(/opt/libvenus.so) ...\n");
    void *h_venus = must_dlopen("/opt/libvenus.so");

    // 2) Resolve the C-style init/deinit first.
    printf("[venus_trace_test] resolving magik_venus_init/deinit ...\n");
    magik_venus_init_t   p_magik_venus_init   = must_dlsym<magik_venus_init_t>(h_venus, "magik_venus_init");
    magik_venus_deinit_t p_magik_venus_deinit = must_dlsym<magik_venus_deinit_t>(h_venus, "magik_venus_deinit");

    // 3) Resolve the C++ API entrypoints we actually need.
    printf("[venus_trace_test] resolving net_create + BaseNet methods ...\n");
    net_create_fn_t p_net_create = must_dlsym<net_create_fn_t>(
        h_venus,
        "_ZN5magik5venus10net_createENS0_10DataFormatENS0_15ShareMemoryModeE");

    load_model_fn_t p_load_model = must_dlsym<load_model_fn_t>(
        h_venus,
        "_ZN5magik5venus7BaseNet10load_modelEPKviiPNS0_11AddressDescE");

    net_simple_fn_t p_net_init = must_dlsym<net_simple_fn_t>(
        h_venus,
        "_ZN5magik5venus7BaseNet4initEv");

    net_simple_fn_t p_net_run = must_dlsym<net_simple_fn_t>(
        h_venus,
        "_ZN5magik5venus7BaseNet3runEv");

    net_simple_fn_t p_net_deinit = must_dlsym<net_simple_fn_t>(
        h_venus,
        "_ZN5magik5venus7BaseNet6deinitEv");

    try {
        g_stage = "venus_init";
        int ret = p_magik_venus_init();
        printf("magik_venus_init() returned: %d\n", ret);
        if (ret != 0) {
            printf("ERROR: magik_venus_init failed, aborting.\n");
            return 1;
        }

        g_stage = "net_create";
        auto net = p_net_create(TensorFormat::NHWC, ShareMemoryMode::DEFAULT);
        if (!net) {
            printf("ERROR: net_create() returned null.\n");
            p_magik_venus_deinit();
            return 1;
        }

        g_stage = "load_model";
        printf("Calling BaseNet::load_model(\"%s\", memory_model=0 PATH) ...\n", model_path);
        ret = p_load_model(net.get(), model_path, /*memory_model=*/0,
                           /*start_off=*/0, /*addr_desc=*/nullptr);
        printf("load_model returned: %d\n", ret);
        if (ret != 0) {
            printf("ERROR: load_model failed, aborting.\n");
            net.reset();
            p_magik_venus_deinit();
            return 1;
        }

        g_stage = "init";
        printf("Calling BaseNet::init() ...\n");
        ret = p_net_init(net.get());
        printf("init returned: %d\n", ret);
        if (ret != 0) {
            printf("ERROR: init failed, aborting.\n");
            net.reset();
            p_magik_venus_deinit();
            return 1;
        }

        // For baseline tracing we do a single forward pass; inputs will be
        // whatever the OEM stack defaults to (likely zeros). We care about
        // NNDMA/AIP side effects more than numerical outputs here.
        g_stage = "run";
        printf("Calling BaseNet::run() ...\n");
        ret = p_net_run(net.get());
        printf("run returned: %d\n", ret);

        g_stage = "deinit";
        printf("Calling BaseNet::deinit() ...\n");
        int ret_deinit = p_net_deinit(net.get());
        printf("BaseNet::deinit returned: %d\n", ret_deinit);

        net.reset();

        g_stage = "venus_deinit";
        int ret_venus_deinit = p_magik_venus_deinit();
        printf("magik_venus_deinit() returned: %d\n", ret_venus_deinit);

        printf("\n✓ Venus baseline test complete\n");
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "\n[venus_trace_test] EXCEPTION at stage %s: %s\n",
                g_stage ? g_stage : "(null)", e.what());
        return 1;
    } catch (...) {
        fprintf(stderr, "\n[venus_trace_test] UNKNOWN EXCEPTION at stage %s\n",
                g_stage ? g_stage : "(null)");
        return 1;
    }
}
