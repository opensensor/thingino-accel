/*
 * thingino-accel - Model Loader for .mgk files
 * Instantiates DerivedMagikModel from compiled .mgk shared libraries
 */

#include "model_loader.h"
#include "magik_model.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstring>

/* C linkage for functions called from C code */
extern "C" {
    void* load_mgk_model(const char *path);
    int run_mgk_model(void *handle);
    void unload_mgk_model(void *handle);
    void* get_mgk_model_instance(void *handle);
}

namespace magik {
namespace venus {

/* Type for DerivedMagikModel constructor
 * Note: Third parameter is void*& (reference) - constructor may modify it
 */
typedef MagikModelBase* (*ModelConstructor)(long long, long long, void*&, void*, void*,
                                             ModelMemoryInfoManager::MemAllocMode,
                                             MagikModelBase::ModuleMode);

/* Model loader implementation */
struct ModelLoader {
    void *dl_handle;
    MagikModelBase *model_instance;
    
    ModelLoader() : dl_handle(nullptr), model_instance(nullptr) {}
    
    ~ModelLoader() {
        if (model_instance) {
            delete model_instance;
        }
        if (dl_handle) {
            dlclose(dl_handle);
        }
    }
};

} // namespace venus
} // namespace magik

/* C-linkage wrapper implementations */
using namespace magik::venus;

/* External runtime variables */
extern "C" {
    extern void *__oram_vbase;
    extern void *__ddr_vbase;
}

extern "C" {

void* load_mgk_model(const char *path) {
    if (!path) {
        fprintf(stderr, "load_mgk_model: NULL path\n");
        return nullptr;
    }

    ModelLoader *loader = new ModelLoader();

    /* Load the .mgk shared library */
    printf("load_mgk_model: Loading %s with dlopen...\n", path);
    fflush(stdout);

    loader->dl_handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);

    printf("load_mgk_model: dlopen returned %p\n", loader->dl_handle);
    fflush(stdout);

    if (!loader->dl_handle) {
        fprintf(stderr, "load_mgk_model: dlopen failed: %s\n", dlerror());
        delete loader;
        return nullptr;
    }

    printf("load_mgk_model: dlopen succeeded\n");
    fflush(stdout);

    /* Look for DerivedMagikModel constructor */
    printf("load_mgk_model: Looking for constructor symbol...\n");
    fflush(stdout);

    ModelConstructor ctor = nullptr;

    const char *ctor_names[] = {
        /* Actual symbol from AEC_T41_16K_NS_OUT_UC.mgk */
        "_ZN17DerivedMagikModelC1ExxPvS0_S0_N5magik5venus22ModelMemoryInfoManager12MemAllocModeENS2_14MagikModelBase10ModuleModeE",
        "_ZN17DerivedMagikModelC2ExxPvS0_S0_N5magik5venus22ModelMemoryInfoManager12MemAllocModeENS2_14MagikModelBase10ModuleModeE",
        /* Try other possible variations */
        "_ZN18DerivedMagikModelC1ExxPvS0_S0_N5magik5venus25ModelMemoryInfoManager13MemAllocModeENS1_14MagikModelBase10ModuleModeE",
        "_ZN18DerivedMagikModelC2ExxPvS0_S0_N5magik5venus25ModelMemoryInfoManager13MemAllocModeENS1_14MagikModelBase10ModuleModeE",
        nullptr
    };

    for (int i = 0; ctor_names[i] != nullptr; i++) {
        printf("load_mgk_model: Trying symbol %d...\n", i);
        fflush(stdout);

        ctor = (ModelConstructor)dlsym(loader->dl_handle, ctor_names[i]);

        printf("load_mgk_model: dlsym returned %p\n", (void*)ctor);
        fflush(stdout);

        if (ctor) {
            printf("Found model constructor at index %d: %p\n", i, (void*)ctor);
            break;
        }
    }

    if (!ctor) {
        fprintf(stderr, "load_mgk_model: Could not find DerivedMagikModel constructor\n");
        delete loader;
        return nullptr;
    }

    printf("Calling constructor with:\n");
    printf("  __oram_vbase = %p\n", __oram_vbase);
    printf("  __ddr_vbase = %p\n", __ddr_vbase);

    /* Instantiate the model
     * Note: param3 is a reference and may be modified by the constructor
     */
    void *oram_base = __oram_vbase;

    try {
        loader->model_instance = ctor(
            0,  /* param1 */
            0,  /* param2 */
            oram_base,  /* ORAM base (reference - may be modified) */
            __ddr_vbase,   /* DDR base */
            nullptr,  /* param5 */
            ModelMemoryInfoManager::MemAllocMode::DEFAULT,
            MagikModelBase::ModuleMode::NORMAL
        );

        if (!loader->model_instance) {
            fprintf(stderr, "load_mgk_model: Constructor returned NULL\n");
            delete loader;
            return nullptr;
        }

        printf("Model instance created successfully at %p\n", loader->model_instance);
        printf("After construction, oram_base = %p (may have been modified)\n", oram_base);

    } catch (const std::exception &e) {
        fprintf(stderr, "load_mgk_model: Exception: %s\n", e.what());
        delete loader;
        return nullptr;
    } catch (...) {
        fprintf(stderr, "load_mgk_model: Unknown exception\n");
        delete loader;
        return nullptr;
    }

    return loader;
}

int run_mgk_model(void *handle) {
    if (!handle) {
        return -1;
    }

    ModelLoader *loader = static_cast<ModelLoader*>(handle);
    if (!loader->model_instance) {
        return -1;
    }

    try {
        return loader->model_instance->run();
    } catch (const std::exception &e) {
        fprintf(stderr, "run_mgk_model: Exception: %s\n", e.what());
        return -1;
    } catch (...) {
        fprintf(stderr, "run_mgk_model: Unknown exception\n");
        return -1;
    }
}

void* get_mgk_model_instance(void *handle) {
    if (!handle) {
        return nullptr;
    }

    ModelLoader *loader = static_cast<ModelLoader*>(handle);
    return loader->model_instance;
}

void unload_mgk_model(void *handle) {
    if (handle) {
        delete static_cast<ModelLoader*>(handle);
    }
}

} // extern "C"

