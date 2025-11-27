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

/* NOTE: The OEM .mgk exposes a C factory function `create` that, based on
 * reverse engineering, wraps `DerivedMagikModel::DerivedMagikModel` with the
 * following signature:
 *
 *   DerivedMagikModel(long long, long long,
 *                     void*, void*, void*,
 *                     ModelMemoryInfoManager::MemAllocMode,
 *                     MagikModelBase::ModuleMode);
 *
 * The C `create` factory uses the *same* parameter list (minus `this`) and
 * simply forwards all arguments into the derived constructor. Our call site in
 * this file must therefore use an identical function type; using a smaller
 * or different signature leads to arguments being misaligned and random stack
 * data being interpreted as pointers / enum values inside the model.
 */
typedef MagikModelBase* (*CreateFunction)(
    long long param1,
    long long param2,
    void *param3,
    void *param4,
    void *param5,
    ModelMemoryInfoManager::MemAllocMode mem_mode,
    MagikModelBase::ModuleMode module_mode);

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

    /* Look for the 'create' function - .mgk models export this as a C function */
    printf("load_mgk_model: Looking for 'create' function...\n");
    fflush(stdout);

    /* Look up the C `create` factory exported by the .mgk. Its true
     * signature mirrors the OEM-derived constructor (see the
     * magik::venus::CreateFunction typedef above).
     */
    CreateFunction create_fn = (CreateFunction)dlsym(loader->dl_handle, "create");

    if (!create_fn) {
        fprintf(stderr, "load_mgk_model: Could not find 'create' function: %s\n", dlerror());
        dlclose(loader->dl_handle);
        delete loader;
        return nullptr;
    }

    printf("Found 'create' function at %p\n", (void*)create_fn);
    fflush(stdout);

    /* Call the create function to instantiate the model. We now pass
     * a full, well-defined argument set that matches the OEM-derived
     * signature instead of relying on an underspecified 4-arg shim.
     */
    long long param1 = 0;
    long long param2 = 0;
    void *param3 = __oram_vbase;
    void *param4 = __ddr_vbase;
    void *param5 = nullptr;
    ModelMemoryInfoManager::MemAllocMode mem_mode =
        ModelMemoryInfoManager::MemAllocMode::DEFAULT;
    MagikModelBase::ModuleMode module_mode = MagikModelBase::ModuleMode::NORMAL;

    printf("Calling create(param1=%lld, param2=%lld, oram=%p, ddr=%p, extra=%p, mem_mode=%d, module_mode=%d)"\
           "...\n",
           (long long)param1, (long long)param2,
           param3, param4, param5,
           (int)mem_mode, (int)module_mode);
    fflush(stdout);

    try {
        printf("About to call create_fn...\n");
        fflush(stdout);

        loader->model_instance = create_fn(param1, param2,
                                           param3, param4, param5,
                                           mem_mode, module_mode);

        printf("create() returned: %p\n", (void*)loader->model_instance);
        fflush(stdout);

        printf("After fflush, before NULL check\n");
        fflush(stdout);

        if (!loader->model_instance) {
            fprintf(stderr, "load_mgk_model: create() returned NULL\n");
            dlclose(loader->dl_handle);
            delete loader;
            return nullptr;
        }

        printf("Model instance created at %p\n", loader->model_instance);
        fflush(stdout);

    } catch (const std::bad_alloc &e) {
        fprintf(stderr, "load_mgk_model: bad_alloc Exception: %s\n", e.what());
        fflush(stderr);
        delete loader;
        return nullptr;
    } catch (const std::exception &e) {
        fprintf(stderr, "load_mgk_model: Exception: %s\n", e.what());
        fflush(stderr);
        delete loader;
        return nullptr;
    } catch (...) {
        fprintf(stderr, "load_mgk_model: Unknown exception\n");
        fflush(stderr);
        delete loader;
        return nullptr;
    }

    printf("load_mgk_model: About to return loader=%p\n", (void*)loader);
    fflush(stdout);

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
    if (!handle) {
        return;
    }

    ModelLoader *loader = static_cast<ModelLoader*>(handle);

    /* Call the 'destroy' function if available */
    if (loader->dl_handle && loader->model_instance) {
        typedef void (*DestroyFunction)(MagikModelBase*);
        DestroyFunction destroy_fn = (DestroyFunction)dlsym(loader->dl_handle, "destroy");

        if (destroy_fn) {
            printf("Calling destroy() on model instance\n");
            destroy_fn(loader->model_instance);
        }
    }

    delete loader;
}

} // extern "C"

