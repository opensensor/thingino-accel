/*
 * thingino-accel - Magik Model/Layer Stubs
 * These are stub implementations for the Magik model framework
 */

#include "venus_types.h"
#include "tensor.h"
#include "../../include/nna_memory.h"
#include <vector>
#include <string>
#include <cstdio>

namespace magik {
namespace venus {

/* Forward declarations */
class TensorXWrapper;
class MagikLayerBase;

/* Pyramid config */
struct PyramidConfig {
    int level;
    int width;
    int height;
};

/* Tensor info */
struct TensorInfo {
    std::string name;
    shape_t shape;
    DataType dtype;
    TensorFormat format;
};

/* Model memory info manager */
class ModelMemoryInfoManager {
public:
    enum class MemAllocMode {
        DEFAULT = 0,
        SHARE_ONE_THREAD = 1,
        ALL_SEPARABLE_MEM = 3,
        SMART_REUSE_MEM = 4,
    };
};

/* Module mode */
enum class ModuleMode {
    NORMAL = 0,
    DEBUG = 1,
};

/* Device class stub */
class Device {
public:
    Device() {}
    ~Device() {}
};

/* TensorX wrapper */
class TensorXWrapper {
public:
    TensorX *tensorx;

    TensorXWrapper() : tensorx(nullptr) {}
    TensorXWrapper(TensorX *tx) : tensorx(tx) {}
    ~TensorXWrapper() {}
};

/* TensorX constructor with size, data pointer, and device */
extern "C" void _ZN5magik5venus7TensorXC1EjPvNS0_6DeviceE(TensorX *this_ptr, uint32_t size, void *data_ptr, Device dev) {
    (void)dev;
    new (this_ptr) TensorX();
    this_ptr->data = data_ptr;
    this_ptr->bytes = size;
    this_ptr->owns_data = false;
}

/* TensorX mudata method */
extern "C" void* _ZN5magik5venus7TensorX6mudataEjj(TensorX *this_ptr, uint32_t offset, uint32_t size) {
    (void)size;
    if (!this_ptr->data) {
        return nullptr;
    }
    return (char*)this_ptr->data + offset;
}

/* Core ORAM functions */
namespace core {
    void* oram_malloc(size_t size) {
        /* Use NNA memory allocator */
        return nna_malloc(size);
    }
    
    void oram_free(void *ptr) {
        nna_free(ptr);
    }
} // namespace core

/* MagikLayerBase */
class MagikLayerBase {
public:
    MagikLayerBase() {}
    virtual ~MagikLayerBase() {}
    
    virtual void set_inputs(std::vector<TensorXWrapper*> inputs) {
        (void)inputs;
    }
    
    virtual void set_outputs(std::vector<TensorXWrapper*> outputs) {
        (void)outputs;
    }
    
    virtual void _flush_cache(std::vector<TensorXWrapper*> tensors) {
        (void)tensors;
    }
};

/* MagikModelBase */
class MagikModelBase {
public:
    MagikModelBase(long long param1, long long param2, void **param3, void *param4,
                   ModelMemoryInfoManager::MemAllocMode mode, ModuleMode module_mode) {
        (void)param1; (void)param2; (void)param3; (void)param4; (void)mode; (void)module_mode;
    }
    
    virtual ~MagikModelBase() {}
    
    virtual int run() {
        printf("MagikModelBase::run() - stub\n");
        return 0;
    }
    
    virtual int reshape() { return 0; }
    virtual int pre_graph_run() { return 0; }
    virtual int free_forward_memory() { return 0; }
    virtual int free_inputs_memory() { return 0; }
    virtual int open_mnni_debug() { return 0; }
    virtual int open_mnni_profiler() { return 0; }
    virtual int set_main_pyramid_config(int level) { (void)level; return 0; }
    virtual int create_and_add_pyramid_config() { return 0; }
    
    virtual int build_tensors(PyramidConfig *config, std::vector<TensorInfo> infos) {
        (void)config; (void)infos;
        return 0;
    }
    
    virtual int update_cache_buffer_ptr(std::vector<MagikLayerBase*> layers, void *ptr) {
        (void)layers; (void)ptr;
        return 0;
    }
};

} // namespace venus
} // namespace magik

/* Extern C wrappers for mangled symbols */
extern "C" {

using namespace magik::venus;

/* MagikModelBase constructor */
void _ZN5magik5venus14MagikModelBaseC2ExxRPvS2_NS0_22ModelMemoryInfoManager12MemAllocModeENS1_10ModuleModeE(
    MagikModelBase *this_ptr, long long p1, long long p2, void **p3, void *p4,
    ModelMemoryInfoManager::MemAllocMode mode, ModuleMode mmode) {
    new (this_ptr) MagikModelBase(p1, p2, p3, p4, mode, mmode);
}

/* MagikModelBase destructor */
void _ZN5magik5venus14MagikModelBaseD2Ev(MagikModelBase *this_ptr) {
    this_ptr->~MagikModelBase();
}

/* MagikModelBase methods */
int _ZN5magik5venus14MagikModelBase3runEv(MagikModelBase *this_ptr) {
    return this_ptr->run();
}

int _ZN5magik5venus14MagikModelBase7reshapeEv(MagikModelBase *this_ptr) {
    return this_ptr->reshape();
}

int _ZN5magik5venus14MagikModelBase13pre_graph_runEv(MagikModelBase *this_ptr) {
    return this_ptr->pre_graph_run();
}

int _ZN5magik5venus14MagikModelBase19free_forward_memoryEv(MagikModelBase *this_ptr) {
    return this_ptr->free_forward_memory();
}

int _ZN5magik5venus14MagikModelBase18free_inputs_memoryEv(MagikModelBase *this_ptr) {
    return this_ptr->free_inputs_memory();
}

int _ZN5magik5venus14MagikModelBase15open_mnni_debugEv(MagikModelBase *this_ptr) {
    return this_ptr->open_mnni_debug();
}

int _ZN5magik5venus14MagikModelBase18open_mnni_profilerEv(MagikModelBase *this_ptr) {
    return this_ptr->open_mnni_profiler();
}

int _ZN5magik5venus14MagikModelBase23set_main_pyramid_configEi(MagikModelBase *this_ptr, int level) {
    return this_ptr->set_main_pyramid_config(level);
}

int _ZN5magik5venus14MagikModelBase29create_and_add_pyramid_configEv(MagikModelBase *this_ptr) {
    return this_ptr->create_and_add_pyramid_config();
}

int _ZN5magik5venus14MagikModelBase13build_tensorsEPNS1_13PyramidConfigESt6vectorINS0_10TensorInfoESaIS5_EE(
    MagikModelBase *this_ptr, PyramidConfig *config, std::vector<TensorInfo> infos) {
    return this_ptr->build_tensors(config, infos);
}

int _ZN5magik5venus14MagikModelBase23update_cache_buffer_ptrESt6vectorIPNS0_14MagikLayerBaseESaIS4_EEPv(
    MagikModelBase *this_ptr, std::vector<MagikLayerBase*> layers, void *ptr) {
    return this_ptr->update_cache_buffer_ptr(layers, ptr);
}

/* MagikLayerBase methods */
void _ZN5magik5venus14MagikLayerBase10set_inputsESt6vectorIPNS0_14TensorXWrapperESaIS4_EE(
    MagikLayerBase *this_ptr, std::vector<TensorXWrapper*> inputs) {
    this_ptr->set_inputs(inputs);
}

void _ZN5magik5venus14MagikLayerBase11set_outputsESt6vectorIPNS0_14TensorXWrapperESaIS4_EE(
    MagikLayerBase *this_ptr, std::vector<TensorXWrapper*> outputs) {
    this_ptr->set_outputs(outputs);
}

void _ZN5magik5venus14MagikLayerBase12_flush_cacheESt6vectorIPNS0_14TensorXWrapperESaIS4_EE(
    MagikLayerBase *this_ptr, std::vector<TensorXWrapper*> tensors) {
    this_ptr->_flush_cache(tensors);
}

/* Core ORAM functions */
void* _ZN5magik5venus4core11oram_mallocEj(uint32_t size) {
    return core::oram_malloc(size);
}

void _ZN5magik5venus4core9oram_freeEPv(void *ptr) {
    core::oram_free(ptr);
}

} /* extern "C" */

