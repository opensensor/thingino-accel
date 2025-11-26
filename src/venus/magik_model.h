/*
 * thingino-accel - Magik Model/Layer Headers
 */

#ifndef MAGIK_MODEL_H
#define MAGIK_MODEL_H

#include "venus_types.h"
#include "tensor.h"
#include <vector>
#include <string>

namespace magik {
namespace venus {

/* Forward declarations */
class TensorXWrapper;
class MagikLayerBase;

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

/* Device class stub */
class Device {
public:
    Device();
    ~Device();
};

/* TensorX wrapper */
class TensorXWrapper {
public:
    TensorX *tensorx;
    
    TensorXWrapper();
    TensorXWrapper(TensorX *tx);
    ~TensorXWrapper();
};

/* Core ORAM functions */
namespace core {
    void* oram_malloc(size_t size);
    void oram_free(void *ptr);
} // namespace core

/* MagikLayerBase */
class MagikLayerBase {
public:
    MagikLayerBase();
    virtual ~MagikLayerBase();

    virtual void set_inputs(std::vector<TensorXWrapper*> inputs);
    virtual void set_outputs(std::vector<TensorXWrapper*> outputs);
    virtual void _flush_cache(std::vector<TensorXWrapper*> tensors);
    virtual std::vector<TensorXWrapper*> get_inputs() const;
    virtual std::vector<TensorXWrapper*> get_outputs() const;
    virtual std::vector<TensorXWrapper*> get_input_wrappers() const;
    virtual std::vector<TensorXWrapper*> get_output_wrappers() const;
    virtual std::string get_name() const;
    virtual int get_layer_id() const;
};

/* MagikModelBase */
class MagikModelBase {
public:
    /* Module mode - nested enum */
    enum class ModuleMode {
        NORMAL = 0,
        DEBUG = 1,
    };

    /* Pyramid configuration - nested class */
    struct PyramidConfig {
        int level;
        int width;
        int height;

        TensorXWrapper* get_tensor_wrapper(std::string &name) const;
    };

    MagikModelBase(long long param1, long long param2, void *&param3, void *param4,
                   ModelMemoryInfoManager::MemAllocMode mode, ModuleMode module_mode);
    virtual ~MagikModelBase();

    virtual int run();
    virtual int reshape();
    virtual int pre_graph_run();
    virtual int free_forward_memory();
    virtual int free_inputs_memory();
    virtual int open_mnni_debug();
    virtual int open_mnni_profiler();
    virtual int set_main_pyramid_config(int level);
    virtual int create_and_add_pyramid_config();

    virtual int build_tensors(PyramidConfig *config, std::vector<TensorInfo> infos);
    virtual int update_cache_buffer_ptr(std::vector<MagikLayerBase*> layers, void *ptr);
    virtual int set_oram_address(void *addr, long long size) const;
    virtual std::string get_output_names() const;
    virtual std::string get_input_names() const;
    virtual TensorXWrapper* get_input(std::string &name) const;
    virtual TensorXWrapper* get_input(int index) const;
    virtual TensorXWrapper* get_output(std::string &name) const;
    virtual TensorXWrapper* get_output(int index) const;
    virtual size_t get_forward_memory_size() const;
};

} // namespace venus
} // namespace magik

#endif /* MAGIK_MODEL_H */

