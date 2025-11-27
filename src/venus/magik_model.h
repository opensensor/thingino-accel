/*
 * thingino-accel - Magik Model/Layer Headers
 */

#ifndef MAGIK_MODEL_H
#define MAGIK_MODEL_H

#include "venus_types.h"
#include "tensor.h"
#include <vector>
#include <string>
#include <map>

namespace magik {
namespace venus {

/* Forward declarations */
class TensorXWrapper;
class MagikLayerBase;

/* Tensor info - layout must match OEM TensorInfo (size 0x70) */
struct TensorInfo {
    std::string name;      // 0x00: tensor name

    // Flags indicating whether this tensor is model input/output.
    // These bytes are used by OEM build_tensors to populate input/output maps.
    uint8_t is_input;      // 0x18
    uint8_t is_output;     // 0x19
    uint8_t reserved0[2];  // 0x1a-0x1b padding

    // Additional metadata strings used by OEM code to derive dtype/format.
    // Exact semantics still under RE, but they are passed through to
    // utils::string2data_type / string2data_format.
    std::string layout;    // 0x1c: data layout / format string (e.g. "NHWC")
    std::string dtype_str; // 0x34: data type string (e.g. "FP32")

    // Shape vector (std::vector<int32_t>) at offset 0x4c.
    shape_t shape;         // 0x4c

    // Misc integer fields used for offsets/strides/etc in OEM code.
    int32_t stride;        // 0x58
    int32_t some_flag;     // 0x5c
    int32_t offset;        // 0x60
    int32_t field_64;      // 0x64
    int32_t field_68;      // 0x68

    uint8_t channel;       // 0x6c
    uint8_t reserved1[3];  // 0x6d-0x6f padding
};

static_assert(sizeof(TensorInfo) == 0x70,
              "TensorInfo size must match OEM (0x70 bytes)");

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

/* TensorX wrapper - layout must match OEM (sizeof == 0x20).
 * Fields and ordering are reconstructed from libmert smart diff:
 *   0x00: TensorX*          content/tensorx
 *   0x04: std::string       name
 *   0x1c: FlushCacheStatus  flush_status
 */
class TensorXWrapper {
public:
    TensorX *tensorx;              /* 0x00 - content pointer */
    std::string name;              /* 0x04 */
    FlushCacheStatus flush_status; /* 0x1c */

    TensorXWrapper();
    TensorXWrapper(TensorX *tx);
    TensorXWrapper(TensorX *tx, std::string n);
    TensorXWrapper(TensorX *tx, std::string n, FlushCacheStatus status);
    ~TensorXWrapper();

    void set_content(TensorX *tx);
    TensorX *get_content() const;

    void set_flush_cache_status(FlushCacheStatus status);
    FlushCacheStatus get_flush_cache_status() const;

    std::string get_name() const;
    int flush_cache();
};

static_assert(sizeof(TensorXWrapper) == 0x20,
              "TensorXWrapper size must match OEM (0x20 bytes)");

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

    /* Pyramid configuration - nested class
     * Size: 0x50 (80 bytes) based on OEM libmert.so
     * Layout:
     * - Offset 0x00-0x0b: basic fields (level, width, height)
     * - Offset 0x0c: std::vector<TensorXWrapper*> tensors_
     * - Offset 0x18-0x1f: reserved/padding
     * - Offset 0x20: std::map for input tensors
     * - Offset 0x38: std::map for output tensors
     */
    struct PyramidConfig {
        int level;
        int width;
        int height;
        std::vector<TensorXWrapper*> tensors_;  // At offset 0x0c

        /* Padding/reserved so that maps line up with OEM offsets */
        char reserved_[0x08];

        /* Name -> tensor wrapper maps (offsets 0x20 and 0x38) */
        std::map<std::string, TensorXWrapper*> input_tensors_;
        std::map<std::string, TensorXWrapper*> output_tensors_;

        TensorXWrapper* get_tensor_wrapper(std::string &name) const;
    };

    static_assert(sizeof(PyramidConfig) == 0x50,
                  "PyramidConfig size must match OEM (0x50 bytes)");

    MagikModelBase(long long param1, long long param2, void *&param3, void *param4,
                   ModelMemoryInfoManager::MemAllocMode mode, ModuleMode module_mode);
    virtual ~MagikModelBase();

    /* Member variables - matching OEM libmert.so layout
     * Based on reverse engineering:
     * - Offset 0x04: vtable pointer (automatic)
     * - Offset 0x10: inputs_ vector
     * - Offset 0x1c: outputs_ vector
     * - Offset 0x34: pyramid_configs_ vector
     */
    char padding_[0x0c];  // Padding from 0x04 to 0x10
    std::vector<TensorXWrapper*> inputs_;      // At offset 0x10
    std::vector<TensorXWrapper*> outputs_;     // At offset 0x1c
    char padding2_[0x0c];  // Padding from 0x28 to 0x34
    std::vector<PyramidConfig*> pyramid_configs_;  // At offset 0x34
    std::vector<MagikLayerBase*> layers_;
    PyramidConfig *main_pyramid_config_;
    virtual int run();
    virtual int reshape();
    virtual int pre_graph_run();
    virtual int free_forward_memory();
    virtual int free_inputs_memory();
    virtual int open_mnni_debug();
    virtual int open_mnni_profiler();
    virtual int set_main_pyramid_config(int level);
    virtual int create_and_add_pyramid_config();
    virtual int add_pyramid_config(PyramidConfig *config);
    virtual PyramidConfig* get_main_pyramid_config();

    virtual int build_tensors(PyramidConfig *config, std::vector<TensorInfo> infos);
    virtual TensorX* create_tensor(TensorInfo info);
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

