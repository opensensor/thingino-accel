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
#include <functional>
#include <memory>

namespace magik {
namespace venus {

/* Forward declarations */
class TensorXWrapper;
class MagikLayerBase;

/* Kernel function type used by MagikKernelLayer */
using KernelFunc = std::function<ReturnValue(
    std::unique_ptr<kernel::KernelParam>&, OpConfig*)>;

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
    virtual int forward();
    virtual int update_cache_buffer_ptr(void *ptr);

protected:
    std::vector<TensorXWrapper*> inputs_;
    std::vector<TensorXWrapper*> outputs_;
    int layer_id_ = 0;
    std::string name_;
};

/* MagikKernelLayer - concrete layer that holds a kernel function */
class MagikKernelLayer : public MagikLayerBase {
public:
    using KernelFunc = std::function<ReturnValue(
        std::unique_ptr<kernel::KernelParam>&, OpConfig*)>;

    MagikKernelLayer(int op_hint,
                     KernelFunc kernel_fn,
                     std::unique_ptr<kernel::KernelParam>& param,
                     OpConfig* op_cfg);

    MagikKernelLayer(int op_hint,
                     KernelFunc kernel_fn,
                     std::unique_ptr<kernel::KernelParam>& param,
                     OpConfig* op_cfg,
                     std::vector<TensorXWrapper*> inputs,
                     std::vector<TensorXWrapper*> outputs);

    virtual ~MagikKernelLayer();

    virtual void set_inputs(std::vector<TensorXWrapper*> inputs) override;
    virtual void set_outputs(std::vector<TensorXWrapper*> outputs) override;
    virtual std::vector<TensorXWrapper*> get_inputs() const override;
    virtual std::vector<TensorXWrapper*> get_outputs() const override;
    virtual std::vector<TensorXWrapper*> get_input_wrappers() const override;
    virtual std::vector<TensorXWrapper*> get_output_wrappers() const override;
    virtual std::string get_name() const override;
    virtual int get_layer_id() const override;
    virtual int forward() override;
    virtual int update_cache_buffer_ptr(void *ptr) override;

    kernel::KernelParam* get_kernel_param();

private:
    int op_hint_;
    KernelFunc kernel_fn_;
    std::unique_ptr<kernel::KernelParam> param_;
    OpConfig* op_cfg_;
    std::vector<TensorXWrapper*> inputs_;
    std::vector<TensorXWrapper*> outputs_;
    std::string name_;
    static int next_layer_id_;
    int layer_id_;
};

    /* MagikModelBase */
    class MagikModelBase {
    public:
        /* Module mode - nested enum */
        enum class ModuleMode {
            NORMAL = 0,
            DEBUG = 1,
        };

        /* Pyramid configuration - nested struct
         *
         * OEM layout recovered from libmert.so:
         *   - magik::venus::MagikModelBase::create_and_add_pyramid_config()
         *       calls operator new(0x50), memset(first 0x18 bytes to 0), then
         *       default-constructs two std::_Rb_tree_header instances at
         *       offsets 0x18 and 0x30.
         *   - magik::venus::MagikModelBase::build_tensors(this, PyramidConfig*)
         *       treats (config + 0x0c) as std::vector<TensorXWrapper*> and
         *       pushes all created wrappers into it, and uses maps at
         *       (config + 0x20) and (config + 0x38).
         *   - magik::venus::MagikModelBase::PyramidConfig::get_tensor_wrapper()
         *       linearly scans the std::vector<TensorXWrapper*> at offset 0x0c.
         *
         * Resulting layout (sizeof(PyramidConfig) == 0x50):
         *   - 0x00: std::vector<MagikLayerBase*> layers_;
         *   - 0x0c: std::vector<TensorXWrapper*> tensors_;
         *   - 0x18: std::map<std::string, TensorXWrapper*> input_tensors_;
         *   - 0x30: std::map<std::string, TensorXWrapper*> output_tensors_;
         *   - 0x48: 8 bytes of reserved/padding.
         */
        struct PyramidConfig {
            std::vector<MagikLayerBase*> layers_;      // 0x00 - populated by .mgk's build_layer()
            std::vector<TensorXWrapper*> tensors_;     // 0x0c - populated by build_tensors()

            /* Name -> tensor wrapper maps (used by OEM for IO lookup) */
            std::map<std::string, TensorXWrapper*> input_tensors_;   // 0x18
            std::map<std::string, TensorXWrapper*> output_tensors_;  // 0x30

            /* Reserved padding to reach total size 0x50. Not used by OEM. */
            char reserved_[0x08];                                      // 0x48

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

    /* Pointer to layer I/O map. The .mgk might access this at a specific offset.
     * We'll allocate the actual map and point to it. */
    std::map<int, std::pair<std::vector<std::string>, std::vector<std::string>>>* layer_io_map_ptr_;

    /* Store tensors separately, keyed by config pointer.
     * This is immune to .mgk corruption of PyramidConfig internals. */
    std::map<PyramidConfig*, std::vector<TensorXWrapper*>> config_tensors_;

    /* Tensor lookup by name, keyed by config pointer. Used by get_tensor_wrapper. */
    std::map<PyramidConfig*, std::map<std::string, TensorXWrapper*>> config_tensor_names_;

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

/*
 * uranus namespace declarations - for compatibility with newer MGK models
 * These must be separate classes/types (not using declarations) to get
 * proper name mangling that matches what .mgk models expect.
 */
namespace magik {
namespace uranus {

/* Forward declarations */
class TensorXWrapper;
class MagikLayerBase;
class MagikModelBase;
struct TensorX;

/* DataType enum for uranus namespace */
enum class DataType : int {
    UNKNOWN = 0,
    FLOAT32 = 1,
    FLOAT16 = 2,
    INT32 = 3,
    UINT32 = 4,
    INT16 = 5,
    UINT16 = 6,
    INT8 = 7,
    UINT8 = 8,
    BOOL = 9,
};

/* DataFormat/TensorFormat for uranus namespace */
enum class DataFormat : int {
    UNKNOWN = 0,
    NCHW = 1,
    NHWC = 2,
    NC4HW4 = 3,
};

using TensorFormat = DataFormat;

/* Device enum */
enum class Device : int {
    CPU = 0,
    NNA = 1,
};

/* DataLocation enum */
enum class DataLocation : int {
    DDR = 0,
    ORAM = 1,
};

/* ModelMemoryInfoManager - duplicate definition for proper mangling */
class ModelMemoryInfoManager {
public:
    enum class MemAllocMode : int {
        DDR_ONLY = 0,
        ORAM_FIRST = 1,
        ORAM_ONLY = 2,
    };
};

/* TensorX structure for uranus namespace */
struct TensorX {
    int32_t *dims_begin;
    int32_t *dims_end;
    void    *reserved0;
    void    *data;
    void    *reserved1;
    int32_t  ref_count;
    void    *dims_meta0;
    void    *dims_meta1;
    void    *dims_meta2;
    uint32_t align;
    DataType dtype;
    TensorFormat format;
    uint32_t bytes;
    uint32_t owns_data;
    int32_t  reserved3;
    uint32_t data_offset;
    int32_t  reserved4;

    TensorX();
    ~TensorX();

    int step(int dim) const;
    size_t get_bytes_size() const;
    void* pdata() const;
    int malloc_mbo(unsigned int size, bool use_oram);
    void free_mbo();
};

/* TensorXWrapper for uranus namespace */
class TensorXWrapper {
public:
    TensorX *tensorx;
    std::string name;
    int flush_status;

    TensorXWrapper();
    TensorXWrapper(TensorX *tx);
    ~TensorXWrapper();

    TensorX* get_content() const;
    void set_content(TensorX *tx);
    std::string get_name() const;
};

/* TensorInfo for uranus namespace */
struct TensorInfo {
    std::string name;
    uint8_t is_input;
    uint8_t is_output;
    uint8_t reserved0[2];
    std::string layout;
    std::string dtype_str;
    std::vector<int32_t> shape;
    int32_t stride;
    int32_t some_flag;
    int32_t offset;
    int32_t field_64;
    int32_t field_68;
    int32_t field_6c;
};

/* MagikLayerBase for uranus namespace */
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
    virtual std::string get_type() const;
    virtual int get_layer_id() const;
    virtual int forward();

protected:
    std::vector<TensorXWrapper*> inputs_;
    std::vector<TensorXWrapper*> outputs_;
    int layer_id_;
    std::string name_;
    std::string type_;
};

/* MagikModelBase in uranus namespace */
class MagikModelBase {
public:
    enum class ModuleMode : int {
        NORMAL = 0,
        PYRAMID = 1,
    };

    struct PyramidConfig {
        std::vector<MagikLayerBase*> layers_;
        std::vector<TensorXWrapper*> tensors_;
        std::map<std::string, TensorXWrapper*> input_tensors_;
        std::map<std::string, TensorXWrapper*> output_tensors_;
        char reserved_[0x08];

        TensorXWrapper* get_tensor_wrapper(std::string &name) const;
    };

    MagikModelBase(long long forward_mem_size, long long param2,
                   void *&ddr_ptr, void *oram_ptr,
                   ModelMemoryInfoManager::MemAllocMode alloc_mode,
                   ModuleMode module_mode);

    virtual ~MagikModelBase();

    /* Core methods */
    virtual void run();
    virtual void reshape();
    virtual void alloc_forward_memory();
    virtual void free_forward_memory();
    virtual void free_inputs_memory();

    /* Pyramid config methods */
    virtual PyramidConfig* create_and_add_pyramid_config();
    virtual void set_main_pyramid_config(int level);
    virtual int build_tensors(PyramidConfig *config, std::vector<TensorInfo> infos);

    /* Tensor access methods */
    virtual TensorXWrapper* get_input(std::string &name) const;
    virtual TensorXWrapper* get_input(int index) const;
    virtual TensorXWrapper* get_output(std::string &name) const;
    virtual TensorXWrapper* get_output(int index) const;
    virtual std::string get_input_names() const;
    virtual std::string get_output_names() const;

    /* Memory methods */
    virtual size_t get_forward_memory_size() const;
    virtual void* get_oram_address() const;
    virtual void set_oram_address(void *addr, long long size) const;
    virtual void update_cache_buffer_ptr(std::vector<MagikLayerBase*> layers, void *ptr);
    virtual void update_ddr_root_ptr(std::vector<MagikLayerBase*> layers, void *ptr);

protected:
    venus::MagikModelBase *venus_impl_;
    std::vector<TensorXWrapper*> inputs_;
    std::vector<TensorXWrapper*> outputs_;
    std::vector<PyramidConfig*> pyramid_configs_;
    PyramidConfig *main_pyramid_config_;
    void *oram_addr_;
    long long oram_size_;
};

/* Helper function to set TensorX properties */
void magik_set_tensorX(TensorX &tensor, std::string name, std::string dtype_str,
                       std::vector<int> shape, void *data, Device device);

/* AIE (Activation/Math) namespace */
namespace aie {
    float banker_round(float val);

    /* Activation function parameter tables - these are global arrays */
    extern float param_sigmoid[256];
    extern float param_tanh[256];
    extern float param_swish[256];
} // namespace aie

/* Utils namespace for uranus - provides type conversion functions */
namespace utils {
    std::string data_type2string(DataType dtype);
    int data_type2bits(DataType dtype);
    int data_type2validbits(DataType dtype);
    DataType string2data_type(const std::string &str);
    DataType string2data_type(std::string str);
    std::string data_format2string(DataFormat fmt);
    DataFormat string2data_format(const std::string &str);
    DataFormat string2data_format(std::string str);
    int string2channel_layout(const std::string &str);
    int string2channel_layout(std::string str);
    Device string2device(std::string str);

    template<typename T>
    DataType type2data_type();
} // namespace utils

} // namespace uranus
} // namespace magik

#endif /* MAGIK_MODEL_H */

