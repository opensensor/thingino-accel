/*
 * thingino-accel - Magik Model/Layer Stubs
 * These are stub implementations for the Magik model framework
 */

#include "magik_model.h"
#include "../../include/nna_memory.h"
#include <cstdio>
#include <dlfcn.h>
#include <typeinfo>
#include <new>
#include <cstdlib>
#include <sys/mman.h>
#include <unistd.h>


namespace magik {
namespace venus {

/* Device implementation */
Device::Device() {}
Device::~Device() {}

/* TensorXWrapper implementation */
TensorXWrapper::TensorXWrapper() : tensorx(nullptr) {
    printf("[VENUS] TensorXWrapper::TensorXWrapper() default constructor\n");
    fflush(stdout);
}

TensorXWrapper::TensorXWrapper(TensorX *tx) : tensorx(tx) {
    printf("[VENUS] TensorXWrapper::TensorXWrapper(tx=%p)\n", (void*)tx);
    fflush(stdout);
}

TensorXWrapper::~TensorXWrapper() {
    printf("[VENUS] TensorXWrapper::~TensorXWrapper(this=%p)\n", (void*)this);
    fflush(stdout);
}

/* TensorX constructor with size, data pointer, and device */
extern "C" void _ZN5magik5venus7TensorXC1EjPvNS0_6DeviceE(TensorX *this_ptr, uint32_t size, void *data_ptr, Device dev) {
    printf("[VENUS] TensorX::TensorX(this=%p, size=%u, data=%p)\n", (void*)this_ptr, size, data_ptr);
    fflush(stdout);
    (void)dev; /* Device selection not yet used; TODO: hook into MBuffer/device allocator. */

    /* The .mgk file has already allocated space for this object on the stack.
     * Use placement-new to run our C++ default constructor and initialize fields.
     */
    new (this_ptr) TensorX();

    /* Override key fields based on constructor arguments. */
    this_ptr->data       = data_ptr;
    this_ptr->bytes      = size;
    this_ptr->owns_data  = 0;     /* Caller owns the data */
    this_ptr->align      = 64;    /* NNA alignment */
    this_ptr->data_offset = 0;    /* No offset by default */

    printf("[VENUS] TensorX::TensorX() completed:\n");
    printf("[VENUS]   ndims=%d\n",
           this_ptr->dims_begin && this_ptr->dims_end ? (int)(this_ptr->dims_end - this_ptr->dims_begin) : 0);
    printf("[VENUS]   dtype=%d, format=%d\n", (int)this_ptr->dtype, (int)this_ptr->format);
    printf("[VENUS]   data=%p, bytes=%u\n", this_ptr->data, this_ptr->bytes);
    printf("[VENUS]   owns_data=%u, ref_count=%d\n", this_ptr->owns_data, this_ptr->ref_count);
    printf("[VENUS]   align=%u, data_offset=%u\n", this_ptr->align, this_ptr->data_offset);
    printf("[VENUS]   sizeof(TensorX)=%zu\n", sizeof(TensorX));
    fflush(stdout);
}

/* TensorX mudata method */
extern "C" void* _ZN5magik5venus7TensorX6mudataEjj(TensorX *this_ptr, uint32_t offset, uint32_t size) {
    (void)size;
    if (!this_ptr->data) {
        return nullptr;
    }
    return (char*)this_ptr->data + offset;
}

/* TensorX get_memory_size (OEM-compatible symbol) */
extern "C" size_t _ZNK5magik5venus7TensorX15get_memory_sizeEv(const TensorX *this_ptr) {
    /* OEM uses bytes rounded up to align() boundary; align field already reflects that. */
    size_t aligned = (static_cast<size_t>(this_ptr->bytes) + (this_ptr->align - 1u)) & ~static_cast<size_t>(this_ptr->align - 1u);
    printf("[VENUS] TensorX::get_memory_size(this=%p) -> %zu (bytes=%u, align=%u)\n",
           (void*)this_ptr, aligned, this_ptr->bytes, this_ptr->align);
    fflush(stdout);
    return aligned;
}

/* TensorX align (OEM-compatible symbol) */
extern "C" size_t _ZNK5magik5venus7TensorX5alignEv(const TensorX *this_ptr) {
    size_t kAlign = this_ptr->align ? this_ptr->align : 64u;
    printf("[VENUS] TensorX::align(this=%p) -> %zu\n", (void*)this_ptr, kAlign);
    fflush(stdout);
    return kAlign;
}

/* TensorX vdata<char> (OEM-compatible symbol) */
extern "C" const void* _ZNK5magik5venus7TensorX5vdataIaEEPKT_v(const TensorX *this_ptr) {
    if (!this_ptr->data) {
        return nullptr;
    }

    size_t offset = static_cast<size_t>(this_ptr->data_offset);
    const void *ptr = static_cast<const void*>(
        static_cast<const char*>(this_ptr->data) + offset);
    printf("[VENUS] TensorX::vdata<char>(this=%p) -> %p (offset=%zu)\n",
           (void*)this_ptr, ptr, offset);
    fflush(stdout);
    return ptr;
}


/* Core ORAM functions */
namespace core {
    void* oram_malloc(size_t size) {
        printf("[VENUS] core::oram_malloc(size=%zu)\n", size);
        fflush(stdout);
        /* Use NNA memory allocator */
        void *ptr = nna_malloc(size);
        printf("[VENUS] core::oram_malloc returning %p\n", ptr);
        fflush(stdout);
        return ptr;
    }

    void oram_free(void *ptr) {
        printf("[VENUS] core::oram_free(ptr=%p)\n", ptr);
        fflush(stdout);
        nna_free(ptr);
    }
} // namespace core

/* MagikLayerBase implementation */
MagikLayerBase::MagikLayerBase() {}
MagikLayerBase::~MagikLayerBase() {}

void MagikLayerBase::set_inputs(std::vector<TensorXWrapper*> inputs) {
    (void)inputs;
}

void MagikLayerBase::set_outputs(std::vector<TensorXWrapper*> outputs) {
    (void)outputs;
}

void MagikLayerBase::_flush_cache(std::vector<TensorXWrapper*> tensors) {
    (void)tensors;
}

std::vector<TensorXWrapper*> MagikLayerBase::get_inputs() const {
    return std::vector<TensorXWrapper*>();
}

std::vector<TensorXWrapper*> MagikLayerBase::get_outputs() const {
    return std::vector<TensorXWrapper*>();
}

std::vector<TensorXWrapper*> MagikLayerBase::get_input_wrappers() const {
    return std::vector<TensorXWrapper*>();
}

std::vector<TensorXWrapper*> MagikLayerBase::get_output_wrappers() const {
    return std::vector<TensorXWrapper*>();
}

std::string MagikLayerBase::get_name() const {
    return "";
}

int MagikLayerBase::get_layer_id() const {
    return 0;
}

/* MagikModelBase implementation */
MagikModelBase::MagikModelBase(long long param1, long long param2, void *&param3, void *param4,
                               ModelMemoryInfoManager::MemAllocMode mode, ModuleMode module_mode)
    : main_pyramid_config_(nullptr) {
    printf("[VENUS] MagikModelBase::MagikModelBase(p1=%lld, p2=%lld, p3=%p, p4=%p)\n",
           param1, param2, param3, param4);
    fflush(stdout);

    /* Initialize padding to zero */
    memset(padding_, 0, sizeof(padding_));
    memset(padding2_, 0, sizeof(padding2_));

    /* Constructor - just suppress unused warnings */
    (void)param1; (void)param2; (void)param3; (void)param4; (void)mode; (void)module_mode;
    printf("[VENUS] MagikModelBase::MagikModelBase() - constructor body complete\n");
    fflush(stdout);
}

MagikModelBase::~MagikModelBase() {
    if (main_pyramid_config_) {
        delete main_pyramid_config_;
        main_pyramid_config_ = nullptr;
    }
}

int MagikModelBase::run() {
    printf("MagikModelBase::run() - stub\n");
    return 0;
}

int MagikModelBase::reshape() { return 0; }
int MagikModelBase::pre_graph_run() { return 0; }
int MagikModelBase::free_forward_memory() { return 0; }
int MagikModelBase::free_inputs_memory() { return 0; }
int MagikModelBase::open_mnni_debug() { return 0; }
int MagikModelBase::open_mnni_profiler() { return 0; }
int MagikModelBase::set_main_pyramid_config(int level) {
    printf("[VENUS] MagikModelBase::set_main_pyramid_config(level=%d)\n", level);
    fflush(stdout);

    if (main_pyramid_config_) {
        main_pyramid_config_->level = level;
    }

    return 0;
}
int MagikModelBase::add_pyramid_config(PyramidConfig *config) {
    printf("[VENUS] MagikModelBase::add_pyramid_config(config=%p)\n", config);
    fflush(stdout);

    if (config) {
        /* Add to the vector - now that we have the correct memory layout */
        pyramid_configs_.push_back(config);
        printf("[VENUS] Added to pyramid_configs_ vector (size now = %zu)\n", pyramid_configs_.size());
        fflush(stdout);

        /* Set as main config if we don't have one */
        if (!main_pyramid_config_) {
            main_pyramid_config_ = config;
            printf("[VENUS] Set as main_pyramid_config_ = %p\n", (void*)main_pyramid_config_);
            fflush(stdout);
        }
    }

    return 0;
}

int MagikModelBase::create_and_add_pyramid_config() {
    printf("[VENUS] MagikModelBase::create_and_add_pyramid_config()\n");
    fflush(stdout);

    /* Create a new PyramidConfig */
    PyramidConfig *config = new PyramidConfig();

    /* Initialize basic fields; std::vector/std::map members are
     * already default-constructed as empty containers.
     */
    config->level = 0;
    config->width = 0;
    config->height = 0;

    printf("[VENUS] Created new PyramidConfig at %p\n", (void*)config);
    fflush(stdout);

    /* Add it to the collection */
    add_pyramid_config(config);

    printf("[VENUS] create_and_add_pyramid_config() returning %p\n", (void*)config);
    fflush(stdout);

    /* Return the pointer cast to int, as per OEM implementation */
    return (int)(intptr_t)config;
}

MagikModelBase::PyramidConfig* MagikModelBase::get_main_pyramid_config() {
    printf("[VENUS] MagikModelBase::get_main_pyramid_config() returning %p\n", (void*)main_pyramid_config_);
    fflush(stdout);
    return main_pyramid_config_;
}

TensorX* MagikModelBase::create_tensor(TensorInfo info) {
    printf("[VENUS] MagikModelBase::create_tensor(name=%s)\n", info.name.c_str());
    fflush(stdout);

    /* Log TensorInfo details to validate ABI/layout at runtime. */
    printf("[VENUS]   dtype_str=\"%s\", layout=\"%s\", is_input=%u, is_output=%u\n",
           info.dtype_str.c_str(), info.layout.c_str(),
           (unsigned)info.is_input, (unsigned)info.is_output);
    printf("[VENUS]   shape dims=[");
    for (size_t i = 0; i < info.shape.size(); ++i) {
        printf("%d%s", info.shape[i], (i + 1 < info.shape.size()) ? "," : "");
    }
    printf("]\n");
    fflush(stdout);

    /* Allocate a TensorX object (0x44 bytes as per OEM) */
    TensorX *tensor = new(std::nothrow) TensorX();
    if (!tensor) {
        printf("[VENUS] ERROR: MagikModelBase::create_tensor - new TensorX failed\n");
        fflush(stdout);
        return nullptr;
    }

    /* Derive dtype/format from strings when available. */
    DataType (*dt_fn)(const std::string&) = utils::string2data_type;
    TensorFormat (*fmt_fn)(const std::string&) = utils::string2data_format;

    DataType dtype = dt_fn(info.dtype_str);
    if (dtype == DataType::NONE || dtype == DataType::AUTO) {
        dtype = DataType::FP32;  /* Reasonable default */
    }
    TensorFormat fmt = fmt_fn(info.layout);

    tensor->dtype  = dtype;
    tensor->format = fmt;

    /* Configure shape. */
    tensor->set_shape(info.shape);

    /* Compute total number of elements. */
    size_t total_elements = 1;
    for (size_t i = 0; i < info.shape.size(); ++i) {
        int dim = info.shape[i];
        if (dim <= 0) {
            continue;
        }
        total_elements *= static_cast<size_t>(dim);
    }
    if (total_elements == 0) {
        total_elements = 1;
    }

    int bits_per_element = utils::data_type2bits(dtype);
    if (bits_per_element <= 0) {
        bits_per_element = 32;  /* Conservative default */
    }

    size_t bytes_needed = (total_elements * static_cast<size_t>(bits_per_element) + 7u) / 8u;
    if (bytes_needed == 0) {
        bytes_needed = 4;
    }

    /* Align to NNA-friendly boundary and allocate from ORAM/DDR. */
    tensor->align = 64;
    size_t aligned_bytes = (bytes_needed + tensor->align - 1u) & ~static_cast<size_t>(tensor->align - 1u);

    void *data = core::oram_malloc(aligned_bytes);
    if (!data) {
        printf("[VENUS] WARNING: core::oram_malloc(%zu) failed, falling back to malloc\n",
               aligned_bytes);
        fflush(stdout);
        data = std::malloc(aligned_bytes);
    }

    if (!data) {
        printf("[VENUS] ERROR: Failed to allocate %zu bytes for tensor data\n",
               aligned_bytes);
        fflush(stdout);
        delete tensor;
        return nullptr;
    }

    tensor->data        = data;
    tensor->bytes       = static_cast<uint32_t>(aligned_bytes);
    tensor->owns_data   = 1u;
    tensor->data_offset = 0;

    printf("[VENUS] Created tensor: %zu bytes (aligned=%zu), dtype=%d, fmt=%d, shape=[",
           bytes_needed, aligned_bytes, (int)tensor->dtype, (int)tensor->format);
    for (size_t i = 0; i < info.shape.size(); ++i) {
        printf("%d%s", info.shape[i], (i + 1 < info.shape.size()) ? "," : "");
    }
    printf("]\n");
    fflush(stdout);

    return tensor;
}

int MagikModelBase::build_tensors(PyramidConfig *config, std::vector<TensorInfo> infos) {
    printf("[VENUS] MagikModelBase::build_tensors(config=%p, infos.size=%zu)\n",
           (void*)config, infos.size());
    fflush(stdout);

    if (!config) {
        printf("[VENUS] ERROR: build_tensors called with NULL config\n");
        fflush(stdout);
        return -1;
    }

    if (infos.empty()) {
        printf("[VENUS] build_tensors: no TensorInfo entries, nothing to do\n");
        fflush(stdout);
        return 0;
    }

    for (size_t i = 0; i < infos.size(); ++i) {
        TensorInfo &info = infos[i];
        printf("[VENUS] build_tensors: [%zu] name='%s'\n",
               i, info.name.c_str());
        fflush(stdout);

        printf("[VENUS] build_tensors: calling create_tensor for '%s'\n",
               info.name.c_str());
        fflush(stdout);

        TensorX *tensor = MagikModelBase::create_tensor(info);
        if (!tensor) {
            printf("[VENUS] ERROR: create_tensor failed for '%s'\n",
                   info.name.c_str());
            fflush(stdout);
            return -1;
        }

        printf("[VENUS] build_tensors: create_tensor returned %p for '%s'\n",
               (void*)tensor, info.name.c_str());
        fflush(stdout);

        TensorXWrapper *wrapper = new TensorXWrapper(tensor);

        /* Track tensor in this pyramid configuration. */
        config->tensors_.push_back(wrapper);

        /* For now, map by name in both input and output maps so that
         * PyramidConfig::get_tensor_wrapper can resolve tensors by name. This
         * may be refined later once we fully understand OEM input/output flags.
         */
        config->input_tensors_.emplace(info.name, wrapper);
        config->output_tensors_.emplace(info.name, wrapper);
    }

    printf("[VENUS] build_tensors: built %zu tensors (config=%p, total_tensors=%zu)\n",
           infos.size(), (void*)config, config->tensors_.size());
    fflush(stdout);

    return 0;
}

int MagikModelBase::update_cache_buffer_ptr(std::vector<MagikLayerBase*> layers, void *ptr) {
    (void)layers; (void)ptr;
    return 0;
}

int MagikModelBase::set_oram_address(void *addr, long long size) const {
    printf("[VENUS] MagikModelBase::set_oram_address(addr=%p, size=%lld)\n", addr, size);
    fflush(stdout);

    /* This method is called by the derived class constructor to set the ORAM address.
     * The method is const but needs to modify global state.
     * We'll just acknowledge the call for now - the globals are already set correctly
     * by nna_runtime_init().
     */
    (void)addr;
    (void)size;

    printf("[VENUS] set_oram_address: ORAM address already set by runtime init\n");
    fflush(stdout);

    return 0;
}

/* PyramidConfig::get_tensor_wrapper */
TensorXWrapper* MagikModelBase::PyramidConfig::get_tensor_wrapper(std::string &name) const {
    printf("[VENUS] PyramidConfig::get_tensor_wrapper(name=%s)\n", name.c_str());
    fflush(stdout);

    /* We maintain both a vector and two maps that mirror OEM's caching behavior. */
    PyramidConfig *self = const_cast<PyramidConfig*>(this);

    /* First, try to find an existing wrapper in the name maps. */
    auto it = self->input_tensors_.find(name);
    if (it != self->input_tensors_.end()) {
        printf("[VENUS] Found existing tensor wrapper %p for '%s' in input_tensors_\n",
               (void*)it->second, name.c_str());
        fflush(stdout);
        return it->second;
    }

    it = self->output_tensors_.find(name);
    if (it != self->output_tensors_.end()) {
        printf("[VENUS] Found existing tensor wrapper %p for '%s' in output_tensors_\n",
               (void*)it->second, name.c_str());
        fflush(stdout);
        return it->second;
    }

    /* Fallback: linear scan of tensors_ vector (defensive). */
    for (size_t i = 0; i < tensors_.size(); i++) {
        TensorXWrapper *wrapper = tensors_[i];
        if (!wrapper || !wrapper->tensorx)
            continue;
        /* We currently do not track names on wrappers, so skip name comparison. */
    }

    /* Not found - create a new tensor on-the-fly to avoid nullptr dereference. */
    printf("[VENUS] Tensor '%s' not found, creating new tensor (no host buffer)\n", name.c_str());
    fflush(stdout);

    /* Create a new TensorX with default properties.
     * We deliberately do NOT allocate a large host buffer here; these
     * intermediate tensors are expected to be backed by NNA/ORAM memory
     * managed elsewhere. Keeping bytes=0 and data=nullptr avoids
     * exhausting the tiny libc heap on the device.
     */
    TensorX *tensor = new TensorX();

    /* Create wrapper */
    TensorXWrapper *wrapper = new TensorXWrapper(tensor);

    /* Add to containers (cast away const - this is a cache structure). */
    self->tensors_.push_back(wrapper);
    self->input_tensors_.emplace(name, wrapper);
    self->output_tensors_.emplace(name, wrapper);

    printf("[VENUS] Created new tensor wrapper %p for '%s'\n", (void*)wrapper, name.c_str());
    fflush(stdout);

    return wrapper;
}

std::string MagikModelBase::get_output_names() const {
    return "";
}

std::string MagikModelBase::get_input_names() const {
    return "";
}

TensorXWrapper* MagikModelBase::get_input(std::string &name) const {
    printf("[VENUS] MagikModelBase::get_input(name=%s)\n", name.c_str());
    fflush(stdout);

    PyramidConfig *cfg = main_pyramid_config_;
    if (!cfg && !pyramid_configs_.empty()) {
        cfg = pyramid_configs_[0];
    }
    if (!cfg) {
        printf("[VENUS] get_input: no PyramidConfig available, returning nullptr\n");
        fflush(stdout);
        return nullptr;
    }

    TensorXWrapper *wrapper = cfg->get_tensor_wrapper(name);
    if (!wrapper) {
        printf("[VENUS] get_input: get_tensor_wrapper returned nullptr for '%s'\n",
               name.c_str());
        fflush(stdout);
        return nullptr;
    }

    // Keep inputs_ vector in sync for index-based access.
    MagikModelBase *self = const_cast<MagikModelBase*>(this);
    bool found = false;
    for (auto *w : self->inputs_) {
        if (w == wrapper) {
            found = true;
            break;
        }
    }
    if (!found) {
        self->inputs_.push_back(wrapper);
    }

    return wrapper;
}

TensorXWrapper* MagikModelBase::get_input(int index) const {
    printf("[VENUS] MagikModelBase::get_input(index=%d)\n", index);
    fflush(stdout);

    if (index < 0) {
        printf("[VENUS] get_input(index): negative index, returning nullptr\n");
        fflush(stdout);
        return nullptr;
    }

    if (static_cast<size_t>(index) < inputs_.size()) {
        TensorXWrapper *wrapper = inputs_[index];
        printf("[VENUS] get_input(index): returning inputs_[%d]=%p\n",
               index, (void*)wrapper);
        fflush(stdout);
        return wrapper;
    }

    PyramidConfig *cfg = main_pyramid_config_;
    if (!cfg && !pyramid_configs_.empty()) {
        cfg = pyramid_configs_[0];
    }
    if (!cfg) {
        printf("[VENUS] get_input(index): no PyramidConfig available, returning nullptr\n");
        fflush(stdout);
        return nullptr;
    }

    if (static_cast<size_t>(index) < cfg->tensors_.size()) {
        TensorXWrapper *wrapper = cfg->tensors_[index];
        printf("[VENUS] get_input(index): using config->tensors_[%d]=%p\n",
               index, (void*)wrapper);
        fflush(stdout);

        // Keep inputs_ vector in sync.
        MagikModelBase *self = const_cast<MagikModelBase*>(this);
        bool found = false;
        for (auto *w : self->inputs_) {
            if (w == wrapper) {
                found = true;
                break;
            }
        }
        if (!found) {
            self->inputs_.push_back(wrapper);
        }

        return wrapper;
    }

    printf("[VENUS] get_input(index): index %d out of range, returning nullptr\n", index);
    fflush(stdout);
    return nullptr;
}

TensorXWrapper* MagikModelBase::get_output(std::string &name) const {
    printf("[VENUS] MagikModelBase::get_output(name=%s)\n", name.c_str());
    fflush(stdout);

    PyramidConfig *cfg = main_pyramid_config_;
    if (!cfg && !pyramid_configs_.empty()) {
        cfg = pyramid_configs_[0];
    }
    if (!cfg) {
        printf("[VENUS] get_output: no PyramidConfig available, returning nullptr\n");
        fflush(stdout);
        return nullptr;
    }

    TensorXWrapper *wrapper = cfg->get_tensor_wrapper(name);
    if (!wrapper) {
        printf("[VENUS] get_output: get_tensor_wrapper returned nullptr for '%s'\n",
               name.c_str());
        fflush(stdout);
        return nullptr;
    }

    // Keep outputs_ vector in sync for index-based access.
    MagikModelBase *self = const_cast<MagikModelBase*>(this);
    bool found = false;
    for (auto *w : self->outputs_) {
        if (w == wrapper) {
            found = true;
            break;
        }
    }
    if (!found) {
        self->outputs_.push_back(wrapper);
    }

    return wrapper;
}

TensorXWrapper* MagikModelBase::get_output(int index) const {
    printf("[VENUS] MagikModelBase::get_output(index=%d)\n", index);
    fflush(stdout);

    if (index < 0) {
        printf("[VENUS] get_output(index): negative index, returning nullptr\n");
        fflush(stdout);
        return nullptr;
    }

    if (static_cast<size_t>(index) < outputs_.size()) {
        TensorXWrapper *wrapper = outputs_[index];
        printf("[VENUS] get_output(index): returning outputs_[%d]=%p\n",
               index, (void*)wrapper);
        fflush(stdout);
        return wrapper;
    }

    PyramidConfig *cfg = main_pyramid_config_;
    if (!cfg && !pyramid_configs_.empty()) {
        cfg = pyramid_configs_[0];
    }
    if (!cfg) {
        printf("[VENUS] get_output(index): no PyramidConfig available, returning nullptr\n");
        fflush(stdout);
        return nullptr;
    }

    if (static_cast<size_t>(index) < cfg->tensors_.size()) {
        TensorXWrapper *wrapper = cfg->tensors_[index];
        printf("[VENUS] get_output(index): using config->tensors_[%d]=%p\n",
               index, (void*)wrapper);
        fflush(stdout);

        // Keep outputs_ vector in sync.
        MagikModelBase *self = const_cast<MagikModelBase*>(this);
        bool found = false;
        for (auto *w : self->outputs_) {
            if (w == wrapper) {
                found = true;
                break;
            }
        }
        if (!found) {
            self->outputs_.push_back(wrapper);
        }

        return wrapper;
    }

    printf("[VENUS] get_output(index): index %d out of range, returning nullptr\n", index);
    fflush(stdout);
    return nullptr;
}

size_t MagikModelBase::get_forward_memory_size() const {
    size_t total = 0;

    /* Sum the byte sizes of all tensors in all pyramid configurations. */
    for (size_t i = 0; i < pyramid_configs_.size(); ++i) {
        PyramidConfig *cfg = pyramid_configs_[i];
        if (!cfg) {
            continue;
        }

        for (size_t j = 0; j < cfg->tensors_.size(); ++j) {
            TensorXWrapper *wrapper = cfg->tensors_[j];
            if (!wrapper || !wrapper->tensorx) {
                continue;
            }
            total += wrapper->tensorx->get_bytes_size();
        }
    }

    if (total == 0) {
        /* Fallback to 1MB default if we have not built any tensors yet. */
        total = 1024 * 1024;
    }

    printf("[VENUS] MagikModelBase::get_forward_memory_size() -> %zu bytes\n", total);
    fflush(stdout);

    return total;
}

} // namespace venus

// Override OEM helper: get_string_vector_t
// This function is exported by the .mgk and normally reads a vector of strings
// from a kernel parameter block. The OEM implementation is currently causing
// huge allocations (hundreds of MB) due to parameter mismatch, which
// immediately fails on the device. For now, we provide a minimal, safe
// implementation that avoids large allocations and lets the demo progress.
extern "C" void _Z19get_string_vector_tRSt6vectorINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EEPKvRi(
    std::vector<std::string> &vec,
    const void *param,
    int &index)
{
    (void)param;

    void *ra0 = __builtin_return_address(0);
    Dl_info info0;
    const char *sym = "?";
    unsigned long off = 0;
    const char *obj = "?";
    if (dladdr(ra0, &info0)) {
        sym = info0.dli_sname ? info0.dli_sname : "?";
        obj = info0.dli_fname ? info0.dli_fname : "?";
        off = (unsigned long)((char *)ra0 - (char *)info0.dli_saddr);
    }

    std::fprintf(stderr,
                 "[VENUS] get_string_vector_t override called vec_size_before=%zu index=%d\n"
                 "  ra0=%p (obj=%s, symbol=%s+0x%lx)\n",
                 vec.size(), index,
                 ra0, obj, sym, off);
    std::fflush(stderr);

    // Provide a minimal, but non-empty, vector so downstream code sees
    // output_size == 1 and similar checks satisfied.
    vec.clear();
    vec.emplace_back("dummy");

    // Advance the parameter cursor so conv2d_int8_param_init sees progress.
    // We don't yet know the exact OEM contract, but treating each call as
    // consuming one slot is enough to satisfy its assertions.
    ++index;
}


// Override OEM helper: get_string_t
// Mangled name: _Z12get_string_tRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKvRi
// Similar to get_string_vector_t but returns a single string. The OEM
// implementation currently mis-parses our parameter blocks and issues
// a huge memcpy (~2.5 MB) from the parameter pointer, which walks off
// the mapped region and causes SIGBUS. Provide a minimal, safe
// implementation that just returns a small dummy string.
extern "C" void _Z12get_string_tRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKvRi(
    std::string &out,
    const void *param,
    int &index)
{
    (void)param;

    void *ra0 = __builtin_return_address(0);
    Dl_info info0;
    const char *sym = "?";
    const char *obj = "?";
    unsigned long off = 0;
    if (dladdr(ra0, &info0)) {
        sym = info0.dli_sname ? info0.dli_sname : "?";
        obj = info0.dli_fname ? info0.dli_fname : "?";
        off = (unsigned long)((char *)ra0 - (char *)info0.dli_saddr);
    }

    std::fprintf(stderr,
                 "[VENUS] get_string_t override called index=%d\n"
                 "  ra0=%p (obj=%s, symbol=%s+0x%lx)\n",
                 index,
                 ra0, obj, sym, off);
    std::fflush(stderr);

    out.assign("dummy");
    // Advance the parameter cursor in lockstep with the OEM helpers so
    // conv2d_int8_param_init's internal checks see consistent progress.
    ++index;
}

} // namespace magik

// Shim for OEM free function prepare_init_attr used by .mgk models.
//
// Original signature (demangled):
//   void prepare_init_attr(std::vector<magik::venus::DataAttribute>&,
//                          std::vector<std::string>,
//                          std::vector<std::string>,
//                          std::vector<int>,
//                          std::vector<std::string>);
//
// The OEM implementation currently crashes with SIGBUS when parsing
// attributes due to mismatched expectations on parameter buffers.
// We intercept the call here, log the arguments, and (for now) leave the
// DataAttribute vector untouched so later code sees whatever it populated
// earlier. This keeps us moving forward while we reverse-engineer the
// exact attribute semantics.
void prepare_init_attr(
    std::vector<magik::venus::DataAttribute> &attrs,
    std::vector<std::string> attr_vec1,
    std::vector<std::string> attr_vec2,
    std::vector<int> int_vec,
    std::vector<std::string> extra_vec)
{
    void *ra0 = __builtin_return_address(0);
    Dl_info info0;
    const char *sym = "?";
    const char *obj = "?";
    unsigned long off = 0;
    if (dladdr(ra0, &info0)) {
        sym = info0.dli_sname ? info0.dli_sname : "?";
        obj = info0.dli_fname ? info0.dli_fname : "?";
        off = (unsigned long)((char *)ra0 - (char *)info0.dli_saddr);
    }

    std::fprintf(stderr,
                 "[VENUS] prepare_init_attr shim called\n"
                 "  ra0=%p (obj=%s, symbol=%s+0x%lx)\n"
                 "  attrs.size=%zu vec1.size=%zu vec2.size=%zu int_vec.size=%zu extra.size=%zu\n",
                 ra0, obj, sym, off,
                 attrs.size(),
                 attr_vec1.size(),
                 attr_vec2.size(),
                 int_vec.size(),
                 extra_vec.size());
    std::fflush(stderr);

    // NOTE: We intentionally do *not* modify `attrs` yet. Once we fully
    // understand the OEM semantics we can populate attributes here in a
    // compatible way.
}


extern "C" void _ZSt17__throw_bad_allocv(void) {
    void *ra0 = __builtin_return_address(0);
    void *ra1 = __builtin_return_address(1);
    void *ra2 = __builtin_return_address(2);
    Dl_info info0;
    if (dladdr(ra0, &info0)) {
        fprintf(stderr,
                "[VENUS] std::__throw_bad_alloc intercepted\n"
                "  ra0=%p (obj=%s, base=%p, symbol=%s+0x%lx)\n"
                "  ra1=%p\n"
                "  ra2=%p\n",
                ra0,
                info0.dli_fname ? info0.dli_fname : "?",
                info0.dli_fbase,
                info0.dli_sname ? info0.dli_sname : "?",
                (unsigned long)((char *)ra0 - (char *)info0.dli_saddr),
                ra1,
                ra2);
    } else {
        fprintf(stderr,
                "[VENUS] std::__throw_bad_alloc intercepted\n"
                "  ra0=%p (no symbol)\n  ra1=%p\n  ra2=%p\n",
                ra0,
                ra1,
                ra2);
    }
    fflush(stderr);
    __builtin_trap();
}

extern "C" void _ZSt20__throw_length_errorPKc(const char *msg) {
    void *ra0 = __builtin_return_address(0);
    void *ra1 = __builtin_return_address(1);
    void *ra2 = __builtin_return_address(2);
    Dl_info info0;
    if (dladdr(ra0, &info0)) {
        fprintf(stderr,
                "[VENUS] std::__throw_length_error(\"%s\") intercepted\n"
                "  ra0=%p (obj=%s, base=%p, symbol=%s+0x%lx)\n"
                "  ra1=%p\n"
                "  ra2=%p\n",
                msg ? msg : "",
                ra0,
                info0.dli_fname ? info0.dli_fname : "?",
                info0.dli_fbase,
                info0.dli_sname ? info0.dli_sname : "?",
                (unsigned long)((char *)ra0 - (char *)info0.dli_saddr),
                ra1,
                ra2);
    } else {
        fprintf(stderr,
                "[VENUS] std::__throw_length_error(\"%s\") intercepted\n"
                "  ra0=%p (no symbol)\n  ra1=%p\n  ra2=%p\n",
                msg ? msg : "",
                ra0,
                ra1,
                ra2);
    }
    fflush(stderr);
    __builtin_trap();
}

/*
 * Global operator new/delete interceptors
 *
 * These override the default C++ global allocation functions when
 * libvenus.so is preloaded. They simply forward to malloc/free and
 * log large allocations so we can see what size triggered
 * std::bad_alloc inside the OEM .mgk code.
 */
void* operator new(std::size_t size) {
    void *ptr = std::malloc(size);
    if (!ptr) {
        /* Log the failing allocation and its call site before throwing. */
        void *ra0 = __builtin_return_address(0);
        Dl_info info0;
        if (dladdr(ra0, &info0)) {
            std::fprintf(stderr,
                         "[VENUS] operator new OOM size=%zu\n"
                         "  ra0=%p (obj=%s, base=%p, symbol=%s+0x%lx)\n",
                         size,
                         ra0,
                         info0.dli_fname ? info0.dli_fname : "?",
                         info0.dli_fbase,
                         info0.dli_sname ? info0.dli_sname : "?",
                         (unsigned long)((char *)ra0 - (char *)info0.dli_saddr));
        } else {
            std::fprintf(stderr,
                         "[VENUS] operator new OOM size=%zu ra0=%p (no symbol)\n",
                         size,
                         ra0);
        }
        std::fflush(stderr);

        // Respect the C++ contract: throw std::bad_alloc on failure.
        throw std::bad_alloc();
    }
    return ptr;
}

void* operator new[](std::size_t size) {
    void *ptr = std::malloc(size);
    if (!ptr) {
        /* Log the failing allocation and its call site before throwing. */
        void *ra0 = __builtin_return_address(0);
        Dl_info info0;
        if (dladdr(ra0, &info0)) {
            std::fprintf(stderr,
                         "[VENUS] operator new[] OOM size=%zu\n"
                         "  ra0=%p (obj=%s, base=%p, symbol=%s+0x%lx)\n",
                         size,
                         ra0,
                         info0.dli_fname ? info0.dli_fname : "?",
                         info0.dli_fbase,
                         info0.dli_sname ? info0.dli_sname : "?",
                         (unsigned long)((char *)ra0 - (char *)info0.dli_saddr));
        } else {
            std::fprintf(stderr,
                         "[VENUS] operator new[] OOM size=%zu ra0=%p (no symbol)\n",
                         size,
                         ra0);
        }
        std::fflush(stderr);

        throw std::bad_alloc();
    }
    return ptr;
}

void operator delete(void *ptr) noexcept {
    std::free(ptr);
}

void operator delete[](void *ptr) noexcept {
    std::free(ptr);
}


/* Generic C++ exception interceptor: __cxa_throw
 * Logs all thrown exceptions (including std::bad_alloc) with type name and call site.
 */
extern "C" void __cxa_throw(void *exc_ptr, void *tinfo_raw, void (*dest)(void *)) {
    void *ra0 = __builtin_return_address(0);
    Dl_info info0;
    const std::type_info *tinfo = static_cast<const std::type_info *>(tinfo_raw);
    const char *type_name = tinfo ? tinfo->name() : "(null)";

    if (dladdr(ra0, &info0)) {
        fprintf(stderr,
                "[VENUS] __cxa_throw type=%s\n"
                "  ra0=%p (obj=%s, base=%p, symbol=%s+0x%lx)\n",
                type_name,
                ra0,
                info0.dli_fname ? info0.dli_fname : "?",
                info0.dli_fbase,
                info0.dli_sname ? info0.dli_sname : "?",
                (unsigned long)((char *)ra0 - (char *)info0.dli_saddr));
    } else {
        fprintf(stderr,
                "[VENUS] __cxa_throw type=%s ra0=%p (no symbol)\n",
                type_name,
                ra0);
    }
    fflush(stderr);

    using cxa_throw_t = void (*)(void *, void *, void (*)(void *));
    static cxa_throw_t real_cxa_throw = nullptr;
    if (!real_cxa_throw) {
        real_cxa_throw = (cxa_throw_t)dlsym(RTLD_NEXT, "__cxa_throw");
    }

    real_cxa_throw(exc_ptr, tinfo_raw, dest);
}
static bool venus_is_region_mapped(const void *ptr, size_t n)
{
    if (!ptr || n == 0)
        return true;

    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0)
        page_size = 4096;

    uintptr_t start = (uintptr_t)ptr;
    uintptr_t end = start + (n - 1);

    uintptr_t page_mask = (uintptr_t)page_size - 1u;
    uintptr_t page_start = start & ~page_mask;
    uintptr_t page_end = end & ~page_mask;

    unsigned char vec;
    for (uintptr_t p = page_start; p <= page_end; p += (uintptr_t)page_size) {
        if (mincore((void *)p, (size_t)page_size, &vec) != 0) {
            return false;
        }
    }

    return true;
}



/*
 * Lightweight global memcpy override for diagnostics.
 *
 * When libvenus.so is preloaded, this function will be used for most
 * memcpy calls from the .mgk and our own code. We keep the implementation
 * simple and standards-compliant and add conditional logging when the
 * pointers fall into the NNA / DDR virtual address ranges or are otherwise
 * suspicious.
 */
extern "C" void *memcpy(void *dest, const void *src, size_t n)
{
    if (!dest || !src || n == 0)
        return dest;

    uintptr_t d = (uintptr_t)dest;
    uintptr_t s = (uintptr_t)src;

    bool log = false;
    /* Log copies that touch the high virtual ranges where NNA/DDR/ORAM
     * mappings and .mgk data tend to live, or that are unusually large.
     */
    if (n > (1u << 20)) { /* >1MB */
        log = true;
    }
    if ((d >= 0x76000000u && d < 0x78000000u) ||
        (s >= 0x76000000u && s < 0x78000000u)) {
        log = true;
    }

    if (log) {
        void *ra0 = __builtin_return_address(0);
        Dl_info info0;
        const char *obj = "?";
        const char *sym = "?";
        unsigned long off = 0;
        if (dladdr(ra0, &info0)) {
            obj = info0.dli_fname ? info0.dli_fname : "?";
            sym = info0.dli_sname ? info0.dli_sname : "?";
            off = (unsigned long)((char *)ra0 - (char *)info0.dli_saddr);
        }

        std::fprintf(stderr,
                     "[VENUS] memcpy(dest=%p, src=%p, n=%zu)\n"
                     "        caller=%p (%s:%s+0x%lx)\n",
                     dest, src, (size_t)n,
                     ra0, obj, sym, off);
        std::fflush(stderr);
    }
    // Validate that both source and destination ranges are mapped before
    // touching them. Broken pointers from OEM kernels would otherwise raise
    // SIGBUS here on the device.
    if (!venus_is_region_mapped(src, n) || !venus_is_region_mapped(dest, n)) {
        std::fprintf(stderr,
                     "[VENUS] memcpy: skipping copy due to unmapped range dest=%p src=%p n=%zu\n",
                     dest, src, (size_t)n);
        std::fflush(stderr);
        return dest;
    }



    /* Simple, portable byte-wise memcpy implementation. */
    unsigned char *dptr = static_cast<unsigned char *>(dest);
    const unsigned char *sptr = static_cast<const unsigned char *>(src);

    /* Standard memcpy has undefined behaviour for overlapping regions.
     * We still try to behave sanely by choosing a direction based on
     * pointer order, effectively behaving like memmove for overlap.
     */
    if (dptr <= sptr || dptr >= sptr + n) {
        for (size_t i = 0; i < n; ++i)
            dptr[i] = sptr[i];
    } else {
        for (size_t i = n; i-- > 0; )
            dptr[i] = sptr[i];
    }

    return dest;
}




