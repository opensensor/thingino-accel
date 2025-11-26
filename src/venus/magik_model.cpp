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

    /* Allocate a TensorX object (0x44 bytes as per OEM) */
    TensorX *tensor = new TensorX();

    /* Set basic properties from TensorInfo */
    tensor->dtype  = info.dtype;
    tensor->format = info.format;
    tensor->set_shape(info.shape);

    /* Calculate size based on shape and dtype */
    size_t total_elements = 1;
    for (size_t i = 0; i < info.shape.size(); i++) {
        total_elements *= static_cast<size_t>(info.shape[i]);
    }

    /* Get bytes per element based on dtype */
    int bits_per_element = 8; /* Default to INT8 */
    switch (info.dtype) {
        case DataType::FP32:   bits_per_element = 32; break;
        case DataType::INT32:  bits_per_element = 32; break;
        case DataType::UINT32: bits_per_element = 32; break;
        case DataType::INT16:  bits_per_element = 16; break;
        case DataType::UINT16: bits_per_element = 16; break;
        case DataType::INT8:   bits_per_element = 8;  break;
        case DataType::UINT8:  bits_per_element = 8;  break;
        case DataType::UINT4B: bits_per_element = 4;  break;
        case DataType::UINT2B: bits_per_element = 2;  break;
        case DataType::BOOL:   bits_per_element = 1;  break;
        default:               bits_per_element = 8;  break;
    }

    size_t bytes_needed = (total_elements * bits_per_element + 7u) / 8u;
    tensor->bytes = static_cast<uint32_t>(bytes_needed);

    /* Allocate data buffer - use ORAM or DDR depending on size */
    /* For now, use regular malloc - TODO: use proper NNA memory allocation */
    tensor->data = malloc(bytes_needed);
    if (!tensor->data) {
        printf("[VENUS] ERROR: Failed to allocate %zu bytes for tensor data\n", bytes_needed);
        fflush(stdout);
        delete tensor;
        return nullptr;
    }

    tensor->owns_data = 1u;

    printf("[VENUS] Created tensor: %zu bytes, shape=[", bytes_needed);
    for (size_t i = 0; i < info.shape.size(); i++) {
        printf("%d%s", info.shape[i], i < info.shape.size()-1 ? "," : "");
    }
    printf("]\n");
    fflush(stdout);

    return tensor;
}

int MagikModelBase::build_tensors(PyramidConfig *config, std::vector<TensorInfo> infos) {
    printf("[VENUS] MagikModelBase::build_tensors(config=%p)\n", config);
    fflush(stdout);

    if (!config) {
        printf("[VENUS] ERROR: build_tensors called with NULL config\n");
        fflush(stdout);
        return -1;
    }

    /* The infos vector is passed by value, which on MIPS might be passed as a pointer
     * to a temporary. We need to be careful accessing it. For now, just return success
     * since the .mgk file's build_tensors will handle the actual tensor creation.
     */
    printf("[VENUS] build_tensors: Returning success (tensors will be built by derived class)\n");
    fflush(stdout);

    (void)infos; /* Suppress unused parameter warning */

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
    printf("[VENUS] MagikModelBase::get_input(name=%s) - returning nullptr\n", name.c_str());
    fflush(stdout);
    (void)name;
    return nullptr;
}

TensorXWrapper* MagikModelBase::get_input(int index) const {
    printf("[VENUS] MagikModelBase::get_input(index=%d) - returning nullptr\n", index);
    fflush(stdout);
    (void)index;
    return nullptr;
}

TensorXWrapper* MagikModelBase::get_output(std::string &name) const {
    printf("[VENUS] MagikModelBase::get_output(name=%s) - returning nullptr\n", name.c_str());
    fflush(stdout);
    (void)name;
    return nullptr;
}

TensorXWrapper* MagikModelBase::get_output(int index) const {
    printf("[VENUS] MagikModelBase::get_output(index=%d) - returning nullptr\n", index);
    fflush(stdout);
    (void)index;
    return nullptr;
}

size_t MagikModelBase::get_forward_memory_size() const {
    return 1024 * 1024;  // 1MB default
}

} // namespace venus
} // namespace magik

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
        // Respect the C++ contract: throw std::bad_alloc on failure.
        throw std::bad_alloc();
    }
    if (size >= 256 * 1024) {
        std::fprintf(stderr, "[VENUS] operator new(%zu) -> %p\n", size, ptr);
        std::fflush(stderr);
    }
    return ptr;
}

void* operator new[](std::size_t size) {
    void *ptr = std::malloc(size);
    if (!ptr) {
        throw std::bad_alloc();
    }
    if (size >= 256 * 1024) {
        std::fprintf(stderr, "[VENUS] operator new[](%zu) -> %p\n", size, ptr);
        std::fflush(stderr);
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



