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
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
#include <functional>
#include <memory>

/* Runtime DDR base exported from runtime.c so we can annotate parameter pointers. */
extern "C" void *__ddr_vbase;

namespace magik {
namespace venus {

/* Device implementation */
Device::Device() {}
Device::~Device() {}

/* TensorXWrapper implementation - layout aligned with OEM libmert. */
TensorXWrapper::TensorXWrapper()
    : tensorx(nullptr), name(), flush_status(FlushCacheStatus::DISABLED) {
    printf("[VENUS] TensorXWrapper::TensorXWrapper() default constructor (this=%p)\n",
           (void*)this);
    fflush(stdout);
}

TensorXWrapper::TensorXWrapper(TensorX *tx)
    : tensorx(tx), name(), flush_status(FlushCacheStatus::DISABLED) {
    printf("[VENUS] TensorXWrapper::TensorXWrapper(tx=%p, this=%p)\n",
           (void*)tx, (void*)this);
    fflush(stdout);
}

TensorXWrapper::TensorXWrapper(TensorX *tx, std::string n)
    : tensorx(tx), name(std::move(n)), flush_status(FlushCacheStatus::DISABLED) {
    printf("[VENUS] TensorXWrapper::TensorXWrapper(tx=%p, name=%s, this=%p)\n",
           (void*)tx, name.c_str(), (void*)this);
    fflush(stdout);
}

TensorXWrapper::TensorXWrapper(TensorX *tx, std::string n, FlushCacheStatus status)
    : tensorx(tx), name(std::move(n)), flush_status(status) {
    printf("[VENUS] TensorXWrapper::TensorXWrapper(tx=%p, name=%s, status=%d, this=%p)\n",
           (void*)tx, name.c_str(), (int)flush_status, (void*)this);
    fflush(stdout);
}

TensorXWrapper::~TensorXWrapper() {
    printf("[VENUS] TensorXWrapper::~TensorXWrapper(this=%p, tensorx=%p, status=%d)\n",
           (void*)this, (void*)tensorx, (int)flush_status);
    fflush(stdout);
}

void TensorXWrapper::set_content(TensorX *tx) {
    printf("[VENUS] TensorXWrapper::set_content(this=%p, tx=%p)\n",
           (void*)this, (void*)tx);
    tensorx = tx;
}

TensorX *TensorXWrapper::get_content() const {
    printf("[VENUS] TensorXWrapper::get_content(this=%p) -> %p\n",
           (void*)this, (void*)tensorx);
    return tensorx;
}

void TensorXWrapper::set_flush_cache_status(FlushCacheStatus status) {
    printf("[VENUS] TensorXWrapper::set_flush_cache_status(this=%p, status=%d)\n",
           (void*)this, (int)status);
    flush_status = status;
}

FlushCacheStatus TensorXWrapper::get_flush_cache_status() const {
    return flush_status;
}

std::string TensorXWrapper::get_name() const {
    return name;
}

int TensorXWrapper::flush_cache() {
    /* OEM implementation checks flush_status and, if enabled, flushes the
     * underlying TensorX data cache via __aie_flushcache_dir.
     * For now we only log and pretend success when enabled.
     */
    if (flush_status != FlushCacheStatus::ENABLE_ONCE &&
        flush_status != FlushCacheStatus::ENABLE_ALWAYS) {
        return 1;
    }

    printf("[VENUS] TensorXWrapper::flush_cache(this=%p, tensorx=%p, status=%d)\n",
           (void*)this, (void*)tensorx, (int)flush_status);
    fflush(stdout);

    // TODO: call into NNA cache flush when we have a confirmed implementation.
    return 1;
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

	/* TensorX reset_data stub (OEM-compatible symbol)
	 * Signature: void TensorX::reset_data(void *buf, void *data, int bytes)
	 * We treat `data` as a direct pointer to the underlying buffer and avoid
	 * any std::shared_ptr<MBuffer> / control-block gymnastics to prevent
	 * layout-dependent corruption.
	 */
	extern "C" void _ZN5magik5venus7TensorX10reset_dataEPvS2_i(
	    TensorX *this_ptr, void *buf, void *data_ptr, int bytes) {
	    (void)buf; /* Unused for now; OEM may use this for host-side aliasing. */

	    long raw_ndims = 0;
	    if (this_ptr->dims_begin && this_ptr->dims_end) {
	        raw_ndims = (long)(this_ptr->dims_end - this_ptr->dims_begin);
	    }

	    printf("[VENUS] TensorX::reset_data(this=%p, buf=%p, data=%p, bytes=%d, raw_ndims=%ld)\n",
	           (void*)this_ptr, buf, data_ptr, bytes, raw_ndims);
	    fflush(stdout);

	    /* Do not touch dims_*; they are owned by set_shape / create_tensor. */
	    this_ptr->data = data_ptr;
	    if (bytes > 0) {
	        this_ptr->bytes = (uint32_t)bytes;
	    }
	    /* We do not give ownership of this buffer to TensorX. */
	    this_ptr->owns_data = 0;
	    if (this_ptr->align == 0) {
	        this_ptr->align = 64;
	    }
	    /* Keep data_offset as-is; OEM sometimes uses non-zero offsets. */
	}

	/* TensorX reset_buffer stub (OEM-compatible symbol)
	 * Signature: void TensorX::reset_buffer(std::shared_ptr<MBuffer>)
	 * The mangled type encodes std::shared_ptr<MBuffer> by value. On 32-bit
	 * this is two pointer-sized words (ptr + control-block). We accept them
	 * as two opaque void* slots to avoid instantiating std::shared_ptr here,
	 * which would otherwise run its destructor and interact with an OEM
	 * control-block layout we don't fully emulate yet.
	 */
	extern "C" void _ZN5magik5venus7TensorX12reset_bufferESt10shared_ptrINS0_7MBufferEE(
	    TensorX *this_ptr, void *sptr_slot0, void *sptr_slot1) {
	    printf("[VENUS] TensorX::reset_buffer(this=%p, sptr0=%p, sptr1=%p) [stub/no-op]\n",
	           (void*)this_ptr, sptr_slot0, sptr_slot1);
	    fflush(stdout);
	    /* Intentionally do nothing: TensorX::data / bytes are managed by
	     * create_tensor / reset_data. MBuffer/shared_ptr lifetime stays in
	     * OEM code and is treated as opaque here. */
	}

	/* TensorX free_mbo stub (OEM-compatible symbol)
	 * Signature: void TensorX::free_mbo()
	 * OEM calls into MBuffer::free_data() here. For now we turn this into a
	 * no-op to avoid double-free of ORAM/DDR buffers that we manage via
	 * core::oram_malloc / core::oram_free. This may leak small OEM-managed
	 * host buffers, but that's acceptable for short-lived inference runs.
	 */
	extern "C" void _ZN5magik5venus7TensorX8free_mboEv(TensorX *this_ptr) {
	    printf("[VENUS] TensorX::free_mbo(this=%p, data=%p, bytes=%u) [stub/no-op]\n",
	           (void*)this_ptr, this_ptr ? this_ptr->data : nullptr,
	           this_ptr ? this_ptr->bytes : 0u);
	    fflush(stdout);
	    /* Do not modify data/bytes/owns_data here. Lifetime is handled by
	     * MagikModelBase::create_tensor and Tensor wrappers. */
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

    /* First pass: dump all TensorInfo metadata so we can debug is_input/is_output
     * flags before any tensor creation. This helps identify mismatch between
     * how many inputs the metadata declares vs. how many the OEM model thinks
     * exist.
     */
    printf("[VENUS] build_tensors: === TensorInfo metadata dump ===\n");
    fflush(stdout);
    for (size_t i = 0; i < infos.size(); ++i) {
        TensorInfo &ti = infos[i];
        printf("[VENUS] TensorInfo[%zu]: name='%s' is_input=%u is_output=%u\n",
               i, ti.name.c_str(), (unsigned)ti.is_input, (unsigned)ti.is_output);
        printf("        dtype_str='%s' layout='%s' channel=%u\n",
               ti.dtype_str.c_str(), ti.layout.c_str(), (unsigned)ti.channel);
        printf("        shape=[");
        for (size_t j = 0; j < ti.shape.size(); ++j) {
            printf("%d%s", ti.shape[j], (j + 1 < ti.shape.size()) ? "," : "");
        }
        printf("]\n");
        fflush(stdout);
    }
    printf("[VENUS] build_tensors: === End TensorInfo dump ===\n");
    fflush(stdout);

    for (size_t i = 0; i < infos.size(); ++i) {
        TensorInfo &info = infos[i];
        printf("[VENUS] build_tensors: [%zu] name='%s' is_input=%u is_output=%u\n",
               i, info.name.c_str(), (unsigned)info.is_input, (unsigned)info.is_output);
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

        /* Populate global input/output vectors based on TensorInfo flags so that
         * host code can query IO tensors by index without relying on any
         * DerivedMagikModel overrides whose layout we don't fully control.
         */
        if (info.is_input) {
            printf("[VENUS] build_tensors: marking '%s' as model INPUT (index=%zu)\n",
                   info.name.c_str(), inputs_.size());
            fflush(stdout);
            inputs_.push_back(wrapper);
            config->input_tensors_.emplace(info.name, wrapper);
        }

        if (info.is_output) {
            printf("[VENUS] build_tensors: marking '%s' as model OUTPUT (index=%zu)\n",
                   info.name.c_str(), outputs_.size());
            fflush(stdout);
            outputs_.push_back(wrapper);
            config->output_tensors_.emplace(info.name, wrapper);
        }
    }

    printf("[VENUS] build_tensors: built %zu tensors (config=%p, total_tensors=%zu)\n",
           infos.size(), (void*)config, config->tensors_.size());
    printf("[VENUS] build_tensors: SUMMARY - inputs_.size()=%zu, outputs_.size()=%zu\n",
           inputs_.size(), outputs_.size());
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

// OEM helper prototype (mangled name) so we can call get_string_t from
// our get_string_vector_t shim.
extern "C" int _Z12get_string_tRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKvRi(
    std::string &out,
    const void *param,
    int &index);

// Override OEM helper: get_string_vector_t
// This function is exported by the .mgk and normally reads a vector of strings
// from a kernel parameter block.
//
// The OEM implementation (from AEC_T41_16K_NS_OUT_UC.mgk) is roughly:
//   - read a 32-bit count from the param blob at param + index
//   - resize the std::vector<std::string> to that count
//   - for each entry call get_string_t() to fill one string
//   - return the updated index cursor
//
// We replicate those semantics here, but add a defensive clamp for obviously
// bogus counts (e.g. when the param pointer is wrong and we would otherwise
// try to allocate millions of strings).
extern "C" int _Z19get_string_vector_tRSt6vectorINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EEPKvRi(
    std::vector<std::string> &vec,
    const void *param,
    int &index)
{
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

    unsigned char bytes[16];
    for (int i = 0; i < 16; ++i) {
        bytes[i] = 0;
    }
    const unsigned char *base = static_cast<const unsigned char *>(param);
    if (base) {
        uintptr_t addr = (uintptr_t)base;
        if (addr >= 0x10000u) {
            for (int i = 0; i < 16; ++i) {
                bytes[i] = base[i];
            }
        }
    }

    // Read element count (32-bit) from the parameter block at the current index.
    int32_t count = 0;
    if (base) {
        uintptr_t addr = (uintptr_t)(base + index);
        if (addr >= 0x10000u) {
            memcpy(&count, base + index, sizeof(count));
            index += (int)sizeof(count);
        }
    }

    // Reasonable upper bound for string vector sizes coming from real .mgk
    // models. Anything far beyond this is almost certainly a corrupted pointer
    // or mis-parsed blob.
    static const int32_t kMaxReasonableCount = 1024;

    // Also look at which object the param pointer lives in; if it points into
    // libvenus.so itself, we know we are *not* looking at a real model param
    // blob.
    Dl_info param_info{};
    const char *param_obj = "?";
    bool param_in_venus = false;
    if (param && dladdr(param, &param_info)) {
        param_obj = param_info.dli_fname ? param_info.dli_fname : "?";
        if (std::strstr(param_obj, "libvenus.so")) {
            param_in_venus = true;
        }
    }

    // Additionally, detect whether the param pointer is inside the logical
    // DDR heap (__ddr_vbase .. __ddr_vbase + 8MB) that we expose to .mgk.
    bool param_in_ddr = false;
    long ddr_offset = -1;
    if (__ddr_vbase && param) {
        uintptr_t ddr_base = (uintptr_t)__ddr_vbase;
        uintptr_t p = (uintptr_t)param;
        static const uintptr_t kDdrSpan = 8u * 1024u * 1024u;
        if (p >= ddr_base && p < ddr_base + kDdrSpan) {
            param_in_ddr = true;
            ddr_offset = (long)(p - ddr_base);
        }
    }

    if (param_in_ddr) {
        // If it clearly lives in DDR, don't also label it as "libvenus".
        param_in_venus = false;
    } else {
        std::fprintf(stderr,
                     "[VENUS] get_string_vector_t: param=%p (obj=%s) not in DDR heap; treating as empty vector\n",
                     param, param_obj);
        vec.clear();
        return index;
    }

    if (count < 0 || count > kMaxReasonableCount) {
        std::fprintf(stderr,
                     "[VENUS] get_string_vector_t: suspicious count=%d (param=%p obj=%s ddr_off=%ld)\n",
                     count, param, param_obj, ddr_offset);
        // If the param buffer clearly lives inside libvenus, treat this as
        // completely bogus and ignore the vector contents.
        if (param_in_venus) {
            count = 0;
        } else if (count > kMaxReasonableCount) {
            count = kMaxReasonableCount;
        }
    }

    std::fprintf(stderr,
                 "[VENUS] get_string_vector_t override\n"
                 "  vec_size_before=%zu index=%d count=%d param=%p\n"
                 "  ra0=%p (obj=%s, symbol=%s+0x%lx)\n"
                 "  param_ddr_base=%p in_ddr=%d ddr_off=%ld\n"
                 "  param[0..15]=%02x %02x %02x %02x %02x %02x %02x %02x "
                 "%02x %02x %02x %02x %02x %02x %02x %02x\n",
                 vec.size(), index, count, param,
                 ra0, obj, sym, off,
                 __ddr_vbase, param_in_ddr ? 1 : 0, ddr_offset,
                 bytes[0], bytes[1], bytes[2], bytes[3],
                 bytes[4], bytes[5], bytes[6], bytes[7],
                 bytes[8], bytes[9], bytes[10], bytes[11],
                 bytes[12], bytes[13], bytes[14], bytes[15]);
    std::fflush(stderr);

    if (count <= 0) {
        vec.clear();
        return index;
    }

    vec.resize(static_cast<size_t>(count));
    for (int32_t i = 0; i < count; ++i) {
        _Z12get_string_tRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKvRi(
            vec[static_cast<size_t>(i)], param, index);
    }

    return index;
}


// Override OEM helper: get_string_t
// Mangled name: _Z12get_string_tRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKvRi
//
// The OEM implementation (from AEC_T41_16K_NS_OUT_UC.mgk) is roughly:
//   - read a 32-bit length from the param blob at param + index
//   - if length != 0, resize the std::string and memcpy the bytes
//   - advance index by the length and return the updated cursor
//
// We replicate those semantics here, but add a defensive clamp for obviously
// bogus lengths (e.g. when the param pointer is wrong and we would otherwise
// try to allocate hundreds of megabytes for a single string).
extern "C" int _Z12get_string_tRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKvRi(
    std::string &out,
    const void *param,
    int &index)
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

    unsigned char bytes[16];
    for (int i = 0; i < 16; ++i) {
        bytes[i] = 0;
    }
    const unsigned char *base = static_cast<const unsigned char *>(param);
    if (base) {
        uintptr_t addr = (uintptr_t)base;
        if (addr >= 0x10000u) {
            for (int i = 0; i < 16; ++i) {
                bytes[i] = base[i];
            }
        }
    }

    // Read string length (32-bit) from the parameter block at the current index.
    uint32_t len = 0;
    if (base) {
        uintptr_t addr = (uintptr_t)(base + index);
        if (addr >= 0x10000u) {
            memcpy(&len, base + index, sizeof(len));
            index += (int)sizeof(len);
        }
    }

    // Reasonable upper bound for string lengths in real .mgk models.
    static const uint32_t kMaxReasonableLen = 16384; // 16 KB

    Dl_info param_info{};
    const char *param_obj = "?";
    bool param_in_venus = false;
    if (param && dladdr(param, &param_info)) {
        param_obj = param_info.dli_fname ? param_info.dli_fname : "?";
        if (std::strstr(param_obj, "libvenus.so")) {
            param_in_venus = true;
        }
    }

    // Additionally, detect whether the param pointer is inside the logical
    // DDR heap (__ddr_vbase .. __ddr_vbase + 8MB) that we expose to .mgk.
    bool param_in_ddr = false;
    long ddr_offset = -1;
    if (__ddr_vbase && param) {
        uintptr_t ddr_base = (uintptr_t)__ddr_vbase;
        uintptr_t p = (uintptr_t)param;
        static const uintptr_t kDdrSpan = 8u * 1024u * 1024u;
        if (p >= ddr_base && p < ddr_base + kDdrSpan) {
            param_in_ddr = true;
            ddr_offset = (long)(p - ddr_base);
        }
    }

    if (param_in_ddr) {
        // If it clearly lives in DDR, don't also label it as "libvenus".
        param_in_venus = false;
    } else {
        std::fprintf(stderr,
                     "[VENUS] get_string_t: param=%p (obj=%s) not in DDR heap; treating as empty string\n",
                     param, param_obj);
        len = 0;
    }

    if (len > kMaxReasonableLen) {
        std::fprintf(stderr,
                     "[VENUS] get_string_t: suspicious len=%u (param=%p obj=%s ddr_off=%ld)\n",
                     len, param, param_obj, ddr_offset);
        if (param_in_venus) {
            // This is almost certainly a bogus param pointer. Treat as an empty
            // string so we do not walk off into unrelated text/data segments.
            len = 0;
        } else {
            len = kMaxReasonableLen;
        }
    }

    if (len == 0 || !base) {
        out.clear();
    } else {
        out.resize(static_cast<size_t>(len));
        memcpy(&out[0], base + index, (size_t)len);
        index += (int)len;
    }

    std::fprintf(stderr,
                 "[VENUS] get_string_t override\n"
                 "  index=%d len=%u param=%p\n"
                 "  ra0=%p (obj=%s, symbol=%s+0x%lx)\n"
                 "  param[0..15]=%02x %02x %02x %02x %02x %02x %02x %02x "
                 "%02x %02x %02x %02x %02x %02x %02x %02x\n",
                 index, len, param,
                 ra0, obj, sym, off,
                 bytes[0], bytes[1], bytes[2], bytes[3],
                 bytes[4], bytes[5], bytes[6], bytes[7],
                 bytes[8], bytes[9], bytes[10], bytes[11],
                 bytes[12], bytes[13], bytes[14], bytes[15]);
    std::fflush(stderr);

    return index;
}

} // namespace magik


// Shim for OEM read_common_param used by .mgk models.
//
// Mangled name (from AEC_T41_16K_NS_OUT_UC.mgk):
//   _Z17read_common_paramRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERSt6vectorIS4_SaIS4_EES9_RS6_IiSaIiEES9_S9_RiS9_S5_PKvSD_
//
// Our goal here is to move closer to the real OEM semantics while still keeping
// extremely strong safety guarantees. We now expose a fully-typed signature,
// and we *do* parse strings from the param blob via our hardened
// get_string_t/get_string_vector_t overrides, but we deliberately keep the
// caller-visible `index` cursor unchanged so we do not yet alter how the .mgk
// code walks the parameter buffer.
extern "C" int
read_common_param_shim(
    std::string &layer_name,
    std::vector<std::string> &bottoms,
    std::vector<std::string> &tops,
    std::vector<int> &int_params,
    std::vector<std::string> &extra_strs1,
    std::vector<std::string> &extra_strs2,
    int &some_flag,
    std::vector<std::string> &extra_strs3,
    std::string &extra_str,
    const void *param,
    int &index)
    __asm__("_Z17read_common_paramRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERSt6vectorIS4_SaIS4_EES9_RS6_IiSaIiEES9_S9_RiS9_S5_PKvSD_");

extern "C" int
read_common_param_shim(
    std::string &layer_name,
    std::vector<std::string> &bottoms,
    std::vector<std::string> &tops,
    std::vector<int> &int_params,
    std::vector<std::string> &extra_strs1,
    std::vector<std::string> &extra_strs2,
    int &some_flag,
    std::vector<std::string> &extra_strs3,
    std::string &extra_str,
    const void *param,
    int &index)
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

    // Reset outputs to known-safe defaults.
    bottoms.clear();
    tops.clear();
    int_params.clear();
    extra_strs1.clear();
    extra_strs2.clear();
    extra_strs3.clear();
    some_flag = 0;
    extra_str.clear();

    int local_index = index;

    // Safely try to decode the layer name and bottom/top tensor names without
    // affecting the caller's view of `index`.
    magik::_Z12get_string_tRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKvRi(
        layer_name, param, local_index);
    magik::_Z19get_string_vector_tRSt6vectorINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EEPKvRi(
        bottoms, param, local_index);
    magik::_Z19get_string_vector_tRSt6vectorINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EEPKvRi(
        tops, param, local_index);

    std::fprintf(stderr,
                 "[VENUS] read_common_param\n"
                 "  ra0=%p (obj=%s, symbol=%s+0x%lx)\n"
                 "  param=%p index=%d local_index=%d\n"
                 "  layer=\"%s\" bottoms=%zu tops=%zu\n",
                 ra0, obj, sym, off,
                 param, index, local_index,
                 layer_name.c_str(),
                 bottoms.size(), tops.size());
    std::fflush(stderr);

    // For now we intentionally do *not* modify `index` and simply return its
    // incoming value. This preserves the original stubbed walking behaviour
    // while giving us real names and I/O lists for reverse-engineering.
    return index;
}

// Shim for OEM read_layer_param used by .mgk models.
//
// Mangled name (from AEC_T41_16K_NS_OUT_UC.mgk backtrace):
//   _Z16read_layer_paramRiRtRyRjPKvS_
//
// We now expose a typed signature but still behave conservatively: we do not
// touch the param blob yet and we leave all outputs at safe defaults. This
// lets us collect call-site information without changing how the .mgk code
// iterates the parameter buffer.
extern "C" int
read_layer_param_shim(
    int &op_type,
    unsigned short &layer_id,
    unsigned long long &param_index,
    unsigned int &flags,
    const void *param,
    int &index)
    __asm__("_Z16read_layer_paramRiRtRyRjPKvS_");

extern "C" int
read_layer_param_shim(
    int &op_type,
    unsigned short &layer_id,
    unsigned long long &param_index,
    unsigned int &flags,
    const void *param,
    int &index)
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

    op_type = 0;
    layer_id = 0;
    param_index = 0;
    flags = 0;

    std::fprintf(stderr,
                 "[VENUS] read_layer_param\n"
                 "  ra0=%p (obj=%s, symbol=%s+0x%lx)\n"
                 "  param=%p index=%d\n",
                 ra0, obj, sym, off,
                 param, index);
    std::fflush(stderr);

    return index;
}



// Shim for OEM bn_scale_int8_param_init used by .mgk models.
//
// The OEM implementation currently crashes with SIGBUS inside
// bn_scale_int8_param_init for this AEC model. We interpose a very
// conservative shim that logs the call site and simply returns success,
// effectively treating BN/scale as a no-op while we reverse-engineer the
// full semantics.
extern "C" magik::venus::ReturnValue
bn_scale_int8_param_init_shim(...) __asm__(
    "_Z24bn_scale_int8_param_initNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS4_SaIS4_EES7_S5_IiSaIiEES7_S7_iS7_S4_iiiS9_S9_iiiiPvSA_xS5_IS9_SaIS9_EESC_S9_S9_S9_S9_bbiiibiiRS5_IN5magik5venus10TensorInfoESaISF_EERSt3mapIiSt4pairIS7_S7_ESt4lessIiESaISK_IKiSL_EEEPNSE_14MagikModelBase13PyramidConfigESt8functionIFNSE_11ReturnValueERSt10unique_ptrINSE_6kernel11KernelParamESt14default_deleteIS10_EEPNSE_8OpConfigEEE");

extern "C" magik::venus::ReturnValue
bn_scale_int8_param_init_shim(...)
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
                 "[VENUS] bn_scale_int8_param_init shim (raw) called\n"
                 "  ra0=%p (obj=%s, symbol=%s+0x%lx)\n",
                 ra0, obj, sym, off);
    std::fflush(stderr);

    // Treat BN/scale as a no-op for now so the model can continue.
    return magik::venus::ReturnValue(0);
}

// Shim for OEM conv2d_int8_param_init used by .mgk models.
//
// Mangled name (from AEC_T41_16K_NS_OUT_UC.mgk backtrace):
//   _Z22conv2d_int8_param_initPKvPvS1_xRSt6vectorIN5magik5venus10TensorInfoESaIS5_EERSt3mapIiSt4pairIS2_INSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaISG_EESI_ESt4lessIiESaISA_IKiSJ_EEEPNS4_14MagikModelBase13PyramidConfigESt8functionIFNS4_11ReturnValueERSt10unique_ptrINS4_6kernel11KernelParamESt14default_deleteISY_EEPNS4_8OpConfigEEE
//
// The OEM implementation asserts on output_size==1 and performs complex
// parameter parsing. For this demo we interpose a very conservative shim
// that simply logs the call site and returns immediately, effectively
// treating conv2d param initialization as a no-op so we avoid tripping
// internal asserts and corrupting STL state.
extern "C" void
conv2d_int8_param_init_shim(...) __asm__(
    "_Z22conv2d_int8_param_initPKvPvS1_xRSt6vectorIN5magik5venus10TensorInfoESaIS5_EERSt3mapIiSt4pairIS2_INSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaISG_EESI_ESt4lessIiESaISA_IKiSJ_EEEPNS4_14MagikModelBase13PyramidConfigESt8functionIFNS4_11ReturnValueERSt10unique_ptrINS4_6kernel11KernelParamESt14default_deleteISY_EEPNS4_8OpConfigEEE");

extern "C" void
conv2d_int8_param_init_shim(...)
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
                 "[VENUS] conv2d_int8_param_init shim (raw) called\n"
                 "  ra0=%p (obj=%s, symbol=%s+0x%lx)\n",
                 ra0, obj, sym, off);
    std::fflush(stderr);

    // No-op: we intentionally do not touch any of the arguments.
    return;
}

// Shim for OEM reshape_param_init used by .mgk models.
//
// Mangled name (from AEC_T41_16K_NS_OUT_UC.mgk backtrace):
//   _Z18reshape_param_initNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS4_SaIS4_EES7_S5_IiSaIiEES7_S7_iS7_S4_iS9_PvSA_xS5_IS9_SaIS9_EESC_S9_S9_S9_S9_bbiiibiiRS5_IN5magik5venus10TensorInfoESaISF_EERSt3mapIiSt4pairIS7_S7_ESt4lessIiESaISK_IKiSL_EEEPNSE_14MagikModelBase13PyramidConfigESt8functionIFNSE_11ReturnValueERSt10unique_ptrINSE_6kernel11KernelParamESt14default_deleteIS10_EEPNSE_8OpConfigEEE
//
// The OEM implementation parses reshape parameters and can hit asserts or
// invalid pointers when combined with our current stubbed helpers. For the
// demo we treat reshape param init as a no-op: we log the call site and
// immediately report success.
extern "C" magik::venus::ReturnValue
reshape_param_init_shim(...) __asm__(
    "_Z18reshape_param_initNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS4_SaIS4_EES7_S5_IiSaIiEES7_S7_iS7_S4_iS9_PvSA_xS5_IS9_SaIS9_EESC_S9_S9_S9_S9_bbiiibiiRS5_IN5magik5venus10TensorInfoESaISF_EERSt3mapIiSt4pairIS7_S7_ESt4lessIiESaISK_IKiSL_EEEPNSE_14MagikModelBase13PyramidConfigESt8functionIFNSE_11ReturnValueERSt10unique_ptrINSE_6kernel11KernelParamESt14default_deleteIS10_EEPNSE_8OpConfigEEE");

extern "C" magik::venus::ReturnValue
reshape_param_init_shim(...)
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
                 "[VENUS] reshape_param_init shim (raw) called\n"
                 "  ra0=%p (obj=%s, symbol=%s+0x%lx)\n",
                 ra0, obj, sym, off);
    std::fflush(stderr);

    return magik::venus::ReturnValue(0);
}

// Shim for OEM gru_param_init used by .mgk models.
//
// Mangled name (from AEC_T41_16K_NS_OUT_UC.mgk backtrace):
//   _Z14gru_param_initNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS4_SaIS4_EES7_S5_IiSaIiEES7_S7_iS7_S4_ibbiS9_iiiyiS9_S9_iiPvSA_xS5_IS9_SaIS9_EESC_S9_S9_S9_S9_bbiiibiiRS5_IN5magik5venus10TensorInfoESaISF_EERSt3mapIiSt4pairIS7_S7_ESt4lessIiESaISK_IKiSL_EEEPNSE_14MagikModelBase13PyramidConfigESt8functionIFNSE_11ReturnValueERSt10unique_ptrINSE_6kernel11KernelParamESt14default_deleteIS10_EEPNSE_8OpConfigEEE
//
// Similar to reshape_param_init, we conservatively treat GRU param init as a
// no-op for the demo. This avoids crashes from partially-initialized state
// while still letting the model wiring proceed.
extern "C" magik::venus::ReturnValue
gru_param_init_shim(...) __asm__(
    "_Z14gru_param_initNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS4_SaIS4_EES7_S5_IiSaIiEES7_S7_iS7_S4_ibbiS9_iiiyiS9_S9_iiPvSA_xS5_IS9_SaIS9_EESC_S9_S9_S9_S9_bbiiibiiRS5_IN5magik5venus10TensorInfoESaISF_EERSt3mapIiSt4pairIS7_S7_ESt4lessIiESaISK_IKiSL_EEEPNSE_14MagikModelBase13PyramidConfigESt8functionIFNSE_11ReturnValueERSt10unique_ptrINSE_6kernel11KernelParamESt14default_deleteIS10_EEPNSE_8OpConfigEEE");

extern "C" magik::venus::ReturnValue
gru_param_init_shim(...)
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
                 "[VENUS] gru_param_init shim (raw) called\n"
                 "  ra0=%p (obj=%s, symbol=%s+0x%lx)\n",
                 ra0, obj, sym, off);
    std::fflush(stderr);

    return magik::venus::ReturnValue(0);
}


// Shim for OEM add_int8_param_init used by .mgk models.
//
// Mangled name (from AEC_T41_16K_NS_OUT_UC.mgk backtrace):
//   _Z19add_int8_param_initPKvPvS1_xRSt6vectorIN5magik5venus10TensorInfoESaIS5_EERSt3mapIiSt4pairIS2_INSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaISG_EESI_ESt4lessIiESaISA_IKiSJ_EEEPNS4_14MagikModelBase13PyramidConfigESt8functionIFNS4_11ReturnValueERSt10unique_ptrINS4_6kernel11KernelParamESt14default_deleteISY_EEPNS4_8OpConfigEEE
//
// The OEM implementation enforces output_size==1 and currently aborts via
// assert. For the demo we intercept it and treat add/eltwise int8 param init
// as a no-op.
extern "C" void
add_int8_param_init_shim(...) __asm__(
    "_Z19add_int8_param_initPKvPvS1_xRSt6vectorIN5magik5venus10TensorInfoESaIS5_EERSt3mapIiSt4pairIS2_INSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaISG_EESI_ESt4lessIiESaISA_IKiSJ_EEEPNS4_14MagikModelBase13PyramidConfigESt8functionIFNS4_11ReturnValueERSt10unique_ptrINS4_6kernel11KernelParamESt14default_deleteISY_EEPNS4_8OpConfigEEE");

extern "C" void
add_int8_param_init_shim(...)
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
                 "[VENUS] add_int8_param_init shim (raw) called\n"
                 "  ra0=%p (obj=%s, symbol=%s+0x%lx)\n",
                 ra0, obj, sym, off);
    std::fflush(stderr);

    // No-op: we intentionally do not touch any of the arguments.
    return;
}

// Shim for OEM permute_param_init used by .mgk models.
//
// Mangled name (from AEC_T41_16K_NS_OUT_UC.mgk backtrace):
//   _Z18permute_param_initNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS4_SaIS4_EES7_S5_IiSaIiEES7_S7_iS7_S4_iS9_PvSA_xS5_IS9_SaIS9_EESC_S9_S9_S9_S9_bbiiibiiRS5_IN5magik5venus10TensorInfoESaISF_EERSt3mapIiSt4pairIS7_S7_ESt4lessIiESaISK_IKiSL_EEEPNSE_14MagikModelBase13PyramidConfigESt8functionIFNSE_11ReturnValueERSt10unique_ptrINSE_6kernel11KernelParamESt14default_deleteIS10_EEPNSE_8OpConfigEEE
//
// For the demo we again treat permute param init as a no-op, just logging
// the call site and returning success.
extern "C" magik::venus::ReturnValue
permute_param_init_shim(...) __asm__(
    "_Z18permute_param_initNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS4_SaIS4_EES7_S5_IiSaIiEES7_S7_iS7_S4_iS9_PvSA_xS5_IS9_SaIS9_EESC_S9_S9_S9_S9_bbiiibiiRS5_IN5magik5venus10TensorInfoESaISF_EERSt3mapIiSt4pairIS7_S7_ESt4lessIiESaISK_IKiSL_EEEPNSE_14MagikModelBase13PyramidConfigESt8functionIFNSE_11ReturnValueERSt10unique_ptrINSE_6kernel11KernelParamESt14default_deleteIS10_EEPNSE_8OpConfigEEE");

extern "C" magik::venus::ReturnValue
permute_param_init_shim(...)
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
                 "[VENUS] permute_param_init shim (raw) called\n"
                 "  ra0=%p (obj=%s, symbol=%s+0x%lx)\n",
                 ra0, obj, sym, off);
    std::fflush(stderr);

    return magik::venus::ReturnValue(0);
}


// Shim for OEM concat_int8_param_init used by .mgk models.
//
// Mangled name (from AEC_T41_16K_NS_OUT_UC.mgk backtrace):
//   _Z22concat_int8_param_initNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS4_SaIS4_EES7_S5_IiSaIiEES7_S7_iS7_S4_iiiS9_S9_S9_S9_iS5_IS5_IS9_SaIS9_EESaISB_EEPvSE_xSB_SB_S9_S9_S9_S9_bbiiibiiRS5_IN5magik5venus10TensorInfoESaISH_EERSt3mapIiSt4pairIS7_S7_ESt4lessIiESaISM_IKiSN_EEEPNSG_14MagikModelBase13PyramidConfigESt8functionIFNSG_11ReturnValueERSt10unique_ptrINSG_6kernel11KernelParamESt14default_deleteIS12_EEPNSG_8OpConfigEEE
//
// For now we treat concat int8 param init as a no-op, mirroring the other
// param_init shims to keep the model building without crashes.
extern "C" magik::venus::ReturnValue
concat_int8_param_init_shim(...) __asm__(
    "_Z22concat_int8_param_initNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS4_SaIS4_EES7_S5_IiSaIiEES7_S7_iS7_S4_iiiS9_S9_S9_S9_iS5_IS5_IS9_SaIS9_EESaISB_EEPvSE_xSB_SB_S9_S9_S9_S9_bbiiibiiRS5_IN5magik5venus10TensorInfoESaISH_EERSt3mapIiSt4pairIS7_S7_ESt4lessIiESaISM_IKiSN_EEEPNSG_14MagikModelBase13PyramidConfigESt8functionIFNSG_11ReturnValueERSt10unique_ptrINSG_6kernel11KernelParamESt14default_deleteIS12_EEPNSG_8OpConfigEEE");

extern "C" magik::venus::ReturnValue
concat_int8_param_init_shim(...)
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
                 "[VENUS] concat_int8_param_init shim (raw) called\n"
                 "  ra0=%p (obj=%s, symbol=%s+0x%lx)\n",
                 ra0, obj, sym, off);
    std::fflush(stderr);

    return magik::venus::ReturnValue(0);
}


// Shim for OEM upsample_int8_param_init used by .mgk models.
//
// Mangled name (from AEC_T41_16K_NS_OUT_UC.mgk backtrace):
//   _Z24upsample_int8_param_initNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS4_SaIS4_EES7_S5_IiSaIiEES7_S7_iS7_S4_iiiifS5_IfSaIfEEPvSC_xS5_IS9_SaIS9_EESE_S9_S9_S9_S9_bbiiibiiRS5_IN5magik5venus10TensorInfoESaISH_EERSt3mapIiSt4pairIS7_S7_ESt4lessIiESaISM_IKiSN_EEEPNSG_14MagikModelBase13PyramidConfigESt8functionIFNSG_11ReturnValueERSt10unique_ptrINSG_6kernel11KernelParamESt14default_deleteIS12_EEPNSG_8OpConfigEEE
//
// As with other *_param_init helpers, we make this a harmless no-op that
// only logs the caller and returns success for the demo.
extern "C" magik::venus::ReturnValue
upsample_int8_param_init_shim(...) __asm__(
    "_Z24upsample_int8_param_initNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS4_SaIS4_EES7_S5_IiSaIiEES7_S7_iS7_S4_iiiifS5_IfSaIfEEPvSC_xS5_IS9_SaIS9_EESE_S9_S9_S9_S9_bbiiibiiRS5_IN5magik5venus10TensorInfoESaISH_EERSt3mapIiSt4pairIS7_S7_ESt4lessIiESaISM_IKiSN_EEEPNSG_14MagikModelBase13PyramidConfigESt8functionIFNSG_11ReturnValueERSt10unique_ptrINSG_6kernel11KernelParamESt14default_deleteIS12_EEPNSG_8OpConfigEEE");

extern "C" magik::venus::ReturnValue
upsample_int8_param_init_shim(...)
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
                 "[VENUS] upsample_int8_param_init shim (raw) called\n"
                 "  ra0=%p (obj=%s, symbol=%s+0x%lx)\n",
                 ra0, obj, sym, off);
    std::fflush(stderr);

    return magik::venus::ReturnValue(0);
}






// Shim for OEM format_convert_param_init used by .mgk models.
//
// Original (demangled) signature (trimmed to key parts):
//   magik::venus::ReturnValue format_convert_param_init(
//       std::string,
//       std::vector<std::string>, std::vector<std::string>,
//       std::vector<int>, std::vector<std::string>, std::vector<std::string>,
//       int, std::vector<std::string>, std::string,
//       int, int, int, int, int,
//       void*, void*, long long,
//       std::vector<std::vector<int>>, std::vector<std::vector<int>>,
//       std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>,
//       bool, bool, int, int, int, bool, int, int,
//       std::vector<magik::venus::TensorInfo>&,
//       std::map<int, std::pair<std::vector<std::string>, std::vector<std::string>>>&,
//       magik::venus::MagikModelBase::PyramidConfig*,
//       std::function<magik::venus::ReturnValue(
//           std::unique_ptr<magik::venus::kernel::KernelParam>&,
//           magik::venus::OpConfig*)>);
//
// We implement a conservative shim that logs the call site and argument
// sizes but does not dereference any of the opaque parameter pointers.
// For now we simply report success so that the model can continue past
// format_convert_param_init while we incrementally reconstruct the real
// behavior.
extern "C" magik::venus::ReturnValue
format_convert_param_init_shim(...) __asm__("_Z25format_convert_param_initNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS4_SaIS4_EES7_S5_IiSaIiEES7_S7_iS7_S4_iiiiiPvSA_xS5_IS9_SaIS9_EESC_S9_S9_S9_S9_bbiiibiiRS5_IN5magik5venus10TensorInfoESaISF_EERSt3mapIiSt4pairIS7_S7_ESt4lessIiESaISK_IKiSL_EEEPNSE_14MagikModelBase13PyramidConfigESt8functionIFNSE_11ReturnValueERSt10unique_ptrINSE_6kernel11KernelParamESt14default_deleteIS10_EEPNSE_8OpConfigEEE");

extern "C" magik::venus::ReturnValue
format_convert_param_init_shim(...)
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
                 "[VENUS] format_convert_param_init shim (raw) called\n"
                 "  ra0=%p (obj=%s, symbol=%s+0x%lx)\n",
                 ra0, obj, sym, off);
    std::fflush(stderr);

    return magik::venus::ReturnValue(0);
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


extern "C" void __assert_fail(const char *expr,
                              const char *file,
                              unsigned int line,
                              const char *func)
{
    // Interpose libc assert failures coming from OEM Magik kernels.
    std::fprintf(stderr,
                 "[VENUS] __assert_fail intercepted: expr=\"%s\" file=\"%s\" line=%u func=\"%s\"\n",
                 expr ? expr : "(null)",
                 file ? file : "(null)",
                 line,
                 func ? func : "(null)");
    std::fflush(stderr);

    if (expr && file && func &&
        std::strstr(expr, "output_size==1") &&
        std::strstr(file, "magik_op_override.cpp"))
    {
        std::fprintf(stderr,
                     "[VENUS] ignoring magik_op_override.cpp output_size==1 assert in %s; continuing execution\n",
                     func ? func : "(null)");
        std::fflush(stderr);
        return;
    }

    using assert_fail_fn = void (*)(const char *, const char *, unsigned int, const char *);
    static assert_fail_fn real_assert_fail = nullptr;
    if (!real_assert_fail) {
        real_assert_fail = (assert_fail_fn)dlsym(RTLD_NEXT, "__assert_fail");
    }

    if (real_assert_fail) {
        real_assert_fail(expr, file, line, func);
    } else {
        std::fprintf(stderr,
                     "[VENUS] __assert_fail: no RTLD_NEXT implementation, calling abort()\n");
        std::fflush(stderr);
        std::abort();
    }
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
 * Lightweight global memchr override for diagnostics.
 *
 * We use this to catch suspicious calls where the pointer is clearly
 * bogus (e.g., very low addresses that will trigger SIGBUS). Logging
 * is intentionally minimal and we avoid recursion by delegating to the
 * real memchr() when invoked from within our own hook.
 */
extern "C" void *memchr(const void *s, int c, size_t n)
{
    using memchr_fn = void *(*)(const void *, int, size_t);
    static memchr_fn real_memchr = nullptr;
    static int in_hook = 0;

    if (!real_memchr) {
        real_memchr = (memchr_fn)dlsym(RTLD_NEXT, "memchr");
    }

    if (!real_memchr) {
        /* Fallback: simple byte-wise scan. */
        const unsigned char *p = static_cast<const unsigned char *>(s);
        const unsigned char *end = p + n;
        unsigned char uc = static_cast<unsigned char>(c);
        for (; p < end; ++p) {
            if (*p == uc)
                return const_cast<unsigned char *>(p);
        }
        return nullptr;
    }

    if (in_hook) {
        return real_memchr(s, c, n);
    }

    in_hook = 1;

    uintptr_t addr = (uintptr_t)s;
    bool suspicious_low = (addr != 0 && addr < 0x10000u && n > 0);

    // Also treat clearly unmapped regions as suspicious so we avoid SIGBUS.
    bool suspicious_unmapped = false;
    if (s && n > 0) {
        size_t check_len = n;
        if (check_len > 4096u)
            check_len = 4096u;
        if (!venus_is_region_mapped(s, check_len)) {
            suspicious_unmapped = true;
        }
    }

    bool suspicious = suspicious_low || suspicious_unmapped;

    if (suspicious) {
        void *ra0 = __builtin_return_address(0);
        Dl_info info0{};
        const char *obj = "?";
        const char *sym = "?";
        unsigned long off = 0;
        void *base = nullptr;
        if (dladdr(ra0, &info0)) {
            obj = info0.dli_fname ? info0.dli_fname : "?";
            sym = info0.dli_sname ? info0.dli_sname : "?";
            off = (unsigned long)((char *)ra0 - (char *)info0.dli_saddr);
            base = info0.dli_fbase;
        }

        std::fprintf(stderr,
                     "[VENUS] memchr_hook(s=%p, c=%d, n=%zu) suspicious_low=%d suspicious_unmapped=%d\n"
                     "        caller=%p (obj=%s, base=%p, symbol=%s+0x%lx)\n",
                     s, c, (size_t)n,
                     suspicious_low ? 1 : 0, suspicious_unmapped ? 1 : 0,
                     ra0, obj, base, sym, off);
        std::fflush(stderr);

        // Avoid calling real memchr on obviously bad addresses to prevent SIGBUS.
        in_hook = 0;
        return nullptr;
    }

    void *ret = real_memchr(s, c, n);
    in_hook = 0;
    return ret;
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

    // Treat clearly bogus or unmapped regions as suspicious so we can
    // avoid SIGBUS when OEM code accidentally passes bad pointers.
    size_t check_len = n;
    if (check_len > 4096u)
        check_len = 4096u;

    bool src_mapped = venus_is_region_mapped(src, check_len);
    bool dst_mapped = venus_is_region_mapped(dest, check_len);
    bool suspicious_low = (s != 0 && s < 0x10000u && n > 0);
    bool suspicious = suspicious_low || !src_mapped || !dst_mapped;

    bool log = false;
    /* Log copies that touch the high virtual ranges where NNA/DDR/ORAM
     * mappings and .mgk data tend to live, or that are unusually large or
     * otherwise suspicious.
     */
    if (n > (1u << 20)) { /* >1MB */
        log = true;
    }
    if ((d >= 0x76000000u && d < 0x78000000u) ||
        (s >= 0x76000000u && s < 0x78000000u) ||
        suspicious) {
        log = true;
    }

    if (log) {
        void *ra0 = __builtin_return_address(0);
        Dl_info info0{};
        const char *obj = "?";
        const char *sym = "?";
        unsigned long off = 0;
        void *base = nullptr;
        if (dladdr(ra0, &info0)) {
            obj = info0.dli_fname ? info0.dli_fname : "?";
            sym = info0.dli_sname ? info0.dli_sname : "?";
            off = (unsigned long)((char *)ra0 - (char *)info0.dli_saddr);
            base = info0.dli_fbase;
        }

        Dl_info src_info{};
        const char *src_obj = "?";
        void *src_base = nullptr;
        if (dladdr(src, &src_info)) {
            src_obj = src_info.dli_fname ? src_info.dli_fname : "?";
            src_base = src_info.dli_fbase;
        }

        Dl_info dst_info{};
        const char *dst_obj = "?";
        void *dst_base = nullptr;
        if (dladdr(dest, &dst_info)) {
            dst_obj = dst_info.dli_fname ? dst_info.dli_fname : "?";
            dst_base = dst_info.dli_fbase;
        }

        std::fprintf(stderr,
                     "[VENUS] memcpy(dest=%p, src=%p, n=%zu)\n"
                     "        caller=%p (obj=%s, base=%p, symbol=%s+0x%lx)\n"
                     "        src_obj=%s base=%p mapped=%d\n"
                     "        dst_obj=%s base=%p mapped=%d\n"
                     "        suspicious_low=%d suspicious_unmapped_src=%d suspicious_unmapped_dst=%d\n",
                     dest, src, (size_t)n,
                     ra0, obj, base, sym, off,
                     src_obj, src_base, src_mapped ? 1 : 0,
                     dst_obj, dst_base, dst_mapped ? 1 : 0,
                     suspicious_low ? 1 : 0,
                     (!src_mapped) ? 1 : 0,
                     (!dst_mapped) ? 1 : 0);
        std::fflush(stderr);
    }

    // If the source or destination region is clearly unmapped or the source
    // address is implausibly low, refuse the copy to avoid crashing the
    // process. For the demo we simply leave the destination untouched.
    if (suspicious) {
        return dest;
    }

    using memcpy_fn = void *(*)(void *, const void *, size_t);
    static memcpy_fn real_memcpy = nullptr;
    if (!real_memcpy) {
        real_memcpy = (memcpy_fn)dlsym(RTLD_NEXT, "memcpy");
    }

    if (!real_memcpy) {
        /* Extremely unlikely: if dlsym fails, fall back to a simple
         * byte-wise implementation.
         */
        unsigned char *dptr = static_cast<unsigned char *>(dest);
        const unsigned char *sptr = static_cast<const unsigned char *>(src);

        if (dptr <= sptr || dptr >= sptr + n) {
            for (size_t i = 0; i < n; ++i)
                dptr[i] = sptr[i];
        } else {
            for (size_t i = n; i-- > 0; )
                dptr[i] = sptr[i];
        }
        return dest;
    }

    return real_memcpy(dest, src, n);
}




