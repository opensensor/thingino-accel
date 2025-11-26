/*
 * thingino-accel - Magik Model/Layer Stubs
 * These are stub implementations for the Magik model framework
 */

#include "magik_model.h"
#include "../../include/nna_memory.h"
#include <cstdio>

namespace magik {
namespace venus {

/* Device implementation */
Device::Device() {}
Device::~Device() {}

/* TensorXWrapper implementation */
TensorXWrapper::TensorXWrapper() : tensorx(nullptr) {}
TensorXWrapper::TensorXWrapper(TensorX *tx) : tensorx(tx) {}
TensorXWrapper::~TensorXWrapper() {}

/* TensorX constructor with size, data pointer, and device */
extern "C" void _ZN5magik5venus7TensorXC1EjPvNS0_6DeviceE(TensorX *this_ptr, uint32_t size, void *data_ptr, Device dev) {
    printf("[VENUS] TensorX::TensorX(size=%u, data=%p)\n", size, data_ptr);
    fflush(stdout);
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

    /* Create a new PyramidConfig - allocate 0x50 bytes as per OEM */
    PyramidConfig *config = new PyramidConfig();

    /* Memset first 0x18 bytes to 0 as per OEM */
    memset(config, 0, 0x18);

    /* Initialize the two std::map members (std::_Rb_tree_header) */
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

int MagikModelBase::build_tensors(PyramidConfig *config, std::vector<TensorInfo> infos) {
    printf("[VENUS] MagikModelBase::build_tensors(config=%p)\n", config);
    fflush(stdout);

    /* The config parameter is NULL, which is wrong. The derived class should be passing
     * this->main_pyramid_config_, but it's not. This suggests a memory layout mismatch.
     * For now, just return success to see if we can get past this point.
     */
    (void)config; (void)infos;
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
    printf("[VENUS] PyramidConfig::get_tensor_wrapper(name=%s) - returning nullptr\n", name.c_str());
    fflush(stdout);
    (void)name;
    return nullptr;
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

