/*
 * thingino-accel - Model Loader for .mgk files
 * Instantiates DerivedMagikModel from compiled .mgk shared libraries
 */

#include "model_loader.h"
#include "magik_model.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include "../../include/nna_types.h"

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
/* NOTE: The DerivedMagikModel constructor signature (from symbol analysis) is:
 *   DerivedMagikModel(void*, long long, void*, long long, void*, long long,
 *                     MemAllocMode, ModuleMode)
 * This appears to be 3 pairs of (pointer, size):
 *   - (ddr_ptr, ddr_size)
 *   - (oram_ptr, oram_size)
 *   - (extra_ptr, extra_size)
 */
typedef MagikModelBase* (*CreateFunction)(
    void *ddr_ptr,
    long long ddr_size,
    void *oram_ptr,
    long long oram_size,
    void *extra_ptr,
    long long extra_size,
    ModelMemoryInfoManager::MemAllocMode mem_mode,
    MagikModelBase::ModuleMode module_mode);

/* Model loader implementation */
struct ModelLoader {
    void *dl_handle;
    MagikModelBase *model_instance;
    void *model_data;      /* Model file loaded into DDR */
    size_t model_data_size;
    size_t elf_end_offset; /* Offset where ELF structure ends in the file */

    ModelLoader() : dl_handle(nullptr), model_instance(nullptr),
                    model_data(nullptr), model_data_size(0), elf_end_offset(0) {}

    ~ModelLoader() {
        if (model_instance) {
            delete model_instance;
        }
        if (dl_handle) {
            dlclose(dl_handle);
        }
        /* Note: model_data is in DDR and should not be freed here */
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

/* Helper function to find the end of ELF structure in an MGK file.
 * Returns the offset where weight data begins, or 0 on error.
 */
static size_t find_elf_end(FILE *fp) {
    /* Read ELF header to find section header table location */
    unsigned char ehdr[52];
    fseek(fp, 0, SEEK_SET);
    if (fread(ehdr, 1, 52, fp) != 52) {
        return 0;
    }

    /* Verify ELF magic */
    if (ehdr[0] != 0x7f || ehdr[1] != 'E' || ehdr[2] != 'L' || ehdr[3] != 'F') {
        return 0;
    }

    /* e_shoff (section header table offset) at byte 32 (32-bit little endian) */
    uint32_t e_shoff = ehdr[32] | (ehdr[33] << 8) | (ehdr[34] << 16) | (ehdr[35] << 24);

    /* e_shentsize at byte 46 */
    uint16_t e_shentsize = ehdr[46] | (ehdr[47] << 8);

    /* e_shnum at byte 48 */
    uint16_t e_shnum = ehdr[48] | (ehdr[49] << 8);

    /* ELF ends after section header table */
    size_t elf_end = e_shoff + (e_shnum * e_shentsize);

    return elf_end;
}

void* load_mgk_model(const char *path) {
    if (!path) {
        fprintf(stderr, "load_mgk_model: NULL path\n");
        return nullptr;
    }

    ModelLoader *loader = new ModelLoader();

    /* Open the file and find weight data location */
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "load_mgk_model: Failed to open %s\n", path);
        delete loader;
        return nullptr;
    }

    fseek(fp, 0, SEEK_END);
    size_t file_size = ftell(fp);

    /* Find where ELF structure ends - weight data starts after this */
    size_t elf_end = find_elf_end(fp);
    if (elf_end == 0 || elf_end >= file_size) {
        fprintf(stderr, "load_mgk_model: Failed to parse ELF structure\n");
        fclose(fp);
        delete loader;
        return nullptr;
    }

    size_t weight_data_size = file_size - elf_end;

    printf("load_mgk_model: File size: %zu, ELF ends at: %zu, weight data: %zu bytes\n",
           file_size, elf_end, weight_data_size);
    fflush(stdout);

    /* Load the weight data into DDR memory */
    fseek(fp, elf_end, SEEK_SET);
    loader->model_data = __ddr_vbase;
    loader->model_data_size = weight_data_size;

    size_t bytes_read = fread(loader->model_data, 1, weight_data_size, fp);
    fclose(fp);

    if (bytes_read != weight_data_size) {
        fprintf(stderr, "load_mgk_model: Failed to read weight data (got %zu, expected %zu)\n",
                bytes_read, weight_data_size);
        delete loader;
        return nullptr;
    }

    /* Store the ELF end offset - we'll need this to adjust the DDR pointer */
    loader->elf_end_offset = elf_end;

    printf("load_mgk_model: Loaded %zu bytes of weight data into DDR at %p\n",
           weight_data_size, loader->model_data);
    printf("load_mgk_model: ELF ends at offset 0x%zx, will adjust DDR pointer\n", elf_end);
    fflush(stdout);

    /* Now load the .mgk as a shared library using dlopen */
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

    /* Call the create function to instantiate the model.
     * Based on symbol analysis, DerivedMagikModel constructor takes:
     *   (void* ddr, long long ddr_size, void* oram, long long oram_size,
     *    void* extra, long long extra_size, MemAllocMode, ModuleMode)
     *
     * The model's compiled code uses offsets relative to the file start to access
     * parameters. Since we only loaded the weight data (which starts at elf_end_offset),
     * we need to adjust the DDR pointer so that:
     *   adjusted_ddr_ptr + file_offset = actual_data_location
     *
     * If weight data is at loader->model_data and starts at file offset elf_end_offset:
     *   adjusted_ddr_ptr = loader->model_data - elf_end_offset
     *
     * This way, when the model accesses offset X (where X >= elf_end_offset),
     * it will correctly access loader->model_data + (X - elf_end_offset).
     */
    void *ddr_ptr = (char*)loader->model_data - loader->elf_end_offset;
    /* Report the full file size so the model thinks it has the whole file */
    long long ddr_size = (long long)(loader->model_data_size + loader->elf_end_offset);

    printf("load_mgk_model: Adjusted DDR pointer: %p (original: %p, offset: 0x%zx)\n",
           ddr_ptr, loader->model_data, loader->elf_end_offset);
    fflush(stdout);
    void *oram_ptr = __oram_vbase;
    long long oram_size = 384 * 1024;      /* 384 KB ORAM */
    void *extra_ptr = nullptr;
    long long extra_size = 0;
    ModelMemoryInfoManager::MemAllocMode mem_mode =
        ModelMemoryInfoManager::MemAllocMode::DEFAULT;
    MagikModelBase::ModuleMode module_mode = MagikModelBase::ModuleMode::NORMAL;

    printf("Calling create(ddr=%p, ddr_size=%lld, oram=%p, oram_size=%lld, extra=%p, extra_size=%lld, mem_mode=%d, module_mode=%d)...\n",
           ddr_ptr, ddr_size, oram_ptr, oram_size, extra_ptr, extra_size,
           (int)mem_mode, (int)module_mode);
    fflush(stdout);

    try {
        printf("About to call create_fn...\n");
        fflush(stdout);

        loader->model_instance = create_fn(ddr_ptr, ddr_size,
                                           oram_ptr, oram_size,
                                           extra_ptr, extra_size,
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

int mgk_model_get_io_tensor_info(void *handle,
                                 int is_input,
                                 unsigned int index,
                                 void **data_out,
                                 int dims_out[4],
                                 int *ndim_out,
                                 int *dtype_out,
                                 int *format_out)
{
    if (!handle || !data_out || !dims_out || !ndim_out ||
        !dtype_out || !format_out) {
        return -1;
    }

    ModelLoader *loader = static_cast<ModelLoader*>(handle);
    MagikModelBase *model = loader ? loader->model_instance : nullptr;
    if (!model) {
        return -1;
    }

    TensorXWrapper *wrapper = nullptr;

    /* Use the base-class IO vectors that we populate from TensorInfo flags
     * during MagikModelBase::build_tensors(). This avoids depending on any
     * DerivedMagikModel::get_input/get_output overrides whose internal
     * containers or layouts we don't fully control.
     *
     * IMPORTANT: We do NOT fall back to the OEM virtual methods when the index
     * is out of range. Doing so triggers SIGBUS due to corrupted OEM internal
     * state. Instead, return -1 immediately to signal "no more tensors".
     */
    try {
        if (is_input) {
            if (index < model->inputs_.size()) {
                wrapper = model->inputs_[index];
                printf("mgk_model_get_io_tensor_info: using MagikModelBase::inputs_[%u]=%p\n",
                       index, (void*)wrapper);
            } else {
                /* Index out of range - do NOT call OEM virtuals, just return error. */
                printf("mgk_model_get_io_tensor_info: input index %u out of range (size=%zu) - returning -1\n",
                       index, model->inputs_.size());
                return -1;
            }
        } else {
            if (index < model->outputs_.size()) {
                wrapper = model->outputs_[index];
                printf("mgk_model_get_io_tensor_info: using MagikModelBase::outputs_[%u]=%p\n",
                       index, (void*)wrapper);
            } else {
                /* Index out of range - do NOT call OEM virtuals, just return error. */
                printf("mgk_model_get_io_tensor_info: output index %u out of range (size=%zu) - returning -1\n",
                       index, model->outputs_.size());
                return -1;
            }
        }
    } catch (...) {
        printf("mgk_model_get_io_tensor_info: exception while accessing base IO vectors (is_input=%d index=%u)\n",
               is_input, index);
        return -1;
    }

    /* If we got here, wrapper should be valid from the base vectors.
     * Only fall back to OEM virtuals if the base vectors are completely empty
     * (i.e., build_tensors was never called or TensorInfo parsing failed entirely).
     */
    if (!wrapper && is_input && model->inputs_.empty()) {
        printf("mgk_model_get_io_tensor_info: inputs_ empty, trying OEM virtual get_input(%u)\n", index);
        try {
            wrapper = model->get_input(static_cast<int>(index));
        } catch (...) {
            printf("mgk_model_get_io_tensor_info: exception calling OEM get_input(%u)\n", index);
            return -1;
        }
    } else if (!wrapper && !is_input && model->outputs_.empty()) {
        printf("mgk_model_get_io_tensor_info: outputs_ empty, trying OEM virtual get_output(%u)\n", index);
        try {
            wrapper = model->get_output(static_cast<int>(index));
        } catch (...) {
            printf("mgk_model_get_io_tensor_info: exception calling OEM get_output(%u)\n", index);
            return -1;
        }
    }

    if (!wrapper || !wrapper->tensorx) {
        printf("mgk_model_get_io_tensor_info: NULL wrapper or tensorx (wrapper=%p)\n",
               (void*)wrapper);
        return -1;
    }

    TensorX *tx = wrapper->tensorx;
    if (!tx) {
        printf("mgk_model_get_io_tensor_info: wrapper->tensorx is NULL\n");
        return -1;
    }

    long raw_ndims = 0;
    if (tx->dims_begin && tx->dims_end) {
        raw_ndims = static_cast<long>(tx->dims_end - tx->dims_begin);
    }
    printf("mgk_model_get_io_tensor_info: tx=%p dims_begin=%p dims_end=%p raw_ndims=%ld data=%p bytes=%u offset=%u\n",
           (void*)tx,
           (void*)tx->dims_begin,
           (void*)tx->dims_end,
           raw_ndims,
           tx->data,
           (unsigned)tx->bytes,
           (unsigned)tx->data_offset);
    printf("  TensorX dump: dims_meta0=%p dims_meta1=%p dims_meta2=%p\n",
           (void*)tx->dims_meta0,
           (void*)tx->dims_meta1,
           (void*)tx->dims_meta2);
    printf("  TensorX dump: align=%u dtype=%d format=%d bytes=%u owns_data=%u reserved3=%d data_offset=%u reserved4=%d\n",
           (unsigned)tx->align,
           (int)tx->dtype,
           (int)tx->format,
           (unsigned)tx->bytes,
           (unsigned)tx->owns_data,
           (int)tx->reserved3,
           (unsigned)tx->data_offset,
           (int)tx->reserved4);
    fflush(stdout);

    if (!tx->data) {
        /* No backing buffer; nothing we can expose. */
        return -1;
    }

    /* Compute logical data pointer: base + data_offset. */
    unsigned char *base = static_cast<unsigned char *>(tx->data);
    unsigned char *ptr  = base + tx->data_offset;
    *data_out = static_cast<void *>(ptr);

    /* Default dims and ndim. */
    for (int i = 0; i < 4; ++i) {
        dims_out[i] = 1;
    }
    int ndim = 0;

    /* Obtain shape from TensorX if available. */
    printf("mgk_model_get_io_tensor_info: about to call get_shape()\n");
    fflush(stdout);
    try {
        shape_t shape = tx->get_shape();
        printf("mgk_model_get_io_tensor_info: get_shape() returned, size=%zu\n", shape.size());
        fflush(stdout);
        ndim = static_cast<int>(shape.size());
        if (ndim <= 0) {
            ndim = 1;
        } else if (ndim > 4) {
            /* Collapse extra dimensions into the first one. */
            size_t total = 1;
            for (size_t i = 0; i < shape.size(); ++i) {
                int32_t d = shape[i] > 0 ? shape[i] : 1;
                total *= static_cast<size_t>(d);
            }
            dims_out[0] = static_cast<int>(total);
            ndim = 1;
        } else {
            for (int i = 0; i < ndim; ++i) {
                int32_t d = shape[i] > 0 ? shape[i] : 1;
                dims_out[i] = d;
            }
        }
        printf("mgk_model_get_io_tensor_info: shape processing done, ndim=%d (shape will be destroyed after this line)\n", ndim);
        fflush(stdout);
        /* Shape vector destructor runs at end of this block - add explicit scope to isolate crash. */
    } catch (...) {
        /* On any failure, keep default dims and ndim=1. */
        printf("mgk_model_get_io_tensor_info: get_shape() threw exception\n");
        fflush(stdout);
        ndim = 1;
    }
    printf("mgk_model_get_io_tensor_info: try-catch block complete, shape destroyed, ndim=%d\n", ndim);
    fflush(stdout);

    printf("mgk_model_get_io_tensor_info: setting *ndim_out=%d\n", ndim);
    fflush(stdout);
    *ndim_out = ndim;

    /* Map Venus DataType to NNA dtype. */
    nna_dtype_t nna_dtype = NNA_DTYPE_UINT8;
    switch (tx->dtype) {
        case DataType::FP32:  nna_dtype = NNA_DTYPE_FLOAT32; break;
        case DataType::UINT32: nna_dtype = NNA_DTYPE_UINT32; break;
        case DataType::INT32:  nna_dtype = NNA_DTYPE_INT32;  break;
        case DataType::UINT16: nna_dtype = NNA_DTYPE_UINT16; break;
        case DataType::INT16:  nna_dtype = NNA_DTYPE_INT16;  break;
        case DataType::UINT8:  nna_dtype = NNA_DTYPE_UINT8;  break;
        case DataType::INT8:   nna_dtype = NNA_DTYPE_INT8;   break;
        default:
            nna_dtype = NNA_DTYPE_UINT8;
            break;
    }
    *dtype_out = static_cast<int>(nna_dtype);

    /* Map Venus TensorFormat to NNA format. */
    nna_format_t nna_fmt = NNA_FORMAT_NHWC;
    switch (tx->format) {
        case TensorFormat::NV12:
            nna_fmt = NNA_FORMAT_NV12;
            break;
        case TensorFormat::NHWC:
        default:
            nna_fmt = NNA_FORMAT_NHWC;
            break;
    }
    *format_out = static_cast<int>(nna_fmt);

    return 0;
}

size_t mgk_model_get_forward_memory_size(void *handle)
{
    if (!handle) {
        return 0;
    }

    ModelLoader *loader = static_cast<ModelLoader*>(handle);
    MagikModelBase *model = loader ? loader->model_instance : nullptr;
    if (!model) {
        return 0;
    }

    /*
     * IMPORTANT:
     * -----------
     *
     * Calling through the virtual `MagikModelBase::get_forward_memory_size()`
     * can dispatch into the OEM-derived vtable inside the .mgk. Our current
     * reverse-engineered class layout is good enough for data members and the
     * non-virtual helpers used during build_tensors(), but the vtable layout
     * for some models is still fragile. In particular, letting this call go
     * through the vtable has been observed to jump to the RTTI object for
     * `magik::venus::layer::ConvParam`, causing a SIGBUS at
     * `_ZTIN5magik5venus5layer9ConvParamE`.
     *
     * To keep things robust while we refine the ABI, we explicitly bind to the
     * base-class implementation. That implementation only walks
     * `pyramid_configs_` and `TensorXWrapper` instances that we construct in
     * our own `MagikModelBase::build_tensors`, so it stays entirely within our
     * controlled code and data structures.
     */
    try {
        const MagikModelBase *base = model;
        return base->MagikModelBase::get_forward_memory_size();
    } catch (...) {
        printf("mgk_model_get_forward_memory_size: exception while querying size\n");
        return 0;
    }
}

} // extern "C"

