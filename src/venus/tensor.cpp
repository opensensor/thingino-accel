/*
 * thingino-accel - Venus Tensor Implementation
 */

#include "tensor.h"
#include "../../include/nna.h"
#include "../../include/nna_memory.h"
#include <cstring>
#include <cstdio>

namespace magik {
namespace venus {

/* TensorX constructor */
TensorX::TensorX()
    : dims_begin(nullptr),
      dims_end(nullptr),
      reserved0(nullptr),
      data(nullptr),
      reserved1(nullptr),
      ref_count(1),
      dims_meta0(nullptr),
      dims_meta1(nullptr),
      dims_meta2(nullptr),
      align(64),
      dtype(DataType::INT8),
      format(TensorFormat::NHWC),
      bytes(0),
      owns_data(0),
      reserved3(0),
      data_offset(0),
      reserved4(0) {
    printf("[VENUS] TensorX::TensorX default ctor this=%p (align=%u)\n", (void*)this, align);
    fflush(stdout);
}

/* TensorX destructor */
TensorX::~TensorX() {
    printf("[VENUS] TensorX::~TensorX(this=%p)\n", (void*)this);
    fflush(stdout);
    /* Free shape storage if we allocated it */
    if (dims_begin) {
        delete[] dims_begin;
        dims_begin = nullptr;
        dims_end = nullptr;
    }
    /* Do NOT free data here - it's managed by Tensor wrapper or external owner. */
}

/* TensorX copy constructor */
TensorX::TensorX(const TensorX &other)
    : dims_begin(nullptr),
      dims_end(nullptr),
      reserved0(other.reserved0),
      data(other.data),
      reserved1(other.reserved1),
      ref_count(1),
      dims_meta0(other.dims_meta0),
      dims_meta1(other.dims_meta1),
      dims_meta2(other.dims_meta2),
      align(other.align),
      dtype(other.dtype),
      format(other.format),
      bytes(other.bytes),
      owns_data(0),  /* Don't take ownership of copied data */
      reserved3(other.reserved3),
      data_offset(other.data_offset),
      reserved4(other.reserved4) {
    if (other.dims_begin && other.dims_end) {
        int ndims = static_cast<int>(other.dims_end - other.dims_begin);
        dims_begin = new int32_t[ndims];
        for (int i = 0; i < ndims; ++i) {
            dims_begin[i] = other.dims_begin[i];
        }
        dims_end = dims_begin + ndims;
    }
    printf("[VENUS] TensorX::TensorX(copy ctor, this=%p, other=%p, ndims=%ld)\n",
           (void*)this, (void*)&other,
           (long)(dims_begin && dims_end ? dims_end - dims_begin : 0));
    fflush(stdout);
}

/* TensorX copy assignment */
TensorX& TensorX::operator=(const TensorX &other) {
    printf("[VENUS] TensorX::operator=(this=%p, other=%p)\n",
           (void*)this, (void*)&other);
    fflush(stdout);

    if (this != &other) {
        if (dims_begin) {
            delete[] dims_begin;
            dims_begin = nullptr;
            dims_end = nullptr;
        }
        dims_begin  = nullptr;
        dims_end    = nullptr;
        reserved0   = other.reserved0;
        data        = other.data;
        reserved1   = other.reserved1;
        /* ref_count is not shared between TensorX instances */
        align       = other.align;
        dtype       = other.dtype;
        format      = other.format;
        bytes       = other.bytes;
        owns_data   = 0;  /* Don't take ownership */
        dims_meta0  = other.dims_meta0;
        dims_meta1  = other.dims_meta1;
        dims_meta2  = other.dims_meta2;
        reserved3   = other.reserved3;
        data_offset = other.data_offset;
        reserved4   = other.reserved4;

        if (other.dims_begin && other.dims_end) {
            int ndims = static_cast<int>(other.dims_end - other.dims_begin);
            dims_begin = new int32_t[ndims];
            for (int i = 0; i < ndims; ++i) {
                dims_begin[i] = other.dims_begin[i];
            }
            dims_end = dims_begin + ndims;
        }
    }
    return *this;
}

/* TensorX step function (const) */
int TensorX::step(int dim) const {
    int ndims = (dims_begin && dims_end) ? static_cast<int>(dims_end - dims_begin) : 0;
    printf("[VENUS] TensorX::step const(this=%p, dim=%d, ndims=%d)\n",
           (void*)this, dim, ndims);
    fflush(stdout);
    if (dim < 0 || dim >= ndims) {
        return 0;
    }

    int step_val = 1;
    for (int i = dim + 1; i < ndims; ++i) {
        step_val *= dims_begin[i];
    }
    return step_val;
}

/* TensorX step function (non-const) */
int TensorX::step(int dim) {
    int ndims = (dims_begin && dims_end) ? static_cast<int>(dims_end - dims_begin) : 0;
    printf("[VENUS] TensorX::step(this=%p, dim=%d, ndims=%d)\n",
           (void*)this, dim, ndims);
    fflush(stdout);
    if (dim < 0 || dim >= ndims) {
        return 0;
    }

    int step_val = 1;
    for (int i = dim + 1; i < ndims; ++i) {
        step_val *= dims_begin[i];
    }
    return step_val;
}

/* TensorX get_bytes_size */
size_t TensorX::get_bytes_size() const {
    printf("[VENUS] TensorX::get_bytes_size(this=%p) -> %u\n",
           (void*)this, bytes);
    fflush(stdout);
    return bytes;
}


/* TensorX helper: set shape from std::vector */
void TensorX::set_shape(const shape_t &shape) {
    long old_ndims = (dims_begin && dims_end) ? (dims_end - dims_begin) : 0;
    printf("[VENUS] TensorX::set_shape(this=%p, old_ndims=%ld, new_ndims=%zu)\n",
           (void*)this, old_ndims, shape.size());
    fflush(stdout);

    if (dims_begin) {
        delete[] dims_begin;
        dims_begin = nullptr;
        dims_end = nullptr;
        reserved0 = nullptr;
    }

    if (shape.empty()) {
        return;
    }

    long raw_ndims = static_cast<long>(shape.size());
    const long kMaxDims = 8;
    if (raw_ndims <= 0) {
        raw_ndims = 1;
    } else if (raw_ndims > kMaxDims) {
        printf("[VENUS] TensorX::set_shape: clamping ndims=%ld to %ld\n",
               raw_ndims, kMaxDims);
        fflush(stdout);
        raw_ndims = kMaxDims;
    }

    int ndims = static_cast<int>(raw_ndims);
    dims_begin = new int32_t[ndims];
    for (int i = 0; i < ndims; ++i) {
        dims_begin[i] = shape[i];
    }
    dims_end = dims_begin + ndims;
    reserved0 = dims_end;  /* capacity pointer for std::vector-style layout */

    printf("[VENUS] TensorX::set_shape: dims_begin=%p dims_end=%p cap=%p [",
           (void*)dims_begin, (void*)dims_end, reserved0);
    for (int i = 0; i < ndims; ++i) {
        printf("%d%s", dims_begin[i], (i + 1 < ndims) ? "," : "");
    }
    printf("]\n");
    fflush(stdout);
}

/* TensorX helper: export shape to std::vector, with safety clamps */
shape_t TensorX::get_shape() const {
    shape_t result;

    /* Log raw dims pointers and computed length before constructing vector. */
    if (!dims_begin || !dims_end) {
        fprintf(stderr,
                "[VENUS] TensorX::get_shape: dims_begin=%p dims_end=%p -> empty shape\n",
                (void*)dims_begin, (void*)dims_end);
        fflush(stderr);
        return result;
    }

    long raw_ndims = (long)(dims_end - dims_begin);
    fprintf(stderr,
            "[VENUS] TensorX::get_shape: dims_begin=%p dims_end=%p raw_ndims=%ld\n",
            (void*)dims_begin, (void*)dims_end, raw_ndims);
    fflush(stderr);

    /* Clamp to a sane range to avoid pathological allocations. */
    const long kMaxDims = 8;
    if (raw_ndims <= 0 || raw_ndims > kMaxDims) {
        fprintf(stderr,
                "[VENUS] TensorX::get_shape: suspicious ndims=%ld, returning empty shape\n",
                raw_ndims);
        fflush(stderr);
        return result;
    }

    int ndims = static_cast<int>(raw_ndims);
    result.assign(dims_begin, dims_begin + ndims);
    return result;
}

/* Constructor with shape */
Tensor::Tensor(shape_t s, TensorFormat fmt) {
    init_tensorx(s, fmt);
}

/* Constructor with initializer list */
Tensor::Tensor(std::initializer_list<int32_t> s, TensorFormat fmt) {
    shape_t shape(s);
    init_tensorx(shape, fmt);
}

/* Constructor with existing data */
Tensor::Tensor(void *data, size_t bytes_size, TensorFormat fmt) {
    tensorx = new TensorX();
    ref_count = new int(1);

    tensorx->data = data;
    tensorx->bytes = bytes_size;
    tensorx->owns_data = false;  /* User owns the data */
    tensorx->format = fmt;
    tensorx->dtype = DataType::INT8;  /* Default */
}

/* Copy constructor */
Tensor::Tensor(const Tensor &t) {
    tensorx = t.tensorx;
    ref_count = t.ref_count;
    (*ref_count)++;
}

/* Constructor from internal TensorX */
Tensor::Tensor(void *tsx) {
    tensorx = static_cast<TensorX*>(tsx);
    ref_count = &(tensorx->ref_count);
    (*ref_count)++;
}

Tensor::Tensor(const void *tsx) {
    tensorx = const_cast<TensorX*>(static_cast<const TensorX*>(tsx));
    ref_count = &(tensorx->ref_count);
    (*ref_count)++;
}

/* Destructor
 *
 * Important: when Tensor wraps an externally managed TensorX
 * (constructed via Tensor(void *tsx) / Tensor(const void *tsx)),
 * ref_count points directly into TensorX::ref_count. In that case we
 * must NOT delete tensorx or the ref_count storage itself; ownership
 * stays with the MagikModelBase / OEM runtime.
 */
Tensor::~Tensor() {
    if (!ref_count) {
        return;
    }

    // External TensorX case: ref_count is aliased to tensorx->ref_count.
    if (tensorx && ref_count == &(tensorx->ref_count)) {
        int &rc = *ref_count;
        if (rc > 0) {
            --rc;
        }
        // Do not delete tensorx or ref_count; they are owned elsewhere.
        return;
    }

    // Owned TensorX case: ref_count was allocated separately.
    if (--(*ref_count) == 0) {
        if (tensorx) {
            if (tensorx->owns_data && tensorx->data) {
                nna_free(tensorx->data);
            }
            delete tensorx;
        }
        delete ref_count;
    }
}

/* Initialize TensorX */
void Tensor::init_tensorx(const shape_t &s, TensorFormat fmt) {
    tensorx = new TensorX();
    ref_count = new int(1);

    tensorx->set_shape(s);
    tensorx->format = fmt;
    tensorx->dtype = DataType::INT8;  /* Default for NNA */
    tensorx->align = 64;

    /* Calculate size (INT8 = 1 byte per element) */
    size_t total = 1;
    for (auto dim : s) {
        total *= static_cast<size_t>(dim);
    }
    tensorx->bytes = static_cast<uint32_t>(total);

    /* Allocate memory using NNA allocator */
    tensorx->data = nna_malloc(tensorx->bytes);
    tensorx->owns_data = tensorx->data ? 1u : 0u;

    if (!tensorx->data) {
        fprintf(stderr, "Tensor: Failed to allocate %zu bytes\n", static_cast<size_t>(tensorx->bytes));
    }
}

/* Get shape with logging and safety clamp */
shape_t Tensor::shape() const {
    if (!tensorx) {
        fprintf(stderr, "[VENUS] Tensor::shape: tensorx=NULL, returning empty shape\n");
        fflush(stderr);
        return shape_t();
    }

    shape_t s = tensorx->get_shape();
    fprintf(stderr,
            "[VENUS] Tensor::shape: tensorx=%p -> ndims=%zu\n",
            (void*)tensorx, s.size());
    fflush(stderr);

    /* Extra safety: cap dimensions to a sane maximum. */
    const size_t kMaxDims = 8;
    if (s.size() > kMaxDims) {
        fprintf(stderr,
                "[VENUS] Tensor::shape: ndims=%zu > %zu, truncating\n",
                s.size(), kMaxDims);
        fflush(stderr);
        s.resize(kMaxDims);
    }

    return s;
}

/* Get data type */
DataType Tensor::data_type() const {
    return tensorx->dtype;
}

/* Reshape */
void Tensor::reshape(shape_t &s) const {
    tensorx->set_shape(s);
}

void Tensor::reshape(std::initializer_list<int32_t> s) const {
    tensorx->set_shape(shape_t(s));
}

/* Free data */
void Tensor::free_data() const {
    if (tensorx->owns_data && tensorx->data) {
        nna_free(tensorx->data);
        tensorx->data = nullptr;
        tensorx->owns_data = false;
    }
}

/* Set data */
int Tensor::set_data(void *data, size_t bytes_size) {
    if (tensorx->owns_data && tensorx->data) {
        nna_free(tensorx->data);
    }
    tensorx->data = data;
    tensorx->bytes = bytes_size;
    tensorx->owns_data = false;
    return 0;
}

/* Get internal TensorX */
void *Tensor::get_tsx() const {
    return tensorx;
}

/* Get step (stride) for dimension */
int Tensor::step(int dim) const {
    return tensorx ? tensorx->step(dim) : 0;
}

} // namespace venus
} // namespace magik

