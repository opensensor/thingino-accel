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

/* TensorX step function (const) */
int TensorX::step(int dim) const {
    if (dim < 0 || dim >= (int)shape.size()) {
        return 0;
    }

    int step = 1;
    for (size_t i = dim + 1; i < shape.size(); i++) {
        step *= shape[i];
    }
    return step;
}

/* TensorX step function (non-const) */
int TensorX::step(int dim) {
    if (dim < 0 || dim >= (int)shape.size()) {
        return 0;
    }

    int step = 1;
    for (size_t i = dim + 1; i < shape.size(); i++) {
        step *= shape[i];
    }
    return step;
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

/* Destructor */
Tensor::~Tensor() {
    if (ref_count && --(*ref_count) == 0) {
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
    
    tensorx->shape = s;
    tensorx->format = fmt;
    tensorx->dtype = DataType::INT8;  /* Default for NNA */
    
    /* Calculate size */
    size_t total = 1;
    for (auto dim : s) {
        total *= dim;
    }
    tensorx->bytes = total;  /* INT8 = 1 byte per element */
    
    /* Allocate memory using NNA allocator */
    tensorx->data = nna_malloc(tensorx->bytes);
    tensorx->owns_data = true;
    
    if (!tensorx->data) {
        fprintf(stderr, "Tensor: Failed to allocate %zu bytes\n", tensorx->bytes);
    }
}

/* Get shape */
shape_t Tensor::shape() const {
    return tensorx->shape;
}

/* Get data type */
DataType Tensor::data_type() const {
    return tensorx->dtype;
}

/* Reshape */
void Tensor::reshape(shape_t &s) const {
    tensorx->shape = s;
}

void Tensor::reshape(std::initializer_list<int32_t> s) const {
    tensorx->shape = shape_t(s);
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
    if (dim < 0 || dim >= (int)tensorx->shape.size()) {
        return 0;
    }
    
    int step = 1;
    for (size_t i = dim + 1; i < tensorx->shape.size(); i++) {
        step *= tensorx->shape[i];
    }
    return step;
}

} // namespace venus
} // namespace magik

