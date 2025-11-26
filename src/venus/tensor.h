/*
 * thingino-accel - Venus Tensor Implementation
 */

#ifndef THINGINO_ACCEL_VENUS_TENSOR_H
#define THINGINO_ACCEL_VENUS_TENSOR_H

#include "venus_types.h"
#include <initializer_list>
#include <memory>

namespace magik {
namespace venus {

/* Internal tensor representation */
struct TensorX {
    shape_t shape;
    DataType dtype;
    TensorFormat format;
    void *data;
    size_t bytes;
    bool owns_data;
    int ref_count;

    TensorX() : dtype(DataType::INT8), format(TensorFormat::NHWC),
                data(nullptr), bytes(0), owns_data(false), ref_count(1) {}

    /* Step function required by .mgk models */
    int step(int dim) const;
    int step(int dim);  /* Non-const version also needed */
};

/* Tensor class matching Venus API */
class Tensor {
public:
    Tensor(shape_t shape, TensorFormat fmt = TensorFormat::NHWC);
    Tensor(std::initializer_list<int32_t> shape, TensorFormat fmt = TensorFormat::NHWC);
    Tensor(void *data, size_t bytes_size, TensorFormat fmt = TensorFormat::NHWC);
    Tensor(const Tensor &t);
    Tensor(void *tsx);       /* for internal */
    Tensor(const void *tsx); /* for internal */
    virtual ~Tensor();

    shape_t shape() const;
    DataType data_type() const;
    void reshape(shape_t &shape) const;
    void reshape(std::initializer_list<int32_t> shape) const;
    
    template <typename T>
    const T *data() const {
        return static_cast<const T*>(tensorx->data);
    }
    
    template <typename T>
    T *mudata() const {
        return static_cast<T*>(tensorx->data);
    }
    
    void free_data() const;
    int set_data(void *data, size_t bytes_size);
    void *get_tsx() const; /* for internal */
    int step(int dim) const;

private:
    TensorX *tensorx;
    int *ref_count;
    
    void init_tensorx(const shape_t &s, TensorFormat fmt);
};

} // namespace venus
} // namespace magik

#endif /* THINGINO_ACCEL_VENUS_TENSOR_H */

