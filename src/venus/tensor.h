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

/* Internal tensor representation
 * OEM size: 0x44 (68 bytes)
 * Layout reconstructed from libmert.so / .mgk usage:
 *   - [0x00] int32_t* dims_begin
 *   - [0x04] int32_t* dims_end
 *   - [0x08] void*    reserved0
 *   - [0x0C] void*    data        (was MBuffer* in OEM; we treat as raw data ptr)
 *   - [0x10] void*    reserved1   (shared_ptr control block in OEM)
 *   - [0x14] int32_t  ref_count
 *   - [0x18] void*    dims_meta0
 *   - [0x1C] void*    dims_meta1
 *   - [0x20] void*    dims_meta2
 *   - [0x24] uint32_t align       (64 for NNA)
 *   - [0x28] DataType dtype
 *   - [0x2C] TensorFormat format
 *   - [0x30] uint32_t bytes       (logical size in bytes)
 *   - [0x34] uint32_t owns_data   (non-zero if this TensorX owns data)
 *   - [0x38] int32_t  reserved3
 *   - [0x3C] uint32_t data_offset (used by vdata<char>)
 *   - [0x40] int32_t  reserved4
 */
struct TensorX {
    int32_t *dims_begin;   /* 0x00 */
    int32_t *dims_end;     /* 0x04 */
    void    *reserved0;    /* 0x08 */
    void    *data;         /* 0x0C */
    void    *reserved1;    /* 0x10 */
    int32_t  ref_count;    /* 0x14 */
    void    *dims_meta0;   /* 0x18 */
    void    *dims_meta1;   /* 0x1C */
    void    *dims_meta2;   /* 0x20 */
    uint32_t align;        /* 0x24 */
    DataType dtype;        /* 0x28 */
    TensorFormat format;   /* 0x2C */
    uint32_t bytes;        /* 0x30 */
    uint32_t owns_data;    /* 0x34 */
    int32_t  reserved3;    /* 0x38 */
    uint32_t data_offset;  /* 0x3C */
    int32_t  reserved4;    /* 0x40 */

    TensorX();  /* Constructor - implemented in tensor.cpp */
    ~TensorX();  /* Destructor */
    TensorX(const TensorX &other);  /* Copy constructor */
    TensorX& operator=(const TensorX &other);  /* Copy assignment */

    /* Step function required by .mgk models */
    int step(int dim) const;
    int step(int dim);  /* Non-const version also needed */

    /* Get bytes size */
    size_t get_bytes_size() const;

    /* Helpers for our Tensor wrapper */
    void set_shape(const shape_t &shape);
    shape_t get_shape() const;
};

static_assert(sizeof(TensorX) == 0x44, "TensorX size must match OEM (0x44 bytes)");

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

