/*
 * thingino-accel - Venus Library Implementation
 * Core type definitions matching Ingenic Venus API
 */

#ifndef THINGINO_ACCEL_VENUS_TYPES_H
#define THINGINO_ACCEL_VENUS_TYPES_H

#include <stdint.h>
#include <vector>
#include <string>

namespace magik {
namespace venus {

/* Tensor format */
enum class TensorFormat : int {
    NHWC = 1,  /* Batch, Height, Width, Channels */
    NV12 = 5,  /* YUV 4:2:0 format */
};

/* DataFormat - separate enum for backward compatibility with old .mgk models */
enum class DataFormat : int {
    NHWC = 1,
    NV12 = 5,
};

/* Data type */
enum class DataType : int {
    NONE = -1,
    AUTO = 0,
    FP32 = 1,
    UINT32 = 2,
    INT32 = 3,
    UINT16 = 4,
    INT16 = 5,
    UINT8 = 6,
    INT8 = 7,
    UINT4B = 10,
    UINT2B = 11,
    BOOL = 18,
};

/* Memory sharing mode */
enum class ShareMemoryMode : int {
    DEFAULT = 0,
    SHARE_ONE_THREAD = 1,
    SET_FROM_EXTERNAL = 2,
    ALL_SEPARABLE_MEM = 3,
    SMART_REUSE_MEM = 4,
};

/* Channel layout */
enum class ChannelLayout : int {
    NONE = -1,
    NV12 = 0,
    BGRA = 1,
    RGBA = 2,
    ARGB = 3,
    RGB = 4,
    GRAY = 5,
    FP = 6
};

/* Shape type */
using shape_t = std::vector<int32_t>;

/* Opaque data attribute type used by OEM Magik kernels.
 * We only need the type name for interop with .mgk functions such as
 * prepare_init_attr; the actual layout lives inside the OEM binaries.
 * Do not rely on any particular fields or size here.
 */
struct DataAttribute {
    // Intentionally empty placeholder; we never access fields directly.
};

/* Opaque kernel parameter type used by OEM Magik kernels. */
namespace kernel {
    struct KernelParam {
        // Placeholder; actual layout lives inside OEM binaries.
    };
}

/* Opaque operator configuration type for OEM Magik kernels. */
struct OpConfig {
    // Placeholder; used only via pointers.
};

/* Generic return value type used by Magik/Venus helpers and kernel
 * parameter init routines. OEM code generally treats this as an integer
 * status code.
 */
struct ReturnValue {
    int code;
    ReturnValue(int c = 0) : code(c) {}
};


/* Utility functions namespace */
namespace utils {
    /* Type conversion utilities */
    int data_type2bits(DataType dtype);
    int data_type2validbits(DataType dtype);
    std::string data_type2string(DataType dtype);
    DataType string2data_type(const std::string &str);
    DataType string2data_type(std::string str);  /* By-value overload */

    std::string data_format2string(TensorFormat fmt);
    TensorFormat string2data_format(const std::string &str);
    TensorFormat string2data_format(std::string str);  /* By-value overload */

    ChannelLayout string2channel_layout(const std::string &str);
    ChannelLayout string2channel_layout(std::string str);  /* By-value overload */

    /* Template for C++ type to DataType conversion */
    template<typename T> DataType type2data_type();
} // namespace utils

} // namespace venus
} // namespace magik

#endif /* THINGINO_ACCEL_VENUS_TYPES_H */

