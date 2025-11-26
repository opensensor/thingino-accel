/*
 * thingino-accel - Venus Utility Functions
 */

#include "venus_types.h"
#include <string>
#include <cstdio>

namespace magik {
namespace venus {
namespace utils {

/* Data type utilities */
int data_type2bits(DataType dtype) {
    switch (dtype) {
        case DataType::FP32:
        case DataType::UINT32:
        case DataType::INT32:
            return 32;
        case DataType::UINT16:
        case DataType::INT16:
            return 16;
        case DataType::UINT8:
        case DataType::INT8:
            return 8;
        case DataType::UINT4B:
            return 4;
        case DataType::UINT2B:
            return 2;
        case DataType::BOOL:
            return 1;
        default:
            return 0;
    }
}

int data_type2validbits(DataType dtype) {
    /* For most types, valid bits == total bits */
    return data_type2bits(dtype);
}

std::string data_type2string(DataType dtype) {
    switch (dtype) {
        case DataType::FP32: return "FP32";
        case DataType::UINT32: return "UINT32";
        case DataType::INT32: return "INT32";
        case DataType::UINT16: return "UINT16";
        case DataType::INT16: return "INT16";
        case DataType::UINT8: return "UINT8";
        case DataType::INT8: return "INT8";
        case DataType::UINT4B: return "UINT4B";
        case DataType::UINT2B: return "UINT2B";
        case DataType::BOOL: return "BOOL";
        default: return "UNKNOWN";
    }
}

DataType string2data_type(const std::string &str) {
    if (str == "FP32") return DataType::FP32;
    if (str == "UINT32") return DataType::UINT32;
    if (str == "INT32") return DataType::INT32;
    if (str == "UINT16") return DataType::UINT16;
    if (str == "INT16") return DataType::INT16;
    if (str == "UINT8") return DataType::UINT8;
    if (str == "INT8") return DataType::INT8;
    if (str == "UINT4B") return DataType::UINT4B;
    if (str == "UINT2B") return DataType::UINT2B;
    if (str == "BOOL") return DataType::BOOL;
    return DataType::NONE;
}

/* Data format utilities */
std::string data_format2string(TensorFormat fmt) {
    switch (fmt) {
        case TensorFormat::NHWC: return "NHWC";
        case TensorFormat::NV12: return "NV12";
        default: return "UNKNOWN";
    }
}

TensorFormat string2data_format(const std::string &str) {
    if (str == "NHWC") return TensorFormat::NHWC;
    if (str == "NV12") return TensorFormat::NV12;
    return TensorFormat::NHWC;  /* Default */
}

/* Channel layout utilities */
ChannelLayout string2channel_layout(const std::string &str) {
    if (str == "RGB") return ChannelLayout::RGB;
    if (str == "BGR") return ChannelLayout::BGRA;  /* Use BGRA as closest match */
    if (str == "GRAY") return ChannelLayout::GRAY;
    return ChannelLayout::NONE;
}

/* Template specializations for type2data_type */
template<> DataType type2data_type<float>() { return DataType::FP32; }
template<> DataType type2data_type<uint32_t>() { return DataType::UINT32; }
template<> DataType type2data_type<int32_t>() { return DataType::INT32; }
template<> DataType type2data_type<uint16_t>() { return DataType::UINT16; }
template<> DataType type2data_type<int16_t>() { return DataType::INT16; }
template<> DataType type2data_type<uint8_t>() { return DataType::UINT8; }
template<> DataType type2data_type<int8_t>() { return DataType::INT8; }
template<> DataType type2data_type<bool>() { return DataType::BOOL; }

} // namespace utils
} // namespace venus
} // namespace magik

