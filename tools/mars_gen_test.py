#!/usr/bin/env python3
"""
Mars Test Model Generator

Generates a simple .mars model file for testing the runtime.
This creates a minimal model with a single Conv2D layer.

STRUCT SIZES (must match C headers):
  mars_header_t: 76 bytes
  mars_tensor_t: 124 bytes
  mars_layer_t: 112 bytes
"""

import struct
import numpy as np
import argparse

# Magic and version
MARS_MAGIC = 0x5352414D  # "MARS" little-endian
MARS_VERSION_MAJOR = 1
MARS_VERSION_MINOR = 0

# Data types
MARS_DTYPE_FLOAT32 = 0
MARS_DTYPE_INT32 = 1
MARS_DTYPE_INT16 = 2
MARS_DTYPE_INT8 = 3
MARS_DTYPE_UINT8 = 4

# Formats
MARS_FORMAT_NHWC = 0
MARS_FORMAT_NCHW = 1

# Layer types
MARS_LAYER_CONV2D = 0
MARS_LAYER_RELU = 5

# Padding
MARS_PAD_SAME = 1

# Activation
MARS_ACT_RELU = 1

# Struct sizes
HEADER_SIZE = 76
TENSOR_SIZE = 124
LAYER_SIZE = 112

def create_header(num_layers, num_tensors, num_inputs, num_outputs,
                  weights_offset, weights_size, input_ids, output_ids):
    """Create 76-byte header (matches C struct)"""
    header = struct.pack('<IHHIIIII',
        MARS_MAGIC,
        MARS_VERSION_MAJOR,
        MARS_VERSION_MINOR,
        0,  # flags
        num_layers,
        num_tensors,
        num_inputs,
        num_outputs
    )
    header += struct.pack('<QQ', weights_offset, weights_size)
    # Input/output tensor IDs (4 each)
    for i in range(4):
        header += struct.pack('<I', input_ids[i] if i < len(input_ids) else 0)
    for i in range(4):
        header += struct.pack('<I', output_ids[i] if i < len(output_ids) else 0)

    assert len(header) == HEADER_SIZE, f"Header size mismatch: {len(header)} vs {HEADER_SIZE}"
    return header

def create_tensor(tid, name, dtype, fmt, shape, data_offset=0, data_size=0,
                  scale=1.0, zero_point=0):
    """Create 124-byte tensor descriptor (matches C struct)"""
    # id: 4 bytes at offset 0
    # name: 60 bytes at offset 4
    # dtype: 4 bytes at offset 64
    # format: 4 bytes at offset 68
    # ndims: 4 bytes at offset 72
    # shape: 24 bytes at offset 76 (6 x int32)
    # data_offset: 8 bytes at offset 100
    # data_size: 8 bytes at offset 108
    # scale: 4 bytes at offset 116
    # zero_point: 4 bytes at offset 120

    name_bytes = name.encode('utf-8')[:59]
    name_bytes = name_bytes + b'\x00' * (60 - len(name_bytes))

    tensor = struct.pack('<I', tid)              # offset 0
    tensor += name_bytes                          # offset 4 (60 bytes)
    tensor += struct.pack('<i', dtype)            # offset 64
    tensor += struct.pack('<i', fmt)              # offset 68
    tensor += struct.pack('<I', len(shape))       # offset 72

    # Shape (6 dims max)
    for i in range(6):
        tensor += struct.pack('<i', shape[i] if i < len(shape) else 0)

    tensor += struct.pack('<Q', data_offset)      # offset 100
    tensor += struct.pack('<Q', data_size)        # offset 108
    tensor += struct.pack('<f', scale)            # offset 116
    tensor += struct.pack('<i', zero_point)       # offset 120

    assert len(tensor) == TENSOR_SIZE, f"Tensor size mismatch: {len(tensor)} vs {TENSOR_SIZE}"
    return tensor

def create_conv_layer(layer_id, input_ids, output_ids, weight_tid, bias_tid,
                      kernel_h, kernel_w, stride=1, activation=MARS_ACT_RELU):
    """Create 112-byte Conv2D layer descriptor (matches C struct)"""
    # id: 4 bytes at offset 0
    # type: 4 bytes at offset 4
    # num_inputs: 4 bytes at offset 8
    # num_outputs: 4 bytes at offset 12
    # input_tensor_ids: 16 bytes at offset 16
    # output_tensor_ids: 16 bytes at offset 32
    # params: 64 bytes at offset 48

    layer = struct.pack('<I', layer_id)           # offset 0
    layer += struct.pack('<i', MARS_LAYER_CONV2D) # offset 4
    layer += struct.pack('<I', 1)                 # num_inputs, offset 8
    layer += struct.pack('<I', 1)                 # num_outputs, offset 12

    # Input tensor IDs (4) at offset 16
    for i in range(4):
        layer += struct.pack('<I', input_ids[i] if i < len(input_ids) else 0xFFFFFFFF)
    # Output tensor IDs (4) at offset 32
    for i in range(4):
        layer += struct.pack('<I', output_ids[i] if i < len(output_ids) else 0xFFFFFFFF)

    # Conv params (64 bytes) at offset 48
    params = struct.pack('<IIIIII',
        kernel_h, kernel_w,     # kernel size
        stride, stride,         # stride
        1, 1                    # dilation
    )
    params += struct.pack('<i', MARS_PAD_SAME)  # padding mode
    params += struct.pack('<IIII', 0, 0, 0, 0)  # explicit padding
    params += struct.pack('<I', 1)              # groups
    params += struct.pack('<i', activation)     # activation
    params += struct.pack('<II', weight_tid, bias_tid)

    layer += params
    # Pad to 112 bytes
    while len(layer) < LAYER_SIZE:
        layer += b'\x00'

    assert len(layer) == LAYER_SIZE, f"Layer size mismatch: {len(layer)} vs {LAYER_SIZE}"
    return layer

def generate_simple_model(output_path, input_h=64, input_w=64, in_ch=3, out_ch=16):
    """Generate a simple single-conv model for testing"""
    print(f"Generating Mars test model: {output_path}")
    print(f"  Input: {input_h}x{input_w}x{in_ch}")
    print(f"  Output channels: {out_ch}")

    # Tensors:
    # 0: input (NHWC)
    # 1: conv weight (OHWI)
    # 2: conv bias
    # 3: output (NHWC)

    kernel_h, kernel_w = 3, 3

    # Create random weights
    weight_shape = (out_ch, kernel_h, kernel_w, in_ch)
    weights = np.random.randint(-128, 127, weight_shape, dtype=np.int8)
    bias = np.zeros(out_ch, dtype=np.int32)

    weight_bytes = weights.tobytes()
    bias_bytes = bias.tobytes()
    total_weights = weight_bytes + bias_bytes

    # Calculate output size (same padding, stride 1)
    out_h, out_w = input_h, input_w

    # Header offsets using correct struct sizes
    weights_offset = HEADER_SIZE + (4 * TENSOR_SIZE) + (1 * LAYER_SIZE)
    weights_offset = ((weights_offset + 63) // 64) * 64  # Align to 64

    print(f"  Header: {HEADER_SIZE} bytes")
    print(f"  Tensors: 4 x {TENSOR_SIZE} = {4 * TENSOR_SIZE} bytes")
    print(f"  Layers: 1 x {LAYER_SIZE} = {LAYER_SIZE} bytes")
    print(f"  Weights offset: {weights_offset}")

    # Create tensors
    tensors = []
    tensors.append(create_tensor(0, "input", MARS_DTYPE_INT8, MARS_FORMAT_NHWC,
                                 [1, input_h, input_w, in_ch]))
    tensors.append(create_tensor(1, "conv1_weight", MARS_DTYPE_INT8, MARS_FORMAT_NHWC,
                                 list(weight_shape), data_offset=0,
                                 data_size=len(weight_bytes)))
    tensors.append(create_tensor(2, "conv1_bias", MARS_DTYPE_INT32, MARS_FORMAT_NHWC,
                                 [out_ch], data_offset=len(weight_bytes),
                                 data_size=len(bias_bytes)))
    tensors.append(create_tensor(3, "output", MARS_DTYPE_INT8, MARS_FORMAT_NHWC,
                                 [1, out_h, out_w, out_ch]))

    # Create layer
    layers = []
    layers.append(create_conv_layer(0, [0], [3], 1, 2, kernel_h, kernel_w))

    # Create header
    header = create_header(
        num_layers=1,
        num_tensors=4,
        num_inputs=1,
        num_outputs=1,
        weights_offset=weights_offset,
        weights_size=len(total_weights),
        input_ids=[0],
        output_ids=[3]
    )

    # Write file
    with open(output_path, 'wb') as f:
        f.write(header)
        for t in tensors:
            f.write(t)
        for l in layers:
            f.write(l)
        # Pad to weights offset
        current = f.tell()
        if current < weights_offset:
            f.write(b'\x00' * (weights_offset - current))
        f.write(total_weights)

    print(f"  Wrote {output_path}: {weights_offset + len(total_weights)} bytes")
    print(f"  Weights: {len(total_weights)} bytes at offset {weights_offset}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Mars test model')
    parser.add_argument('-o', '--output', default='test_model.mars',
                        help='Output file path')
    parser.add_argument('--height', type=int, default=64, help='Input height')
    parser.add_argument('--width', type=int, default=64, help='Input width')
    parser.add_argument('--channels', type=int, default=3, help='Input channels')
    parser.add_argument('--out-channels', type=int, default=16, help='Output channels')

    args = parser.parse_args()
    generate_simple_model(args.output, args.height, args.width,
                          args.channels, args.out_channels)

