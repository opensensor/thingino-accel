#!/usr/bin/env python3
"""
ONNX to Mars JSON converter.
Extracts model structure and weights into a simple format for the Rust compiler.
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import onnx
import numpy as np


def extract_tensor(tensor_proto):
    """Extract tensor data from TensorProto."""
    name = tensor_proto.name
    dims = list(tensor_proto.dims)
    dtype = tensor_proto.data_type  # 1=FLOAT, 6=INT32, 7=INT64, etc.
    
    # Get raw data
    if tensor_proto.raw_data:
        data = tensor_proto.raw_data
    elif tensor_proto.float_data:
        data = struct.pack(f'{len(tensor_proto.float_data)}f', *tensor_proto.float_data)
    elif tensor_proto.int32_data:
        data = struct.pack(f'{len(tensor_proto.int32_data)}i', *tensor_proto.int32_data)
    elif tensor_proto.int64_data:
        data = struct.pack(f'{len(tensor_proto.int64_data)}q', *tensor_proto.int64_data)
    else:
        data = b''
    
    return {
        'name': name,
        'dims': dims,
        'dtype': dtype,
        'data_offset': 0,  # Will be filled in later
        'data_len': len(data),
    }, data


def extract_node(node_proto):
    """Extract node info from NodeProto."""
    attrs = {}
    for attr in node_proto.attribute:
        if attr.type == onnx.AttributeProto.INT:
            attrs[attr.name] = attr.i
        elif attr.type == onnx.AttributeProto.INTS:
            attrs[attr.name] = list(attr.ints)
        elif attr.type == onnx.AttributeProto.FLOAT:
            attrs[attr.name] = attr.f
        elif attr.type == onnx.AttributeProto.FLOATS:
            attrs[attr.name] = list(attr.floats)
        elif attr.type == onnx.AttributeProto.STRING:
            attrs[attr.name] = attr.s.decode('utf-8')
        elif attr.type == onnx.AttributeProto.STRINGS:
            attrs[attr.name] = [s.decode('utf-8') for s in attr.strings]
    
    return {
        'op_type': node_proto.op_type,
        'name': node_proto.name,
        'inputs': list(node_proto.input),
        'outputs': list(node_proto.output),
        'attributes': attrs,
    }


def extract_value_info(vi):
    """Extract shape from ValueInfoProto."""
    tensor_type = vi.type.tensor_type
    shape = []
    if tensor_type.HasField('shape'):
        for dim in tensor_type.shape.dim:
            if dim.HasField('dim_value'):
                shape.append(dim.dim_value)
            else:
                shape.append(-1)  # Dynamic dimension
    return {
        'name': vi.name,
        'dtype': tensor_type.elem_type,
        'shape': shape,
    }


def convert_onnx(onnx_path: Path, output_path: Path):
    """Convert ONNX model to Mars JSON + binary format."""
    model = onnx.load(str(onnx_path))
    graph = model.graph
    
    # Extract initializers (weights)
    initializers = []
    weight_data = b''
    for init in graph.initializer:
        info, data = extract_tensor(init)
        info['data_offset'] = len(weight_data)
        weight_data += data
        initializers.append(info)
    
    # Extract nodes
    nodes = [extract_node(n) for n in graph.node]
    
    # Extract inputs (excluding initializers)
    init_names = {i.name for i in graph.initializer}
    inputs = [extract_value_info(vi) for vi in graph.input if vi.name not in init_names]
    
    # Extract outputs
    outputs = [extract_value_info(vi) for vi in graph.output]
    
    # Build manifest
    manifest = {
        'opset': model.opset_import[0].version if model.opset_import else 0,
        'inputs': inputs,
        'outputs': outputs,
        'nodes': nodes,
        'initializers': initializers,
        'weight_bytes': len(weight_data),
    }
    
    # Write JSON manifest
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Write binary weights
    bin_path = output_path.with_suffix('.bin')
    with open(bin_path, 'wb') as f:
        f.write(weight_data)
    
    print(f"Wrote {json_path} ({json_path.stat().st_size} bytes)")
    print(f"Wrote {bin_path} ({len(weight_data)} bytes)")
    print(f"  {len(initializers)} initializers, {len(nodes)} nodes")


def main():
    parser = argparse.ArgumentParser(description='Convert ONNX to Mars JSON format')
    parser.add_argument('input', type=Path, help='Input ONNX file')
    parser.add_argument('-o', '--output', type=Path, help='Output base path (default: input with .mars suffix)')
    args = parser.parse_args()
    
    output = args.output or args.input.with_suffix('.mars')
    convert_onnx(args.input, output)


if __name__ == '__main__':
    main()

