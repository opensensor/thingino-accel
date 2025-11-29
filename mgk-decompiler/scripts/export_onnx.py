#!/usr/bin/env python3
"""
Export MGK model structure to ONNX format for visualization.

This script reads the JSON output from mgk-decompiler and creates
an ONNX model representation for visualization with tools like Netron.

Note: The exported ONNX model is for VISUALIZATION ONLY - it represents
the model structure but may not be directly runnable due to custom ops.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import onnx
    from onnx import helper, TensorProto
except ImportError:
    print("Error: onnx package not found. Install with: pip install onnx")
    sys.exit(1)


def create_tensor_value_info(name: str, dtype: str = None, format: str = None):
    """Create ONNX tensor value info from MGK tensor info."""
    # Default to float for unknown types
    onnx_dtype = TensorProto.FLOAT
    if dtype:
        dtype_map = {
            'UINT8': TensorProto.UINT8,
            'INT8': TensorProto.INT8,
            'FP32': TensorProto.FLOAT,
            'FP16': TensorProto.FLOAT16,
            'INT32': TensorProto.INT32,
        }
        onnx_dtype = dtype_map.get(dtype.upper(), TensorProto.FLOAT)
    
    # Default shape - use dynamic dimensions
    shape = ['batch', 'seq', 'features']  # Common for RNN models
    
    return helper.make_tensor_value_info(name, onnx_dtype, shape)


def create_layer_node(layer: dict, idx: int) -> onnx.NodeProto:
    """Create ONNX node from MGK layer config."""
    layer_type = layer.get('layer_type', 'Unknown')
    name = layer.get('name', f'layer_{idx}')
    
    # Map MGK layer types to ONNX operators
    op_type_map = {
        'GRU': 'GRU',
        'Conv': 'Conv',
        'Pool': 'MaxPool',  # Could also be AvgPool
        'Concat': 'Concat',
        'Add': 'Add',
        'Reshape': 'Reshape',
        'Permute': 'Transpose',
        'Slice': 'Slice',
        'Upsample': 'Resize',
        'FormatConvert': 'Identity',  # No direct equivalent
        'DeQuantize': 'DequantizeLinear',
        'BatchNorm': 'BatchNormalization',
        'Feature': 'Identity',  # Passthrough
        'SqueezeUnsqueeze': 'Squeeze',
        'GenerateBox': 'Identity',  # Custom op
    }
    
    op_type = op_type_map.get(layer_type, 'Identity')
    
    # Get input/output tensors
    inputs = [t['name'] for t in layer.get('input_tensors', [])]
    outputs = [t['name'] for t in layer.get('output_tensors', [])]
    
    # Default inputs/outputs if not specified
    if not inputs:
        inputs = [f'{name}_input']
    if not outputs:
        outputs = [name]
    
    return helper.make_node(
        op_type,
        inputs=inputs,
        outputs=outputs,
        name=name,
    )


def export_to_onnx(json_path: Path, output_path: Path):
    """Export MGK model JSON to ONNX format."""
    with open(json_path) as f:
        data = json.load(f)

    model_name = data.get('model_name', 'mgk_model')
    model_config = data.get('model_config', {})

    if not model_config:
        print("Error: No model_config found in JSON. Run mgk-decompiler with -o first.")
        return False

    layers = model_config.get('layers', [])

    # Build a sequential chain of Identity nodes for now
    # This makes the model runnable even if the original graph is complex
    nodes = []
    prev_output = 'input'

    for idx, layer in enumerate(layers):
        layer_name = layer.get('name', f'layer_{idx}')
        layer_type = layer.get('layer_type', 'Unknown')

        # Use layer name as output
        output_name = f'{layer_name}_out'

        # Create Identity node (simplest runnable op)
        node = helper.make_node(
            'Identity',
            inputs=[prev_output],
            outputs=[output_name],
            name=layer_name,
        )
        # Store original layer type as an attribute for visualization
        node.doc_string = f"Original type: {layer_type}"
        nodes.append(node)
        prev_output = output_name

    # Add final output node
    final_output = 'output'
    nodes.append(helper.make_node(
        'Identity',
        inputs=[prev_output],
        outputs=[final_output],
        name='final_output',
    ))

    # Create input/output definitions with concrete shapes for testing
    # AEC model: batch=1, seq_len=16, features=64
    inputs = [
        helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 16, 64]),
    ]

    outputs = [
        helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 16, 64]),
    ]

    # Create graph
    graph = helper.make_graph(
        nodes,
        model_name,
        inputs,
        outputs,
    )

    # Create model with compatible IR version
    model = helper.make_model(graph, producer_name='mgk-decompiler', opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 6  # Compatible with older ONNX Runtime

    # Add model metadata
    weight_stats = data.get('weight_stats', {})
    model.doc_string = f"""
MGK Model: {model_name}
Total layers: {len(layers)}
Weight size: {weight_stats.get('total_bytes', 0):,} bytes
Sparsity: {weight_stats.get('zero_percentage', 0):.1f}%

NOTE: This ONNX model is a simplified sequential representation.
Original layer types are preserved in node doc_strings.
"""

    # Save model
    onnx.save(model, str(output_path))
    print(f"ONNX model saved to: {output_path}")
    print(f"  Layers: {len(nodes)}")
    print(f"  Inputs: {[i.name for i in inputs]}")
    print(f"  Outputs: {[o.name for o in outputs]}")

    return True


def main():
    parser = argparse.ArgumentParser(description='Export MGK model to ONNX format')
    parser.add_argument('input', type=Path, help='Input JSON file from mgk-decompiler')
    parser.add_argument('-o', '--output', type=Path, help='Output ONNX file')
    args = parser.parse_args()
    
    if not args.output:
        args.output = args.input.with_suffix('.onnx')
    
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    success = export_to_onnx(args.input, args.output)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

