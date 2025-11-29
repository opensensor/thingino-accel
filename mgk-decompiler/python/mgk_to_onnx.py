#!/usr/bin/env python3
"""
MGK to ONNX Converter

Converts the JSON output from the Rust MGK decompiler to an ONNX model.
This is the second stage of the decompilation pipeline.

Usage:
    python mgk_to_onnx.py --input model.json --output model.onnx
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    import numpy as np
except ImportError:
    print("Error: onnx and numpy are required. Install with: pip install onnx numpy")
    sys.exit(1)


# Map MGK layer types to ONNX operators
LAYER_TYPE_TO_ONNX = {
    "Conv": "Conv",
    "Pool": "MaxPool",  # Could also be AveragePool
    "Add": "Add",
    "Concat": "Concat",
    "Reshape": "Reshape",
    "Permute": "Transpose",
    "Gru": "GRU",
    "Normalize": "BatchNormalization",
    "Upsample": "Upsample",
    "Slice": "Slice",
    "FormatConvert": None,  # Internal format conversion, not in ONNX
    "DeQuantize": "DequantizeLinear",
    "GenerateBox": None,  # Custom op, may need special handling
    "SqueezeUnsqueeze": "Squeeze",  # or Unsqueeze
}


def load_decompiler_output(json_path: Path) -> Dict[str, Any]:
    """Load the JSON output from the Rust decompiler."""
    with open(json_path, 'r') as f:
        return json.load(f)


def create_onnx_graph(data: Dict[str, Any]) -> onnx.GraphProto:
    """Create an ONNX graph from decompiler output."""
    model_name = data.get("model_name", "decompiled_model")
    layers = data.get("layers", [])
    
    nodes = []
    inputs = []
    outputs = []
    initializers = []
    
    # Track tensor names for connecting layers
    tensor_counter = 0
    
    def next_tensor_name() -> str:
        nonlocal tensor_counter
        name = f"tensor_{tensor_counter}"
        tensor_counter += 1
        return name
    
    # Create nodes for each layer
    for layer in layers:
        layer_type = layer.get("layer_type", "Unknown")
        layer_id = layer.get("id", 0)
        
        onnx_op = LAYER_TYPE_TO_ONNX.get(layer_type)
        if onnx_op is None:
            print(f"Warning: Skipping unsupported layer type: {layer_type}")
            continue
        
        # Create input/output tensor names
        input_name = layer.get("input_tensors", [next_tensor_name()])[0] if layer.get("input_tensors") else next_tensor_name()
        output_name = layer.get("output_tensors", [next_tensor_name()])[0] if layer.get("output_tensors") else next_tensor_name()
        
        # Create the node
        node = helper.make_node(
            onnx_op,
            inputs=[input_name],
            outputs=[output_name],
            name=f"{layer_type}_{layer_id}"
        )
        nodes.append(node)
    
    # Create a placeholder input
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 1, 256, 1]  # Placeholder shape
    )
    inputs.append(input_tensor)
    
    # Create a placeholder output
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, None  # Unknown shape
    )
    outputs.append(output_tensor)
    
    # Create the graph
    graph = helper.make_graph(
        nodes,
        model_name,
        inputs,
        outputs,
        initializers
    )
    
    return graph


def create_onnx_model(data: Dict[str, Any]) -> onnx.ModelProto:
    """Create an ONNX model from decompiler output."""
    graph = create_onnx_graph(data)
    
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)]
    )
    
    model.ir_version = 7
    model.producer_name = "mgk-decompiler"
    model.producer_version = "0.1.0"
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Convert MGK decompiler JSON to ONNX")
    parser.add_argument("-i", "--input", required=True, help="Input JSON file from decompiler")
    parser.add_argument("-o", "--output", required=True, help="Output ONNX file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    print(f"Loading decompiler output: {input_path}")
    data = load_decompiler_output(input_path)
    
    print(f"Model name: {data.get('model_name', 'unknown')}")
    print(f"Layers: {len(data.get('layers', []))}")
    
    print("Creating ONNX model...")
    model = create_onnx_model(data)
    
    print(f"Saving ONNX model: {output_path}")
    onnx.save(model, str(output_path))
    
    print("Done!")


if __name__ == "__main__":
    main()

