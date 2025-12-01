#!/usr/bin/env python3
"""
Extract intermediate layer outputs from ONNX model for comparison with Mars runtime.
"""
import sys
import numpy as np
import onnx
from onnx import numpy_helper
try:
    import onnxruntime as ort
    from PIL import Image
except ImportError:
    print("Install dependencies: pip install onnxruntime pillow numpy onnx")
    sys.exit(1)

def preprocess_image(image_path, input_size=640):
    """Preprocess image to NCHW float32."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((input_size, input_size), Image.BILINEAR)
    img_np = np.array(img, dtype=np.float32) / 255.0
    # NCHW format
    img_np = np.transpose(img_np, (2, 0, 1))
    img_np = np.expand_dims(img_np, axis=0)
    return img_np

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model.onnx> <image.jpg> [layer_name]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    target_layer = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Load model
    model = onnx.load(model_path)
    
    # Get all intermediate output names
    intermediate_outputs = []
    for node in model.graph.node:
        for output in node.output:
            intermediate_outputs.append(output)
    
    print(f"Model has {len(intermediate_outputs)} intermediate outputs")
    
    # Find first conv output (usually after QuantizeLinear)
    conv_outputs = []
    for i, node in enumerate(model.graph.node):
        if 'Conv' in node.op_type:
            conv_outputs.append((i, node.name, node.output[0]))
    
    print(f"\nFirst 5 Conv outputs:")
    for i, (idx, name, output) in enumerate(conv_outputs[:5]):
        print(f"  [{idx}] {name}: {output}")
    
    # If target layer specified, find it
    if target_layer:
        outputs_to_get = [target_layer]
    else:
        # Get first conv output and first few layer outputs
        outputs_to_get = [conv_outputs[0][2]] if conv_outputs else []
        # Also get the final outputs
        for out in model.graph.output:
            outputs_to_get.append(out.name)
    
    print(f"\nExtracting outputs: {outputs_to_get}")
    
    # Add intermediate outputs to model
    for output_name in outputs_to_get:
        # Check if already an output
        existing = [o.name for o in model.graph.output]
        if output_name not in existing:
            # Find the value info
            for vi in model.graph.value_info:
                if vi.name == output_name:
                    model.graph.output.append(vi)
                    break
            else:
                # Create a new output
                model.graph.output.append(onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, None))
    
    # Create session with modified model
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(model.SerializeToString(), sess_options)
    
    # Preprocess image
    img_np = preprocess_image(image_path)
    print(f"\nInput shape: {img_np.shape}")
    
    # Run inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(outputs_to_get, {input_name: img_np})
    
    # Print outputs
    for name, output in zip(outputs_to_get, outputs):
        print(f"\n{name}:")
        print(f"  Shape: {output.shape}, dtype: {output.dtype}")
        print(f"  Range: [{output.min():.4f}, {output.max():.4f}]")
        
        # For NCHW outputs, show first 16 values at [0,0] in NHWC order
        if output.ndim == 4:
            if output.shape[1] > output.shape[3]:  # NCHW
                nhwc = np.transpose(output, (0, 2, 3, 1))
                print(f"  NHWC first 16 at [0,0]: {nhwc[0, 0, 0, :16].tolist()}")
            else:  # Already NHWC
                print(f"  NHWC first 16 at [0,0]: {output[0, 0, 0, :16].tolist()}")
        else:
            print(f"  First 16 values: {output.flatten()[:16].tolist()}")

if __name__ == "__main__":
    main()

