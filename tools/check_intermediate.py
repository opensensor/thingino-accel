#!/usr/bin/env python3
"""
Get intermediate layer outputs from ONNX model to compare with Mars.
"""

import numpy as np
import onnxruntime as ort
import onnx
import sys

def load_ppm(path):
    with open(path, 'rb') as f:
        header = f.readline().decode().strip()
        assert header == 'P6'
        line = f.readline().decode().strip()
        while line.startswith('#'):
            line = f.readline().decode().strip()
        w, h = map(int, line.split())
        maxval = int(f.readline().decode().strip())
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(h, w, 3)

def preprocess_nchw(img):
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return img

model_path = sys.argv[1] if len(sys.argv) > 1 else "models/yolov5n_3heads_ir8.onnx"
image_path = sys.argv[2] if len(sys.argv) > 2 else "test_image_640.ppm"

print(f"Model: {model_path}")
print(f"Image: {image_path}")

# Load model and add intermediate outputs
model = onnx.load(model_path)

# Find the first few conv outputs after first conv
# First conv: /model.0/conv/Conv -> /model.0/conv/Conv_output_0
# We want the quantized output: /model.0/conv/Conv_output_0_QuantizeLinear_Output

intermediate_outputs = []
targets = [
    'model.0/conv/Conv_output_0',  # First conv
    'model.0/act/Mul_output_0',    # First SiLU
    'model.9/cv3/act/Mul_output_0',  # End of backbone C3
    'model.13/cv3/act/Mul_output_0', # End of SPPF branch
    'model.17/Concat_output_0',   # First FPN concat
    'model.20/Concat_output_0',   # Second FPN concat
    'model.23/Concat_output_0',   # Third FPN concat
]
for node in model.graph.node:
    output_name = node.output[0] if node.output else None
    if output_name:
        for t in targets:
            if t in output_name:
                print(f"Found: {node.op_type} -> {output_name}")
                intermediate_outputs.append(output_name)
                break
        
# Add intermediate outputs to model
for output_name in intermediate_outputs:
    # Determine data type from node output
    dtype = onnx.TensorProto.FLOAT
    if 'QuantizeLinear_Output' in output_name:
        dtype = onnx.TensorProto.INT8
    value_info = onnx.helper.make_tensor_value_info(output_name, dtype, None)
    model.graph.output.append(value_info)

# Save modified model
tmp_model_path = "/tmp/model_with_intermediates.onnx"
onnx.save(model, tmp_model_path)

# Load and run
print("\nLoading image...")
img = load_ppm(image_path)
inp = preprocess_nchw(img)
print(f"Input shape: {inp.shape}")
print(f"Input first pixel RGB (float): {inp[0, :, 0, 0]}")
print(f"Input first pixel RGB (INT8 scale 0.007874): {(inp[0, :, 0, 0] / 0.007874).astype(int)}")

sess = ort.InferenceSession(tmp_model_path)
print(f"\nOutputs: {[o.name for o in sess.get_outputs()]}")

outputs = sess.run(None, {sess.get_inputs()[0].name: inp})

for i, o in enumerate(outputs):
    name = sess.get_outputs()[i].name
    print(f"\n{name}")
    print(f"  Shape: {o.shape}, dtype: {o.dtype}")
    if o.dtype == np.int8:
        # NCHW format - print first 16 channels at pos (0,0)
        if len(o.shape) == 4 and o.shape[1] > 0:
            print(f"  NCHW pos[0,0] first 16 ch: {o[0, :16, 0, 0].tolist()}")
    else:
        # Float - print first 16 channels at pos (0,0)
        if len(o.shape) == 4 and o.shape[1] > 0:
            vals = o[0, :16, 0, 0]
            print(f"  NCHW pos[0,0] first 16 ch (float): {vals.tolist()}")
            # Also show what INT8 quantized values would be
            scale = 0.236442  # First conv output scale from Mars
            int8_vals = np.clip(vals / scale, -128, 127).astype(np.int8)
            print(f"  NCHW pos[0,0] first 16 ch (INT8 @ scale {scale}): {int8_vals.tolist()}")

