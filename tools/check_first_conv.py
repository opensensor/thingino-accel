#!/usr/bin/env python3
"""Check first conv layer output between reference ONNX and expected Mars output."""

import numpy as np
import onnxruntime as ort
import onnx
from onnx import numpy_helper

def load_ppm(path):
    with open(path, 'rb') as f:
        header = f.readline().decode().strip()
        assert header == 'P6', f"Expected P6, got {header}"
        line = f.readline().decode().strip()
        while line.startswith('#'):
            line = f.readline().decode().strip()
        w, h = map(int, line.split())
        maxval = int(f.readline().decode().strip())
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(h, w, 3)

def preprocess_nchw(img):
    """YOLOv5 NCHW preprocessing: RGB, 0-1 range, NCHW format."""
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, 0)  # Add batch dim
    return img

def preprocess_nhwc_int8(img, scale=1.0/127.0):
    """Mars NHWC INT8 preprocessing."""
    img_float = img.astype(np.float32) / 255.0
    img_int8 = np.clip(img_float / scale, -128, 127).astype(np.int8)
    return img_int8  # Keep as HWC

model_path = "models/yolov5n_3heads_ir8.onnx"
image_path = "test_image_640.ppm"

print("Loading image...")
img = load_ppm(image_path)
print(f"Image shape: {img.shape}")

# Check input preprocessing match
inp_nchw = preprocess_nchw(img)
inp_nhwc_int8 = preprocess_nhwc_int8(img, scale=0.007874)

print(f"\nNCHW input shape: {inp_nchw.shape}, dtype={inp_nchw.dtype}")
print(f"NCHW first pixel RGB: {inp_nchw[0, :, 0, 0]}")
print(f"NHWC INT8 first pixel RGB: {inp_nhwc_int8[0, 0, :]}")

# Expected INT8 values with scale 0.007874 (1/127)
expected_int8 = np.clip(inp_nchw[0, :, 0, 0] / 0.007874, -128, 127).astype(np.int8)
print(f"Expected INT8 from float: {expected_int8}")

# Load ONNX model and find first conv output
print("\nLoading ONNX model...")
model = onnx.load(model_path)

# Find first Conv node
for node in model.graph.node:
    if node.op_type == 'Conv':
        print(f"First Conv: {node.name}")
        print(f"  Inputs: {node.input}")
        print(f"  Outputs: {node.output}")
        break

# Find first conv output name (after DequantizeLinear if QDQ)
first_conv_output = None
for node in model.graph.node:
    if 'model.0/conv/Conv' in node.name or '/model.0/conv/Conv' in node.output[0]:
        first_conv_output = node.output[0]
        print(f"Found first conv output: {first_conv_output}")
        break

# Also look for the quantized output
for node in model.graph.node:
    if 'QuantizeLinear' in node.op_type and first_conv_output and first_conv_output in node.input:
        print(f"Quantized output: {node.output[0]}")

# Get first conv weights
print("\nFirst Conv weights:")
for init in model.graph.initializer:
    if 'model.0.conv.weight' in init.name:
        w = numpy_helper.to_array(init)
        print(f"  {init.name}: shape={w.shape}, dtype={w.dtype}")
        if w.dtype == np.int8:
            print(f"  First 16 values (flattened): {w.flatten()[:16].tolist()}")
        break

# Compare with what Mars debug showed:
print("\n=== Mars debug showed ===")
print("Input first 16 bytes: 37 25 21 35 23 19 34 22 18 34 22 18 36 23 20 38")
print("Weight first 16 bytes: 0 0 4 -7 -5 0 -4 -1 0 -2 -1 -1 2 2 -1 4")
print("Conv0 Output first 16 bytes: 19 6 15 4 10 -2 7 4 16 -5 19 19 -14 -8 31 9")

# Calculate expected first pixel preprocessing
first_pixel = img[0, 0, :]  # RGB
print(f"\nActual first pixel RGB: {first_pixel}")
scale_in = 0.007874
int8_expected = np.clip(first_pixel.astype(np.float32) / 255.0 / scale_in, -128, 127).astype(np.int8)
print(f"Expected INT8 (scale={scale_in}): {int8_expected}")

# Run model and get intermediate output
print("\n=== Running inference to get first conv output ===")

# Add first conv output to model outputs
model_with_intermediates = onnx.load(model_path)
# Find the QuantizeLinear output after first conv
for node in model_with_intermediates.graph.node:
    if 'model.0/conv/Conv_output_0_QuantizeLinear' in node.name:
        intermediate_output = node.output[0]
        break
    if '/model.0/act/Mul_output_0' in node.output:
        intermediate_output = node.output[0]

# Use the original model outputs plus check first conv's quantized output
sess = ort.InferenceSession(model_path)
out_names = [o.name for o in sess.get_outputs()]
print(f"Model outputs: {out_names}")

# Run inference
outputs = sess.run(None, {sess.get_inputs()[0].name: inp_nchw})

# Print final head outputs for comparison
print("\n=== Final head outputs (compare with Mars) ===")
for i, o in enumerate(outputs):
    print(f"Head {i}: shape={o.shape}")
    # NCHW format - print first position
    print(f"  NCHW[0, :20, 0, 0]: {o[0, :20, 0, 0].tolist()}")

    # Also print at position where Mars found max objectness
    if i == 2:  # Head 2 (20x20)
        # Mars found max at pos[2,16] - let's check both Mars pos and reference max pos
        mars_y, mars_x = 2, 16
        ref_y, ref_x = 15, 8
        print(f"  Mars max pos [y=2,x=16] channel 4 (obj): {o[0, 4, mars_y, mars_x]}")
        print(f"  Ref max pos [y=15,x=8] channel 4 (obj): {o[0, 4, ref_y, ref_x]}")

