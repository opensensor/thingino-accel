#!/usr/bin/env python3
"""Test ONNX model inference and compare with Mars runtime."""

import sys
import numpy as np
import onnxruntime as ort
from PIL import Image

def load_image(path):
    """Load image (PPM, JPG, etc)."""
    with Image.open(path) as img:
        return np.array(img.convert('RGB'))

def preprocess(image, input_scale=0.007874):
    """Preprocess image to INT8 NHWC format."""
    # Resize to 640x640 if needed
    if image.shape[:2] != (640, 640):
        from PIL import Image as PILImage
        img = PILImage.fromarray(image)
        img = img.resize((640, 640))
        image = np.array(img)
    
    # Quantize to INT8 using input scale
    # Mars does: int8_val = round(pixel_val * input_scale / 1.0)
    # Since ONNX expects float input, we'll use that
    float_input = image.astype(np.float32) / 255.0
    
    # Also compute what Mars sees
    int8_input = np.clip(np.round(image * input_scale), -128, 127).astype(np.int8)
    
    return float_input, int8_input

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def decode_yolov5_head(output, scale_idx, anchors, strides, grid_sizes, threshold=0.25):
    """Decode YOLOv5 detection head output."""
    # output shape: [1, 255, H, W] (NCHW) or [1, H, W, 255] (NHWC)
    if output.shape[1] == 255:  # NCHW
        output = np.transpose(output, (0, 2, 3, 1))  # -> NHWC
    
    grid_h, grid_w = output.shape[1], output.shape[2]
    stride = strides[scale_idx]
    anchor = anchors[scale_idx]
    
    print(f"\n[Scale {scale_idx}] Grid={grid_h}x{grid_w}, stride={stride}")
    print(f"  Output shape: {output.shape}, dtype: {output.dtype}")
    print(f"  First 16 values at [0,0]: {output[0,0,0,:16]}")
    
    # Find max objectness
    max_obj = -1e9
    max_pos = None
    
    detections = []
    for y in range(grid_h):
        for x in range(grid_w):
            for a in range(3):
                offset = a * 85
                box = output[0, y, x, offset:offset+85]
                
                bx, by, bw, bh, obj = box[:5]
                obj_conf = sigmoid(obj)
                
                if obj_conf > max_obj:
                    max_obj = obj_conf
                    max_pos = (y, x, a)
                
                if obj_conf < threshold:
                    continue
                
                classes = box[5:]
                best_class = np.argmax(classes)
                class_conf = sigmoid(classes[best_class])
                final_conf = obj_conf * class_conf
                
                if final_conf < threshold:
                    continue
                
                cx = (sigmoid(bx) * 2 - 0.5 + x) * stride
                cy = (sigmoid(by) * 2 - 0.5 + y) * stride
                w = (sigmoid(bw) * 2) ** 2 * anchor[a][0]
                h = (sigmoid(bh) * 2) ** 2 * anchor[a][1]
                
                detections.append({
                    'x1': cx - w/2, 'y1': cy - h/2,
                    'x2': cx + w/2, 'y2': cy + h/2,
                    'conf': final_conf, 'class': best_class
                })
    
    print(f"  Max objectness: {max_obj:.3f} at pos {max_pos}")
    return detections

def dump_intermediate_outputs(model_path, image_path):
    """Run ONNX model and dump first few layer outputs."""
    import onnx
    from onnx import helper

    print(f"\n=== Dumping intermediate outputs ===")
    print(f"Model: {model_path}")

    # Load model
    model = onnx.load(model_path)

    # Find first Conv node output
    conv_outputs = []
    for node in model.graph.node:
        if 'Conv' in node.op_type and len(conv_outputs) < 3:
            conv_outputs.append(node.output[0])
            print(f"Found {node.op_type}: {node.name} -> {node.output[0]}")

    if not conv_outputs:
        print("No Conv nodes found!")
        return

    # Add intermediate outputs to the model
    for out_name in conv_outputs:
        # Check if this is already in outputs
        existing = [o.name for o in model.graph.output]
        if out_name not in existing:
            # Find the value info for this tensor (use FLOAT as internal type)
            model.graph.output.append(
                helper.make_tensor_value_info(out_name, onnx.TensorProto.FLOAT, None)
            )

    # Save modified model
    temp_model_path = '/tmp/model_with_intermediates.onnx'
    model.ir_version = 8  # Downgrade for compatibility
    onnx.save(model, temp_model_path)

    # Load and run
    sess = ort.InferenceSession(temp_model_path)

    # Load image
    image = load_image(image_path)
    float_input, int8_input = preprocess(image)

    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape

    if input_shape[-1] == 3:  # NHWC
        input_data = float_input[np.newaxis, ...]
    else:  # NCHW
        input_data = np.transpose(float_input, (2, 0, 1))[np.newaxis, ...]

    output_names = [o.name for o in sess.get_outputs()]
    print(f"\nOutput names: {output_names}")

    outputs = sess.run(output_names, {input_name: input_data.astype(np.float32)})

    for i, (name, out) in enumerate(zip(output_names, outputs)):
        print(f"\n{name}:")
        print(f"  Shape: {out.shape}, dtype: {out.dtype}")
        print(f"  Min: {out.min()}, Max: {out.max()}, Mean: {out.mean():.3f}")
        # Flatten and show first 16 values
        flat = out.flatten()
        print(f"  First 16 (raw order): {flat[:16]}")

        # For Conv outputs in NCHW, extract position [0,0] across all channels
        if len(out.shape) == 4 and out.shape[1] > 1:  # NCHW
            # out[0, :, 0, 0] = all channels at spatial (0,0)
            pos00_all_ch = out[0, :, 0, 0]
            print(f"  Pos[0,0] all channels (NCHW->NHWC): {pos00_all_ch[:16]}")

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model.onnx> <image>")
        print(f"       {sys.argv[0]} --dump-intermediate <model.onnx> <image>")
        sys.exit(1)

    if sys.argv[1] == '--dump-intermediate':
        dump_intermediate_outputs(sys.argv[2], sys.argv[3])
        return

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    print(f"Loading model: {model_path}")
    sess = ort.InferenceSession(model_path)
    
    print(f"Loading image: {image_path}")
    image = load_image(image_path)
    print(f"  Image shape: {image.shape}")
    print(f"  First 16 RGB values: {image.flatten()[:16]}")
    
    float_input, int8_input = preprocess(image)
    print(f"  Float input shape: {float_input.shape}")
    print(f"  INT8 input first 16: {int8_input.flatten()[:16]}")
    
    # Run inference
    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]
    print(f"\nInput name: {input_name}")
    print(f"Output names: {output_names}")
    
    # ONNX expects NCHW, but our model might want NHWC
    # Check input shape
    input_shape = sess.get_inputs()[0].shape
    print(f"Expected input shape: {input_shape}")
    
    if input_shape[-1] == 3:  # NHWC
        input_data = float_input[np.newaxis, ...]  # [1, 640, 640, 3]
    else:  # NCHW
        input_data = np.transpose(float_input, (2, 0, 1))[np.newaxis, ...]  # [1, 3, 640, 640]
    
    print(f"Running inference with input shape: {input_data.shape}")
    outputs = sess.run(output_names, {input_name: input_data.astype(np.float32)})
    
    # YOLOv5 anchors
    anchors = [
        [[10, 13], [16, 30], [33, 23]],
        [[30, 61], [62, 45], [59, 119]],
        [[116, 90], [156, 198], [373, 326]]
    ]
    strides = [8, 16, 32]
    grid_sizes = [80, 40, 20]
    
    print(f"\nModel has {len(outputs)} outputs")
    all_dets = []
    for i, out in enumerate(outputs):
        print(f"\nOutput {i}: shape={out.shape}, min={out.min()}, max={out.max()}, mean={out.mean():.2f}")
        dets = decode_yolov5_head(out, i, anchors, strides, grid_sizes)
        all_dets.extend(dets)
    
    print(f"\n=== Total detections: {len(all_dets)} ===")
    for i, d in enumerate(all_dets[:20]):
        print(f"  [{i}] class={d['class']}, conf={d['conf']:.3f}, box=[{d['x1']:.0f},{d['y1']:.0f},{d['x2']:.0f},{d['y2']:.0f}]")

if __name__ == '__main__':
    main()

