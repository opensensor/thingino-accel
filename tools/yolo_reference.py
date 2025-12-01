#!/usr/bin/env python3
"""
Run YOLOv5 ONNX model on test image to get reference output.
Compare with Mars runtime output.
"""

import numpy as np
import onnxruntime as ort
import sys
from pathlib import Path

# YOLOv5 anchors (same as in mars_detect.c)
ANCHORS = [
    [[10, 13], [16, 30], [33, 23]],      # 80x80, stride 8
    [[30, 61], [62, 45], [59, 119]],     # 40x40, stride 16
    [[116, 90], [156, 198], [373, 326]], # 20x20, stride 32
]
STRIDES = [8, 16, 32]
CONF_THRESHOLD = 0.35
NUM_CLASSES = 80

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

def load_ppm(path):
    """Load PPM image file."""
    with open(path, 'rb') as f:
        header = f.readline().decode().strip()
        assert header == 'P6', f"Expected P6, got {header}"
        # Skip comments
        line = f.readline().decode().strip()
        while line.startswith('#'):
            line = f.readline().decode().strip()
        w, h = map(int, line.split())
        maxval = int(f.readline().decode().strip())
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(h, w, 3)

def preprocess(img):
    """Preprocess image for YOLOv5: RGB, 0-1 range, NCHW format."""
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, 0)  # Add batch dim
    return img

def decode_head(output, scale_idx, scale=1.0):
    """Decode YOLOv5 detection head output."""
    # Output shape: [1, 255, H, W] (NCHW) or [1, H, W, 255] (NHWC)
    # Reshape to [1, 3, 85, H, W] then transpose to [1, 3, H, W, 85]

    # Dequantize if INT8
    if output.dtype == np.int8:
        output = output.astype(np.float32) * scale
        print(f"  Dequantized with scale={scale}")

    if output.shape[1] == 255:  # NCHW: [1, 255, H, W]
        _, c, h, w = output.shape
        output = output.reshape(1, 3, 85, h, w)
        output = output.transpose(0, 1, 3, 4, 2)  # [1, 3, H, W, 85]
    else:  # NHWC: [1, H, W, 255]
        _, h, w, c = output.shape
        output = output.reshape(1, h, w, 3, 85)
        output = output.transpose(0, 3, 1, 2, 4)  # [1, 3, H, W, 85]
    
    stride = STRIDES[scale_idx]
    anchors = np.array(ANCHORS[scale_idx])
    
    # Create grid
    grid_h, grid_w = output.shape[2], output.shape[3]
    grid_y, grid_x = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing='ij')
    
    detections = []
    
    # Debug: print values at key positions
    print(f"\n  [Head {scale_idx}] Shape: {output.shape}, stride={stride}")
    
    # Check positions where we expect people
    if scale_idx == 0:
        check_pos = [(31, 18, "Zidane"), (22, 62, "Ancelotti"), (40, 40, "Center"), (0, 0, "Corner"), (49, 45, "MarsMax")]
    elif scale_idx == 1:
        check_pos = [(15, 9, "Zidane"), (11, 31, "Ancelotti"), (20, 20, "Center"), (0, 0, "Corner"), (2, 3, "MarsMax")]
    else:
        check_pos = [(7, 4, "Zidane"), (5, 15, "Ancelotti"), (10, 10, "Center"), (0, 0, "Corner"), (2, 16, "MarsMax")]

    for y, x, name in check_pos:
        for a in range(3):
            box = output[0, a, y, x]  # [85]
            raw_obj = box[4]
            obj_conf = sigmoid(raw_obj)
            cls_scores = sigmoid(box[5:])
            best_cls = np.argmax(cls_scores)
            if obj_conf > 0.05 or name in ["Corner", "MarsMax"]:
                print(f"    {name} pos[{y},{x},a{a}]: raw_obj={raw_obj:.1f} obj={obj_conf:.3f}, cls{best_cls}={cls_scores[best_cls]:.3f}")
    
    # Find max objectness
    obj_all = sigmoid(output[0, :, :, :, 4])
    max_idx = np.unravel_index(np.argmax(obj_all), obj_all.shape)
    max_obj = obj_all[max_idx]
    print(f"  Max objectness: {max_obj:.3f} at anchor={max_idx[0]}, y={max_idx[1]}, x={max_idx[2]}")
    
    # Decode all detections
    for a in range(3):
        for y in range(grid_h):
            for x in range(grid_w):
                box = output[0, a, y, x]
                obj_conf = sigmoid(box[4])
                if obj_conf < CONF_THRESHOLD:
                    continue
                
                cls_scores = sigmoid(box[5:])
                best_cls = np.argmax(cls_scores)
                final_conf = obj_conf * cls_scores[best_cls]
                if final_conf < CONF_THRESHOLD:
                    continue
                
                # Decode box
                bx, by, bw, bh = box[0], box[1], box[2], box[3]
                cx = (sigmoid(bx) * 2 - 0.5 + x) * stride
                cy = (sigmoid(by) * 2 - 0.5 + y) * stride
                w = (sigmoid(bw) * 2) ** 2 * anchors[a, 0]
                h = (sigmoid(bh) * 2) ** 2 * anchors[a, 1]
                
                detections.append({
                    'x1': cx - w/2, 'y1': cy - h/2,
                    'x2': cx + w/2, 'y2': cy + h/2,
                    'conf': final_conf, 'cls': best_cls
                })
    
    return detections

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/yolov5n_3heads.onnx"
    image_path = sys.argv[2] if len(sys.argv) > 2 else "/mnt/nfs/test_image.ppm"
    
    print(f"Model: {model_path}")
    print(f"Image: {image_path}")
    
    # Load model
    sess = ort.InferenceSession(model_path)
    print(f"Inputs: {[i.name for i in sess.get_inputs()]}")
    print(f"Outputs: {[o.name for o in sess.get_outputs()]}")
    
    # Load and preprocess image
    img = load_ppm(image_path)
    print(f"Image shape: {img.shape}")
    inp = preprocess(img)
    print(f"Input shape: {inp.shape}")
    
    # Run inference
    outputs = sess.run(None, {sess.get_inputs()[0].name: inp})
    print(f"\nOutputs: {len(outputs)} heads")
    for i, o in enumerate(outputs):
        print(f"  Head {i}: {o.shape}, dtype={o.dtype}")
        # Print raw INT8 values at position 0,0
        if o.dtype == np.int8:
            if o.shape[1] == 255:  # NCHW
                print(f"    NCHW pos[0,0] first 20 vals: {o[0, :20, 0, 0].tolist()}")
            else:  # NHWC
                print(f"    NHWC pos[0,0] first 20 vals: {o[0, 0, 0, :20].tolist()}")
    
    # Output scales from Mars compiler (found earlier)
    output_scales = [0.135779, 0.116839, 0.107981]

    # Decode detections
    all_dets = []
    for i, out in enumerate(outputs):
        dets = decode_head(out, i, scale=output_scales[i])
        all_dets.extend(dets)
    
    print(f"\n=== Detections ({len(all_dets)}) ===")
    for d in sorted(all_dets, key=lambda x: -x['conf'])[:20]:
        print(f"  cls={d['cls']:2d} ({d['conf']:.1%}): [{d['x1']:.0f}, {d['y1']:.0f}, {d['x2']:.0f}, {d['y2']:.0f}]")

if __name__ == "__main__":
    main()

