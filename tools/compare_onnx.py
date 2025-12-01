#!/usr/bin/env python3
"""
Compare Mars runtime output with ONNX reference.
Loads the same image, preprocesses it identically, runs through ONNX,
and dumps key output values for comparison.
"""
import sys
import numpy as np
try:
    import onnxruntime as ort
    from PIL import Image
except ImportError:
    print("Install dependencies: pip install onnxruntime pillow numpy")
    sys.exit(1)

def preprocess_image(image_path, input_size=640, nchw=True):
    """Preprocess image exactly as mars_detect does."""
    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size
    print(f"Original image: {orig_w}x{orig_h}")

    # Resize to input_size x input_size (simple resize, not letterbox)
    img = img.resize((input_size, input_size), Image.BILINEAR)

    # Convert to numpy and normalize to 0-1
    img_np = np.array(img, dtype=np.float32) / 255.0

    # Check raw RGB values before preprocessing
    print(f"RGB first 16 values (0-255 range): {(img_np[0,0:6,:].flatten() * 255).astype(np.uint8)}")

    if nchw:
        # NCHW format: [1, C, H, W]
        img_np = np.transpose(img_np, (2, 0, 1))  # HWC -> CHW
        img_np = np.expand_dims(img_np, axis=0)    # CHW -> NCHW
    else:
        # NHWC format: [1, H, W, C]
        img_np = np.expand_dims(img_np, axis=0)

    print(f"Input shape: {img_np.shape}, dtype: {img_np.dtype}")
    print(f"Input range: [{img_np.min():.4f}, {img_np.max():.4f}]")

    return img_np, orig_w, orig_h

def analyze_detection_output(output, scale_idx, grid_h, grid_w, stride, scale, is_nchw=True):
    """Analyze a YOLO detection head output."""
    print(f"\n[Scale {scale_idx}] Grid={grid_h}x{grid_w}, stride={stride}, scale={scale:.6f}")

    # Convert NCHW [1, 255, H, W] to NHWC [1, H, W, 255] if needed
    if is_nchw and output.ndim == 4 and output.shape[1] == 255:
        output = np.transpose(output, (0, 2, 3, 1))
        print(f"  Converted to NHWC: {output.shape}")

    # Output shape: [1, H, W, 255] for 3 anchors * 85 values
    # 85 = 4 (box) + 1 (obj) + 80 (classes)

    # Find max objectness
    max_obj = -999
    max_pos = None

    # Check specific positions
    test_positions = [
        (grid_h//2 - 9, grid_w//2 - 2),  # roughly center-left
        (grid_h//4 + 2, grid_w - 2),      # upper right area
        (grid_h//2, grid_w//2),           # center
        (0, 0),                            # top-left corner
    ]

    for y, x in test_positions:
        if y >= grid_h or x >= grid_w or y < 0 or x < 0:
            continue
        # Get objectness for all 3 anchors
        obj_values = []
        for a in range(3):
            obj_idx = a * 85 + 4  # offset to objectness
            obj_raw = float(output[0, y, x, obj_idx]) * scale
            obj_values.append(obj_raw)

            if obj_raw > max_obj:
                max_obj = obj_raw
                max_pos = (y, x, a)

        # Convert to confidence using sigmoid
        confs = [1.0 / (1.0 + np.exp(-v)) for v in obj_values]
        print(f"    pos[{y},{x}]: obj_raw=[{obj_values[0]:.2f},{obj_values[1]:.2f},{obj_values[2]:.2f}] -> conf=[{confs[0]:.3f},{confs[1]:.3f},{confs[2]:.3f}]")

    # Scan for absolute max
    for y in range(grid_h):
        for x in range(grid_w):
            for a in range(3):
                obj_idx = a * 85 + 4
                obj_raw = float(output[0, y, x, obj_idx]) * scale
                if obj_raw > max_obj:
                    max_obj = obj_raw
                    max_pos = (y, x, a)

    max_conf = 1.0 / (1.0 + np.exp(-max_obj))
    # Get raw int8 value at max position
    y, x, a = max_pos
    obj_idx = a * 85 + 4
    raw_int8 = int(output[0, y, x, obj_idx])
    print(f"  Max objectness: {max_conf:.3f} at pos[{y},{x},a{a}] (raw_scaled={max_obj:.2f}, raw_int8={raw_int8})")

    # Find all positions with objectness > 0.25
    high_obj_count = 0
    for y in range(grid_h):
        for x in range(grid_w):
            for a in range(3):
                obj_idx = a * 85 + 4
                obj_raw = float(output[0, y, x, obj_idx]) * scale
                obj_conf = 1.0 / (1.0 + np.exp(-obj_raw))
                if obj_conf > 0.25:
                    high_obj_count += 1
                    if high_obj_count <= 5:
                        # Get class scores
                        class_start = a * 85 + 5
                        class_scores = output[0, y, x, class_start:class_start+80].astype(np.float32) * scale
                        class_probs = 1.0 / (1.0 + np.exp(-class_scores))
                        best_class = np.argmax(class_probs)
                        best_class_prob = class_probs[best_class]
                        combined = obj_conf * best_class_prob
                        print(f"    High obj at [{y},{x},a{a}]: obj={obj_conf:.3f}, class={best_class}({best_class_prob:.3f}), combined={combined:.3f}")
    print(f"  Total positions with obj > 0.25: {high_obj_count}")

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model.onnx> <image.ppm|jpg|png>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    
    print(f"Loading ONNX model: {model_path}")
    session = ort.InferenceSession(model_path)
    
    # Get input info
    input_info = session.get_inputs()[0]
    print(f"Input name: {input_info.name}, shape: {input_info.shape}, type: {input_info.type}")
    
    # Get output info
    print("Outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: shape={out.shape}, type={out.type}")
    
    # Preprocess image
    img_np, orig_w, orig_h = preprocess_image(image_path)
    
    # Run inference
    print("\nRunning inference...")
    outputs = session.run(None, {input_info.name: img_np})
    
    print(f"\nGot {len(outputs)} outputs")
    
    # Analyze each detection head
    strides = [8, 16, 32]  # YOLOv5 strides
    # Quantization scales from the .mars file output
    scales = [0.135779, 0.116839, 0.107981]  # From device output

    for i, output in enumerate(outputs):
        print(f"\nOutput {i}: shape={output.shape}, dtype={output.dtype}")
        print(f"  Range: [{output.min():.4f}, {output.max():.4f}]")
        print(f"  First 16 values: {output.flatten()[:16]}")

        # Check if NCHW format [1, 255, H, W] or NHWC [1, H, W, 255]
        is_nchw = len(output.shape) == 4 and output.shape[1] == 255
        is_nhwc = len(output.shape) == 4 and output.shape[-1] == 255

        if is_nchw:
            grid_h, grid_w = output.shape[2], output.shape[3]
            # Convert to NHWC for consistent comparison with Mars runtime
            nhwc_output = np.transpose(output, (0, 2, 3, 1))
            print(f"  NHWC first 16 at [0,0]: {nhwc_output[0, 0, 0, :16].tolist()}")
            analyze_detection_output(output, i, grid_h, grid_w, strides[i], scales[i], is_nchw=True)
        elif is_nhwc:
            grid_h, grid_w = output.shape[1], output.shape[2]
            print(f"  NHWC first 16 at [0,0]: {output[0, 0, 0, :16].tolist()}")
            analyze_detection_output(output, i, grid_h, grid_w, strides[i], scales[i], is_nchw=False)

if __name__ == "__main__":
    main()

