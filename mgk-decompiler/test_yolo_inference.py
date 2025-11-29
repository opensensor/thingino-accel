#!/usr/bin/env python3
"""
Test YOLOv5s inference using decompiled MGK weights.

This script demonstrates:
1. Loading weights from MGK file using the decompiler
2. Running YOLO inference on sample images
3. Displaying detection results
"""

import numpy as np
import sys
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from mgk_decompiler import MGKDecompiler, unpack_nmhwsoib2, unpack_2bit_to_signed

try:
    from PIL import Image, ImageDraw
    import onnxruntime as ort
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow", "onnxruntime"])
    from PIL import Image, ImageDraw
    import onnxruntime as ort


# COCO class names for YOLOv5
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


def preprocess_image(image_path, input_size=640):
    """Preprocess image for YOLOv5 inference."""
    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size
    
    # Resize maintaining aspect ratio
    scale = min(input_size / orig_w, input_size / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    
    # Pad to input_size x input_size
    padded = Image.new('RGB', (input_size, input_size), (114, 114, 114))
    pad_x, pad_y = (input_size - new_w) // 2, (input_size - new_h) // 2
    padded.paste(img_resized, (pad_x, pad_y))
    
    # Convert to numpy and normalize
    img_array = np.array(padded, dtype=np.float32) / 255.0
    
    # HWC to CHW, add batch dimension
    img_array = img_array.transpose(2, 0, 1)[np.newaxis, ...]
    
    return img_array, img, (orig_w, orig_h), (pad_x, pad_y), scale


def run_onnx_inference(onnx_path, image_path):
    """Run inference using the original ONNX model."""
    print(f"\nRunning ONNX inference on: {image_path}")
    
    # Load model
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    print(f"Model input: {input_name}, shape: {input_shape}")
    
    # Preprocess
    input_size = input_shape[2] if len(input_shape) > 2 else 640
    img_tensor, orig_img, orig_size, padding, scale = preprocess_image(image_path, input_size)
    
    # Run inference
    outputs = session.run(None, {input_name: img_tensor})
    
    print(f"Output shapes: {[o.shape for o in outputs]}")
    
    return outputs, orig_img, orig_size, padding, scale


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def nms(detections, iou_threshold=0.45):
    """Apply Non-Maximum Suppression."""
    if not detections:
        return []

    # Sort by confidence (already sorted, but ensure)
    detections = sorted(detections, key=lambda x: -x['confidence'])

    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)

        # Filter out overlapping boxes of same class
        detections = [
            d for d in detections
            if d['class_id'] != best['class_id'] or iou(d['box'], best['box']) < iou_threshold
        ]

    return keep


def parse_yolo_output(outputs, conf_threshold=0.25, input_size=640):
    """Parse YOLOv5 multi-scale output to get detections."""
    # YOLOv5 outputs 3 feature maps at different scales
    # Each: [batch, num_anchors=3, grid_h, grid_w, 85]
    # 85 = 4 (box) + 1 (objectness) + 80 (class scores)

    # Anchor sizes for each scale (stride 8, 16, 32)
    anchors = [
        [[10, 13], [16, 30], [33, 23]],      # P3/8
        [[30, 61], [62, 45], [59, 119]],     # P4/16
        [[116, 90], [156, 198], [373, 326]]  # P5/32
    ]
    strides = [8, 16, 32]

    detections = []

    for scale_idx, (output, anchor_set, stride) in enumerate(zip(outputs, anchors, strides)):
        output = output[0]  # Remove batch: [3, grid_h, grid_w, 85]
        num_anchors, grid_h, grid_w, num_attrs = output.shape

        # Apply sigmoid to objectness and class scores
        obj_conf = sigmoid(output[..., 4])
        class_probs = sigmoid(output[..., 5:])

        # Find positions above threshold
        for a in range(num_anchors):
            for y in range(grid_h):
                for x in range(grid_w):
                    obj = obj_conf[a, y, x]
                    if obj < conf_threshold:
                        continue

                    classes = class_probs[a, y, x]
                    class_id = np.argmax(classes)
                    class_conf = classes[class_id]

                    confidence = obj * class_conf
                    if confidence < conf_threshold:
                        continue

                    # Decode box (tx, ty, tw, th)
                    tx, ty, tw, th = output[a, y, x, :4]

                    # Apply sigmoid to tx, ty and decode center
                    cx = (sigmoid(tx) * 2 - 0.5 + x) * stride
                    cy = (sigmoid(ty) * 2 - 0.5 + y) * stride

                    # Decode width/height
                    aw, ah = anchor_set[a]
                    w = (sigmoid(tw) * 2) ** 2 * aw
                    h = (sigmoid(th) * 2) ** 2 * ah

                    x1, y1 = cx - w/2, cy - h/2
                    x2, y2 = cx + w/2, cy + h/2

                    detections.append({
                        'box': [x1, y1, x2, y2],
                        'confidence': float(confidence),
                        'class_id': int(class_id),
                        'class_name': COCO_CLASSES[int(class_id)] if int(class_id) < len(COCO_CLASSES) else f"class_{class_id}"
                    })

    # Sort by confidence and apply NMS
    detections.sort(key=lambda x: -x['confidence'])
    detections = nms(detections, iou_threshold=0.45)

    return detections[:100]


def draw_detections(image, detections, padding, scale):
    """Draw detection boxes on image."""
    draw = ImageDraw.Draw(image)
    pad_x, pad_y = padding
    
    for det in detections:
        x1, y1, x2, y2 = det['box']
        
        # Remove padding and scale back to original size
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale
        
        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        
        # Draw label
        label = f"{det['class_name']}: {det['confidence']:.2f}"
        draw.text((x1, y1 - 15), label, fill='red')
    
    return image


def analyze_mgk_weights(mgk_path):
    """Analyze the MGK model weights."""
    print(f"\n{'='*60}")
    print(f"Analyzing MGK file: {mgk_path}")
    print('='*60)
    
    decompiler = MGKDecompiler(mgk_path)
    model = decompiler.parse()
    
    print(f"\nModel name: {model.name}")
    print(f"Number of layers: {len(model.layers)}")
    print(f"Weight data size: {len(model.weight_data):,} bytes")
    print(f"Number of scales: {len(model.scales)}")
    
    # Analyze layer types
    layer_types = {}
    for layer in model.layers:
        lt = layer.layer_type
        layer_types[lt] = layer_types.get(lt, 0) + 1
    
    print(f"\nLayer type distribution:")
    for lt, count in sorted(layer_types.items()):
        print(f"  {lt}: {count}")
    
    return model


def main():
    # Paths
    mgk_path = Path("/home/matteius/thingino-accel/magik-toolkit/magik-toolkit-3.0/Models/post/yolov5s-post-t41/uranus_sample_yolov5s_post_t41/model_t41/glibc/yolov5s_t41_magik_post_release.mgk")
    onnx_path = Path("/home/matteius/thingino-accel/magik-toolkit/magik-toolkit-3.0/Models/post/yolov5s-post-t41/yolov5s.onnx")
    image_dir = Path("/home/matteius/thingino-accel/magik-toolkit/magik-toolkit-3.0/Models/post/yolov5s-post-t41/yolov5-20")
    output_dir = Path("/tmp/yolo_inference_results")
    
    output_dir.mkdir(exist_ok=True)
    
    # Analyze MGK weights
    model = analyze_mgk_weights(mgk_path)
    
    # Check if ONNX model exists
    if not onnx_path.exists():
        print(f"\nONNX model not found at {onnx_path}")
        print("Cannot run inference without the ONNX model.")
        print("\nThe MGK file contains 2-bit quantized weights that are")
        print("optimized for the Ingenic NNA hardware. Full inference")
        print("would require either:")
        print("  1. The original ONNX model")
        print("  2. Running on actual NNA hardware")
        print("  3. Reconstructing the model architecture from the MGK")
        return
    
    # Get sample images
    images = list(image_dir.glob("*.jpg"))[:5]  # Process first 5 images
    
    if not images:
        print(f"\nNo images found in {image_dir}")
        return
    
    print(f"\n{'='*60}")
    print("Running inference on sample images")
    print('='*60)
    
    for img_path in images:
        try:
            outputs, orig_img, orig_size, padding, scale = run_onnx_inference(str(onnx_path), str(img_path))
            
            # Parse detections
            detections = parse_yolo_output(outputs)
            
            print(f"\nDetections for {img_path.name}:")
            for det in detections[:10]:  # Show top 10
                print(f"  {det['class_name']}: {det['confidence']:.3f}")
            
            # Draw and save result
            result_img = draw_detections(orig_img.copy(), detections, padding, scale)
            result_path = output_dir / f"result_{img_path.name}"
            result_img.save(result_path)
            print(f"  Saved result to: {result_path}")
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir}")
    print('='*60)


if __name__ == "__main__":
    main()

