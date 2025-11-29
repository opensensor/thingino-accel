#!/usr/bin/env python3
"""Test YOLO ONNX model on sample images with full detection."""

import argparse
import numpy as np
import onnxruntime as ort
import cv2
from pathlib import Path


# COCO class names
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


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize and pad image."""
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def xywh2xyxy(x):
    """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms(boxes, scores, iou_threshold=0.45):
    """Non-maximum suppression."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def postprocess(outputs, conf_thres=0.25, iou_thres=0.45, img_shape=(640, 640)):
    """Post-process YOLOv5 outputs."""
    # outputs is list of [1,3,H,W,85] tensors at different scales
    # 85 = 4 (box) + 1 (obj conf) + 80 (class probs)
    detections = []
    anchors = {
        80: [[10,13], [16,30], [33,23]],      # P3/8
        40: [[30,61], [62,45], [59,119]],     # P4/16
        20: [[116,90], [156,198], [373,326]]  # P5/32
    }
    strides = {80: 8, 40: 16, 20: 32}

    for output in outputs:
        bs, na, ny, nx, nc = output.shape
        stride = strides.get(ny, 8)
        anchor = anchors.get(ny, [[10,13], [16,30], [33,23]])

        # Create grid
        yv, xv = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        grid = np.stack([xv, yv], axis=-1).reshape(1, 1, ny, nx, 2).astype(np.float32)
        anchor_grid = np.array(anchor).reshape(1, na, 1, 1, 2).astype(np.float32)

        # Decode
        output = 1 / (1 + np.exp(-output))  # sigmoid
        xy = (output[..., :2] * 2 - 0.5 + grid) * stride
        wh = (output[..., 2:4] * 2) ** 2 * anchor_grid
        conf = output[..., 4:5]
        cls = output[..., 5:]

        # Flatten
        xy = xy.reshape(-1, 2)
        wh = wh.reshape(-1, 2)
        conf = conf.reshape(-1)
        cls = cls.reshape(-1, 80)

        # Filter by confidence
        mask = conf > conf_thres
        xy, wh, conf, cls = xy[mask], wh[mask], conf[mask], cls[mask]

        if len(conf) > 0:
            # Get class scores
            cls_conf = cls.max(axis=1)
            cls_id = cls.argmax(axis=1)
            scores = conf * cls_conf

            # Create boxes
            boxes = np.concatenate([xy - wh/2, xy + wh/2], axis=1)
            detections.append(np.column_stack([boxes, scores, cls_id]))

    if not detections:
        return np.array([])

    detections = np.vstack(detections)

    # NMS per class
    final = []
    for cls_id in np.unique(detections[:, 5]):
        cls_mask = detections[:, 5] == cls_id
        cls_dets = detections[cls_mask]
        keep = nms(cls_dets[:, :4], cls_dets[:, 4], iou_thres)
        final.extend(cls_dets[keep])

    return np.array(final) if final else np.array([])


def draw_detections(img, detections, ratio, pad):
    """Draw detection boxes on image."""
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        # Rescale to original image
        x1 = (x1 - pad[0]) / ratio
        y1 = (y1 - pad[1]) / ratio
        x2 = (x2 - pad[0]) / ratio
        y2 = (y2 - pad[1]) / ratio

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls_name = COCO_CLASSES[int(cls_id)] if int(cls_id) < len(COCO_CLASSES) else f"cls{int(cls_id)}"

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{cls_name}: {conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img


def main():
    parser = argparse.ArgumentParser(description="Test YOLO ONNX model")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", help="Output image path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    print(f"=== YOLO ONNX Detection ===")
    print(f"Model: {args.model}")
    print(f"Image: {args.image}")

    session = ort.InferenceSession(args.model)
    input_info = session.get_inputs()[0]

    print(f"Input: {input_info.name}, shape={input_info.shape}")
    for out in session.get_outputs():
        print(f"Output: {out.name}, shape={out.shape}")

    # Load and preprocess
    img = cv2.imread(args.image)
    img_orig = img.copy()
    img_prep, ratio, pad = letterbox(img, 640)
    img_prep = img_prep[:, :, ::-1].astype(np.float32) / 255.0
    img_prep = img_prep.transpose(2, 0, 1)[np.newaxis, ...]

    print(f"\nRunning inference...")
    outputs = session.run(None, {input_info.name: img_prep})

    print(f"Processing {len(outputs)} output scales...")
    dets = postprocess(outputs, conf_thres=args.conf)

    print(f"\nDetections: {len(dets)}")
    for det in dets:
        cls_name = COCO_CLASSES[int(det[5])] if int(det[5]) < len(COCO_CLASSES) else f"cls{int(det[5])}"
        print(f"  {cls_name}: {det[4]:.3f} @ [{det[0]:.0f},{det[1]:.0f},{det[2]:.0f},{det[3]:.0f}]")

    if args.output and len(dets) > 0:
        img_out = draw_detections(img_orig, dets, ratio, pad)
        cv2.imwrite(args.output, img_out)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()

