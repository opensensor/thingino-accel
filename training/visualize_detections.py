#!/usr/bin/env python3
"""
Visual testing script for detection model.
Runs inference on images and draws bounding boxes with class labels.
"""
import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tinydet import TinyDet

# Configuration
IMG_W, IMG_H = 320, 192
GRID_W, GRID_H = 20, 12
NUM_CLASSES = 4
CLASS_NAMES = ['person', 'vehicle', 'cat', 'dog']
CLASS_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1,y1,x2,y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)


def decode_predictions(pred, conf_thresh=0.25, nms_thresh=0.45):
    """Decode model predictions to bounding boxes"""
    C, H, W = pred.shape
    detections = []
    
    obj = torch.sigmoid(pred[0])  # [H, W]
    classes = torch.sigmoid(pred[5:])  # [C, H, W]
    
    for gy in range(H):
        for gx in range(W):
            obj_conf = obj[gy, gx].item()
            if obj_conf < conf_thresh:
                continue
            
            cls_scores = classes[:, gy, gx]
            cls_conf, cls_id = cls_scores.max(0)
            cls_conf = cls_conf.item()
            cls_id = cls_id.item()
            
            final_conf = obj_conf * cls_conf
            if final_conf < conf_thresh:
                continue
            
            x_off = torch.sigmoid(pred[1, gy, gx]).item()
            y_off = torch.sigmoid(pred[2, gy, gx]).item()
            w = torch.sigmoid(pred[3, gy, gx]).item()
            h = torch.sigmoid(pred[4, gy, gx]).item()
            
            cx = (gx + x_off) / W
            cy = (gy + y_off) / H
            
            detections.append({
                'conf': final_conf,
                'class': cls_id,
                'cx': cx, 'cy': cy, 'w': w, 'h': h
            })
    
    # NMS per class
    final_dets = []
    for cls_id in range(NUM_CLASSES):
        cls_dets = [d for d in detections if d['class'] == cls_id]
        cls_dets.sort(key=lambda x: x['conf'], reverse=True)
        
        keep = []
        while cls_dets:
            best = cls_dets.pop(0)
            keep.append(best)
            remaining = []
            for d in cls_dets:
                iou = compute_iou(
                    [best['cx'] - best['w']/2, best['cy'] - best['h']/2,
                     best['cx'] + best['w']/2, best['cy'] + best['h']/2],
                    [d['cx'] - d['w']/2, d['cy'] - d['h']/2,
                     d['cx'] + d['w']/2, d['cy'] + d['h']/2]
                )
                if iou < nms_thresh:
                    remaining.append(d)
            cls_dets = remaining
        final_dets.extend(keep)
    
    return final_dets


def draw_detections(img, detections, orig_size=None):
    """Draw bounding boxes on image"""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    for det in detections:
        x1 = int((det['cx'] - det['w']/2) * w)
        y1 = int((det['cy'] - det['h']/2) * h)
        x2 = int((det['cx'] + det['w']/2) * w)
        y2 = int((det['cy'] + det['h']/2) * h)
        
        color = CLASS_COLORS[det['class']]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        label = f"{CLASS_NAMES[det['class']]} {det['conf']:.2f}"
        bbox = draw.textbbox((x1, y1-15), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1-15), label, fill=(255, 255, 255), font=font)
    
    return img


def run_inference(model, img_path, device, conf_thresh=0.25):
    """Run inference on a single image"""
    img = Image.open(img_path).convert('RGB')
    orig_size = img.size
    
    # Resize for model
    img_resized = img.resize((IMG_W, IMG_H), Image.BILINEAR)
    
    # Preprocess
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = (img_np - mean) / std
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    # Inference
    with torch.no_grad():
        output = model(img_tensor)
    
    # Decode
    detections = decode_predictions(output[0].cpu(), conf_thresh=conf_thresh)
    
    # Draw on resized image
    result = draw_detections(img_resized.copy(), detections)
    
    return result, detections, img_resized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model weights')
    parser.add_argument('--images', type=str, nargs='+', required=True, help='Image paths')
    parser.add_argument('--output', type=str, default='detection_results', help='Output directory')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--num-classes', type=int, default=4, help='Number of classes')
    args = parser.parse_args()
    
    global NUM_CLASSES
    NUM_CLASSES = args.num_classes
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device}")
    
    # Load model
    model = TinyDet(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"Loaded model from {args.model}")
    
    # Create output directory
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Process images
    for img_path in args.images:
        print(f"\nProcessing {img_path}...")
        result, detections, _ = run_inference(model, img_path, device, args.conf)
        
        print(f"  Found {len(detections)} detections:")
        for det in detections:
            print(f"    {CLASS_NAMES[det['class']]}: {det['conf']:.3f} at ({det['cx']:.2f}, {det['cy']:.2f})")
        
        out_path = out_dir / f"{Path(img_path).stem}_detected.jpg"
        result.save(out_path)
        print(f"  Saved to {out_path}")


if __name__ == '__main__':
    main()

