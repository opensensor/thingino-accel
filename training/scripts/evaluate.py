#!/usr/bin/env python3
"""
TinyDet Evaluation Script

Computes mAP, precision, recall, and per-class metrics.

Usage:
  python scripts/evaluate.py --config params.yaml --weights runs/current/tinydet_best.pth
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from tinydet import TinyDet

# Import dataset from scripts/train.py
from train import DetectionDataset, load_config


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
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
    """Decode model predictions to bounding boxes."""
    B, C, H, W = pred.shape
    num_classes = C - 5
    detections = []
    
    for b in range(B):
        batch_dets = []
        obj = torch.sigmoid(pred[b, 0])
        classes = torch.sigmoid(pred[b, 5:])
        
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
                
                x_off = torch.sigmoid(pred[b, 1, gy, gx]).item()
                y_off = torch.sigmoid(pred[b, 2, gy, gx]).item()
                w = torch.sigmoid(pred[b, 3, gy, gx]).item()
                h = torch.sigmoid(pred[b, 4, gy, gx]).item()
                
                cx = (gx + x_off) / W
                cy = (gy + y_off) / H
                
                batch_dets.append({
                    'conf': final_conf, 'class': cls_id,
                    'cx': cx, 'cy': cy, 'w': w, 'h': h
                })
        
        # NMS per class
        final_dets = []
        for cls_id in range(num_classes):
            cls_dets = sorted([d for d in batch_dets if d['class'] == cls_id],
                            key=lambda x: x['conf'], reverse=True)
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
        
        detections.append(final_dets)
    
    return detections


def decode_targets(target, grid_h, grid_w, num_classes):
    """Decode target tensor to boxes."""
    boxes = []
    pos_mask = target[0] > 0.5
    
    for pos in pos_mask.nonzero().tolist():
        gy, gx = pos
        cx = (gx + target[1, gy, gx].item()) / grid_w
        cy = (gy + target[2, gy, gx].item()) / grid_h
        w = target[3, gy, gx].item()
        h = target[4, gy, gx].item()
        cls_id = target[5:, gy, gx].argmax().item()
        boxes.append({'cx': cx, 'cy': cy, 'w': w, 'h': h, 'class': cls_id})
    
    return boxes


def compute_ap(tp_list, fp_list, num_gt, num_points=11):
    """Compute Average Precision using interpolation."""
    if num_gt == 0:
        return 0.0, [], []

    if len(tp_list) == 0 and len(fp_list) == 0:
        return 0.0, [], []

    all_preds = [(conf, 1, 0) for conf, _ in tp_list] + [(conf, 0, 1) for conf, _ in fp_list]
    all_preds.sort(key=lambda x: x[0], reverse=True)
    
    tp_cumsum = 0
    fp_cumsum = 0
    precisions = []
    recalls = []
    
    for conf, tp, fp in all_preds:
        tp_cumsum += tp
        fp_cumsum += fp
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recall = tp_cumsum / (num_gt + 1e-6)
        precisions.append(precision)
        recalls.append(recall)
    
    # 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, num_points):
        p = 0
        for prec, rec in zip(precisions, recalls):
            if rec >= t:
                p = max(p, prec)
        ap += p / num_points

    return ap, precisions, recalls


def evaluate(config: dict, weights_path: str):
    """Run evaluation on validation set."""
    model_cfg = config.get('model', {})
    eval_cfg = config.get('evaluate', {})
    prep_cfg = config.get('prepare', {})
    class_names = config.get('classes', ['person', 'vehicle', 'cat', 'dog'])

    num_classes = model_cfg.get('num_classes', 4)
    img_h = model_cfg.get('input_height', 192)
    img_w = model_cfg.get('input_width', 320)
    grid_h = model_cfg.get('grid_height', 12)
    grid_w = model_cfg.get('grid_width', 20)

    conf_thresh = eval_cfg.get('conf_threshold', 0.25)
    nms_thresh = eval_cfg.get('nms_threshold', 0.45)
    iou_thresh = eval_cfg.get('iou_threshold', 0.5)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    print(f"Loading weights from {weights_path}")
    model = TinyDet(num_classes=num_classes).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Dataset
    data_dir = Path(prep_cfg.get('output_dir', 'datasets/prepared'))
    val_ann = data_dir / 'annotations' / 'instances_val.json'
    img_dirs = [
        data_dir / 'images',
        'coco/train2017',
        'coco/val2017',
        'oxford_pets/images',
    ]

    val_ds = DetectionDataset(str(val_ann), img_dirs, (img_h, img_w), num_classes, augment=False)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

    print(f"Evaluating on {len(val_ds)} images...")

    # Collect predictions and ground truth
    all_tp = defaultdict(list)
    all_fp = defaultdict(list)
    all_gt = defaultdict(int)

    with torch.no_grad():
        for imgs, targets in tqdm(val_loader, desc='Evaluating'):
            imgs = imgs.to(device)
            outputs = model(imgs)

            preds = decode_predictions(outputs, conf_thresh, nms_thresh)

            for i, (pred_batch, target) in enumerate(zip(preds, targets)):
                gt_boxes = decode_targets(target, grid_h, grid_w, num_classes)

                for gt in gt_boxes:
                    all_gt[gt['class']] += 1
                    gt['matched'] = False

                for pred in pred_batch:
                    best_iou = 0
                    best_gt = None

                    for gt in gt_boxes:
                        if gt['class'] != pred['class'] or gt['matched']:
                            continue
                        iou = compute_iou(
                            [pred['cx'] - pred['w']/2, pred['cy'] - pred['h']/2,
                             pred['cx'] + pred['w']/2, pred['cy'] + pred['h']/2],
                            [gt['cx'] - gt['w']/2, gt['cy'] - gt['h']/2,
                             gt['cx'] + gt['w']/2, gt['cy'] + gt['h']/2]
                        )
                        if iou > best_iou:
                            best_iou = iou
                            best_gt = gt

                    if best_iou >= iou_thresh and best_gt:
                        all_tp[pred['class']].append((pred['conf'], 1))
                        best_gt['matched'] = True
                    else:
                        all_fp[pred['class']].append((pred['conf'], 1))

    # Compute metrics per class
    results = {'classes': {}, 'mAP': 0.0}
    all_precisions = {}
    all_recalls = {}

    for cls_id in range(num_classes):
        name = class_names[cls_id] if cls_id < len(class_names) else f'class_{cls_id}'
        ap, precs, recs = compute_ap(all_tp[cls_id], all_fp[cls_id], all_gt[cls_id])

        tp_count = len(all_tp[cls_id])
        fp_count = len(all_fp[cls_id])
        gt_count = all_gt[cls_id]

        precision = tp_count / (tp_count + fp_count + 1e-6)
        recall = tp_count / (gt_count + 1e-6)

        results['classes'][name] = {
            'AP': ap, 'precision': precision, 'recall': recall,
            'TP': tp_count, 'FP': fp_count, 'GT': gt_count
        }
        all_precisions[name] = precs
        all_recalls[name] = recs

        print(f"  {name}: AP={ap:.3f}, P={precision:.3f}, R={recall:.3f} (TP={tp_count}, FP={fp_count}, GT={gt_count})")

    results['mAP'] = np.mean([r['AP'] for r in results['classes'].values()])
    print(f"\nmAP@{iou_thresh}: {results['mAP']:.4f}")

    # Save metrics
    metrics_dir = Path('metrics')
    metrics_dir.mkdir(exist_ok=True)

    with open(metrics_dir / 'eval_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save PR curve CSV for DVC plots
    with open(metrics_dir / 'pr_curve.csv', 'w') as f:
        f.write('class,precision,recall\n')
        for name in class_names[:num_classes]:
            if name in all_precisions:
                for p, r in zip(all_precisions[name], all_recalls[name]):
                    f.write(f'{name},{p:.6f},{r:.6f}\n')

    print(f"\nMetrics saved to {metrics_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate TinyDet')
    parser.add_argument('--config', type=str, default='params.yaml')
    parser.add_argument('--weights', type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    evaluate(config, args.weights)


if __name__ == '__main__':
    main()
