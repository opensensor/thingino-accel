#!/usr/bin/env python3
"""
Improved training script for 4-class detection model.
Addresses issues in previous training:
1. Better data augmentation (multi-scale, mosaic, color)
2. Improved loss functions (CIoU, focal with proper weighting)
3. Proper evaluation metrics (mAP, recall, precision)
4. Visual debugging during training
"""
import os
import sys
import json
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Import model
sys.path.insert(0, str(Path(__file__).parent))
from tinydet import TinyDet

# Configuration
BATCH_SIZE = 32
EPOCHS = 100  # Fine-tuning doesn't need as many epochs
LR = 2e-4  # Lower LR for fine-tuning (was 1e-3)
IMG_W, IMG_H = 320, 192  # W, H
GRID_W, GRID_H = 20, 12  # W, H (stride 16)
NUM_CLASSES = 4
CLASS_NAMES = ['person', 'vehicle', 'cat', 'dog']

# Training settings
WARMUP_EPOCHS = 5
NEG_POS_RATIO = 100.0  # Much higher ratio - most cells should be negative
LABEL_SMOOTHING = 0.02
OBJ_LOSS_WEIGHT = 5.0  # Higher weight on objectness to suppress false positives
BOX_LOSS_WEIGHT = 3.0  # Prioritize bounding box accuracy over classification
CLS_LOSS_WEIGHT = 0.5  # Lower classification weight - localization matters more


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


def compute_ciou_loss(pred_box, target_box, eps=1e-7):
    """
    Compute CIoU loss for box regression.
    pred_box, target_box: [N, 4] in format [cx, cy, w, h] normalized
    """
    # Convert to x1,y1,x2,y2
    pred_x1 = pred_box[:, 0] - pred_box[:, 2] / 2
    pred_y1 = pred_box[:, 1] - pred_box[:, 3] / 2
    pred_x2 = pred_box[:, 0] + pred_box[:, 2] / 2
    pred_y2 = pred_box[:, 1] + pred_box[:, 3] / 2

    target_x1 = target_box[:, 0] - target_box[:, 2] / 2
    target_y1 = target_box[:, 1] - target_box[:, 3] / 2
    target_x2 = target_box[:, 0] + target_box[:, 2] / 2
    target_y2 = target_box[:, 1] + target_box[:, 3] / 2

    # Intersection
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    # Union
    pred_area = pred_box[:, 2] * pred_box[:, 3]
    target_area = target_box[:, 2] * target_box[:, 3]
    union = pred_area + target_area - inter + eps

    iou = inter / union

    # Enclosing box
    enc_x1 = torch.min(pred_x1, target_x1)
    enc_y1 = torch.min(pred_y1, target_y1)
    enc_x2 = torch.max(pred_x2, target_x2)
    enc_y2 = torch.max(pred_y2, target_y2)

    # Distance term
    c_x = (pred_box[:, 0] + target_box[:, 0]) / 2 - (enc_x1 + enc_x2) / 2
    c_y = (pred_box[:, 1] + target_box[:, 1]) / 2 - (enc_y1 + enc_y2) / 2
    rho2 = c_x ** 2 + c_y ** 2
    c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps

    # Aspect ratio term
    v = (4 / (math.pi ** 2)) * torch.pow(
        torch.atan(target_box[:, 2] / (target_box[:, 3] + eps)) -
        torch.atan(pred_box[:, 2] / (pred_box[:, 3] + eps)), 2
    )
    alpha = v / (1 - iou + v + eps)

    ciou = iou - rho2 / c2 - alpha * v
    return (1 - ciou).mean()


class FocalLoss(nn.Module):
    """Quality-aware focal loss with class weighting"""
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Per-class weights
        self.reduction = reduction

    def forward(self, inputs, targets, weights=None):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = (1 - pt) ** self.gamma * bce

        if weights is not None:
            focal = focal * weights

        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal


class ImprovedDetectionDataset(Dataset):
    """Dataset with improved augmentation pipeline"""

    def __init__(self, ann_file, img_dirs, img_size=(192, 320), augment=True, mosaic_prob=0.3):
        with open(ann_file) as f:
            data = json.load(f)

        self.images = {img['id']: img for img in data['images']}
        self.img_size = img_size  # H, W
        self.augment = augment
        self.mosaic_prob = mosaic_prob if augment else 0
        self.img_dirs = img_dirs if isinstance(img_dirs, list) else [img_dirs]

        # Group annotations by image
        self.img_anns = defaultdict(list)
        for ann in data['annotations']:
            self.img_anns[ann['image_id']].append(ann)

        self.img_ids = list(self.img_anns.keys())

        # Compute class distribution
        self.class_counts = [0] * NUM_CLASSES
        for anns in self.img_anns.values():
            for ann in anns:
                self.class_counts[ann['category_id']] += 1

        total = sum(self.class_counts) + 1e-6
        # Class weights: higher weight for rare classes
        self.class_weights = torch.tensor([
            (total / (NUM_CLASSES * (c + 1))) ** 0.5 for c in self.class_counts
        ])
        print(f"  Class distribution: {dict(zip(CLASS_NAMES, self.class_counts))}")
        print(f"  Class weights: {[f'{w:.2f}' for w in self.class_weights.tolist()]}")

    def __len__(self):
        return len(self.img_ids)

    def find_image(self, filename):
        for img_dir in self.img_dirs:
            path = Path(img_dir) / filename
            if path.exists():
                return path
        return None

    def load_image_and_boxes(self, idx):
        """Load image and bounding boxes"""
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        img_path = self.find_image(img_info['file_name'])

        if img_path is None:
            return None, []

        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size

        boxes = []
        for ann in self.img_anns[img_id]:
            x, y, w, h = ann['bbox']
            # Normalize to 0-1
            boxes.append({
                'cx': (x + w/2) / orig_w,
                'cy': (y + h/2) / orig_h,
                'w': w / orig_w,
                'h': h / orig_h,
                'class': ann['category_id']
            })

        return img, boxes

    def apply_augmentations(self, img, boxes):
        """Apply augmentation pipeline"""
        if not self.augment:
            return img, boxes

        # Random horizontal flip
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            boxes = [{'cx': 1 - b['cx'], 'cy': b['cy'], 'w': b['w'], 'h': b['h'], 'class': b['class']} for b in boxes]

        # Random scale (0.5 to 1.5)
        if random.random() > 0.5:
            scale = random.uniform(0.7, 1.3)
            boxes = [{'cx': b['cx'], 'cy': b['cy'], 'w': b['w'] * scale, 'h': b['h'] * scale, 'class': b['class']} for b in boxes]

        # Color augmentation
        if random.random() > 0.3:
            # Brightness
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.4))
        if random.random() > 0.3:
            # Contrast
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.6, 1.4))
        if random.random() > 0.3:
            # Saturation
            img = ImageEnhance.Color(img).enhance(random.uniform(0.5, 1.5))

        # Random blur
        if random.random() > 0.9:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        # Clip boxes to valid range
        valid_boxes = []
        for b in boxes:
            cx, cy, w, h = b['cx'], b['cy'], b['w'], b['h']
            # Ensure box stays within image
            x1, y1 = max(0, cx - w/2), max(0, cy - h/2)
            x2, y2 = min(1, cx + w/2), min(1, cy + h/2)
            new_w, new_h = x2 - x1, y2 - y1
            # Keep boxes that are at least 3% of image size
            if new_w > 0.03 and new_h > 0.03:
                valid_boxes.append({
                    'cx': (x1 + x2) / 2,
                    'cy': (y1 + y2) / 2,
                    'w': new_w,
                    'h': new_h,
                    'class': b['class']
                })

        return img, valid_boxes

    def create_mosaic(self, indices):
        """Create 4-image mosaic augmentation"""
        mosaic_size = (self.img_size[0] * 2, self.img_size[1] * 2)  # H, W
        mosaic_img = Image.new('RGB', (mosaic_size[1], mosaic_size[0]), (114, 114, 114))

        all_boxes = []
        positions = [
            (0, 0),
            (self.img_size[1], 0),
            (0, self.img_size[0]),
            (self.img_size[1], self.img_size[0])
        ]

        for pos_idx, idx in enumerate(indices):
            img, boxes = self.load_image_and_boxes(idx)
            if img is None:
                continue

            # Resize to target size
            img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
            x_off, y_off = positions[pos_idx]
            mosaic_img.paste(img, (x_off, y_off))

            # Adjust box coordinates
            for b in boxes:
                new_cx = (b['cx'] * self.img_size[1] + x_off) / mosaic_size[1]
                new_cy = (b['cy'] * self.img_size[0] + y_off) / mosaic_size[0]
                new_w = b['w'] * self.img_size[1] / mosaic_size[1]
                new_h = b['h'] * self.img_size[0] / mosaic_size[0]
                all_boxes.append({'cx': new_cx, 'cy': new_cy, 'w': new_w, 'h': new_h, 'class': b['class']})

        # Random crop back to original size
        crop_x = random.randint(0, self.img_size[1])
        crop_y = random.randint(0, self.img_size[0])
        mosaic_img = mosaic_img.crop((crop_x, crop_y, crop_x + self.img_size[1], crop_y + self.img_size[0]))

        # Adjust boxes for crop
        final_boxes = []
        for b in all_boxes:
            # Convert to pixels in mosaic
            cx_px = b['cx'] * mosaic_size[1] - crop_x
            cy_px = b['cy'] * mosaic_size[0] - crop_y
            w_px = b['w'] * mosaic_size[1]
            h_px = b['h'] * mosaic_size[0]

            # Convert back to normalized coords in cropped image
            cx = cx_px / self.img_size[1]
            cy = cy_px / self.img_size[0]
            w = w_px / self.img_size[1]
            h = h_px / self.img_size[0]

            # Clip to valid range
            x1, y1 = max(0, cx - w/2), max(0, cy - h/2)
            x2, y2 = min(1, cx + w/2), min(1, cy + h/2)
            if x2 > x1 + 0.03 and y2 > y1 + 0.03:
                final_boxes.append({
                    'cx': (x1 + x2) / 2,
                    'cy': (y1 + y2) / 2,
                    'w': x2 - x1,
                    'h': y2 - y1,
                    'class': b['class']
                })

        return mosaic_img, final_boxes

    def create_target_grid(self, boxes):
        """Create target tensor from boxes"""
        target = torch.zeros(5 + NUM_CLASSES, GRID_H, GRID_W)

        for b in boxes:
            # Find grid cell
            gx = int(b['cx'] * GRID_W)
            gy = int(b['cy'] * GRID_H)
            gx = min(max(gx, 0), GRID_W - 1)
            gy = min(max(gy, 0), GRID_H - 1)

            # If cell already has an object, check IoU and keep better one
            if target[0, gy, gx] > 0:
                # Calculate IoU with existing box
                existing_cx = (gx + target[1, gy, gx]) / GRID_W
                existing_cy = (gy + target[2, gy, gx]) / GRID_H
                existing_w = target[3, gy, gx]
                existing_h = target[4, gy, gx]

                # Keep the larger box (more important to detect)
                existing_area = existing_w * existing_h
                new_area = b['w'] * b['h']
                if new_area < existing_area:
                    continue

            # Set target
            target[0, gy, gx] = 1.0  # objectness
            target[1, gy, gx] = b['cx'] * GRID_W - gx  # x offset (0-1)
            target[2, gy, gx] = b['cy'] * GRID_H - gy  # y offset (0-1)
            target[3, gy, gx] = b['w']  # normalized width
            target[4, gy, gx] = b['h']  # normalized height
            target[5 + b['class'], gy, gx] = 1.0  # class

        return target

    def __getitem__(self, idx):
        # Mosaic augmentation
        if self.augment and random.random() < self.mosaic_prob:
            indices = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]
            img, boxes = self.create_mosaic(indices)
            img, boxes = self.apply_augmentations(img, boxes)
        else:
            img, boxes = self.load_image_and_boxes(idx)
            if img is None:
                return torch.zeros(3, *self.img_size), torch.zeros(5 + NUM_CLASSES, GRID_H, GRID_W)
            img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
            img, boxes = self.apply_augmentations(img, boxes)

        # Convert to tensor
        img_np = np.array(img).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = (img_np - mean) / std
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()

        # Create target
        target = self.create_target_grid(boxes)

        return img_tensor, target


class DetectionLoss(nn.Module):
    """Combined detection loss with proper weighting"""

    def __init__(self, class_weights, device):
        super().__init__()
        self.focal = FocalLoss(gamma=2.0)
        self.class_weights = class_weights.to(device)
        self.device = device

    def forward(self, pred, target, epoch=0):
        """
        pred: [B, 5+C, H, W] - raw logits
        target: [B, 5+C, H, W] - ground truth
        """
        pos_mask = target[:, 0] > 0.5  # [B, H, W]
        neg_mask = ~pos_mask
        num_pos = pos_mask.sum().item()

        if num_pos == 0:
            # No objects - penalize ALL cells with high objectness
            obj_pred = pred[:, 0][neg_mask]
            obj_target = target[:, 0][neg_mask]
            obj_loss = F.binary_cross_entropy_with_logits(obj_pred, obj_target)
            return obj_loss, torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        # OBJECTNESS LOSS - Use ALL cells to learn proper objectness
        # This is crucial - the model must learn that most cells are background
        all_obj_pred = pred[:, 0].flatten()
        all_obj_target = target[:, 0].flatten()

        # Weight positive samples more since they're rare
        pos_weight = neg_mask.sum().float() / (pos_mask.sum().float() + 1e-6)
        pos_weight = torch.clamp(pos_weight, 1.0, 50.0)  # Cap at 50x

        # BCE with positive weighting
        obj_loss = F.binary_cross_entropy_with_logits(
            all_obj_pred, all_obj_target,
            pos_weight=pos_weight
        )

        # BOX LOSS using CIoU
        # Get grid coordinates for positive cells
        pos_indices = pos_mask.nonzero()  # [N, 3] - batch, y, x

        # Predicted boxes (apply sigmoid to offsets)
        pred_offset_x = torch.sigmoid(pred[:, 1][pos_mask])
        pred_offset_y = torch.sigmoid(pred[:, 2][pos_mask])
        pred_w = torch.sigmoid(pred[:, 3][pos_mask])
        pred_h = torch.sigmoid(pred[:, 4][pos_mask])

        # Convert to center coords
        gx = pos_indices[:, 2].float()
        gy = pos_indices[:, 1].float()
        pred_cx = (gx + pred_offset_x) / GRID_W
        pred_cy = (gy + pred_offset_y) / GRID_H

        pred_boxes = torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=1)

        # Target boxes
        target_offset_x = target[:, 1][pos_mask]
        target_offset_y = target[:, 2][pos_mask]
        target_w = target[:, 3][pos_mask]
        target_h = target[:, 4][pos_mask]
        target_cx = (gx + target_offset_x) / GRID_W
        target_cy = (gy + target_offset_y) / GRID_H

        target_boxes = torch.stack([target_cx, target_cy, target_w, target_h], dim=1)

        box_loss = compute_ciou_loss(pred_boxes, target_boxes)

        # CLASS LOSS with label smoothing and weighting
        class_pred = pred[:, 5:].permute(0, 2, 3, 1)[pos_mask]  # [N, C]
        class_target = target[:, 5:].permute(0, 2, 3, 1)[pos_mask]  # [N, C]

        # Label smoothing
        class_target = class_target * (1 - LABEL_SMOOTHING) + LABEL_SMOOTHING / NUM_CLASSES

        # Weighted BCE per class
        class_loss = torch.tensor(0.0, device=self.device)
        for c in range(NUM_CLASSES):
            c_loss = F.binary_cross_entropy_with_logits(class_pred[:, c], class_target[:, c])
            class_loss = class_loss + self.class_weights[c] * c_loss
        class_loss = class_loss / self.class_weights.sum()

        return obj_loss, box_loss, class_loss



def decode_predictions(pred, conf_thresh=0.25, nms_thresh=0.45):
    """Decode model predictions to bounding boxes"""
    B, C, H, W = pred.shape
    detections = []

    for b in range(B):
        batch_dets = []
        obj = torch.sigmoid(pred[b, 0])  # [H, W]
        classes = torch.sigmoid(pred[b, 5:])  # [C, H, W]

        for gy in range(H):
            for gx in range(W):
                obj_conf = obj[gy, gx].item()
                if obj_conf < conf_thresh:
                    continue

                # Get class scores
                cls_scores = classes[:, gy, gx]
                cls_conf, cls_id = cls_scores.max(0)
                cls_conf = cls_conf.item()
                cls_id = cls_id.item()

                final_conf = obj_conf * cls_conf
                if final_conf < conf_thresh:
                    continue

                # Decode box
                x_off = torch.sigmoid(pred[b, 1, gy, gx]).item()
                y_off = torch.sigmoid(pred[b, 2, gy, gx]).item()
                w = torch.sigmoid(pred[b, 3, gy, gx]).item()
                h = torch.sigmoid(pred[b, 4, gy, gx]).item()

                cx = (gx + x_off) / W
                cy = (gy + y_off) / H

                batch_dets.append({
                    'conf': final_conf,
                    'class': cls_id,
                    'cx': cx, 'cy': cy, 'w': w, 'h': h
                })

        # NMS per class
        final_dets = []
        for cls_id in range(NUM_CLASSES):
            cls_dets = [d for d in batch_dets if d['class'] == cls_id]
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

        detections.append(final_dets)

    return detections


def compute_metrics(predictions, targets, iou_thresh=0.5):
    """Compute precision, recall, and mAP"""
    all_tp = defaultdict(list)
    all_fp = defaultdict(list)
    all_gt = defaultdict(int)

    for pred_batch, target in zip(predictions, targets):
        # Get ground truth boxes
        gt_boxes = []
        pos_mask = target[0] > 0.5
        for gy, gx in pos_mask.nonzero().tolist():
            cx = (gx + target[1, gy, gx].item()) / GRID_W
            cy = (gy + target[2, gy, gx].item()) / GRID_H
            w = target[3, gy, gx].item()
            h = target[4, gy, gx].item()
            cls_id = target[5:, gy, gx].argmax().item()
            gt_boxes.append({'cx': cx, 'cy': cy, 'w': w, 'h': h, 'class': cls_id, 'matched': False})
            all_gt[cls_id] += 1

        # Match predictions to GT
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

            if best_iou >= iou_thresh and best_gt is not None:
                all_tp[pred['class']].append((pred['conf'], 1))
                best_gt['matched'] = True
            else:
                all_fp[pred['class']].append((pred['conf'], 1))

    # Compute AP per class
    aps = {}
    for c in range(NUM_CLASSES):
        if all_gt[c] == 0:
            aps[c] = 0.0
            continue

        # Sort by confidence
        tp_list = sorted(all_tp[c], key=lambda x: x[0], reverse=True)
        fp_list = sorted(all_fp[c], key=lambda x: x[0], reverse=True)

        # Merge and sort
        all_preds = [(conf, 1, 0) for conf, _ in tp_list] + [(conf, 0, 1) for conf, _ in fp_list]
        all_preds.sort(key=lambda x: x[0], reverse=True)

        # Compute precision-recall curve
        tp_cumsum = 0
        fp_cumsum = 0
        precisions = []
        recalls = []

        for conf, tp, fp in all_preds:
            tp_cumsum += tp
            fp_cumsum += fp
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
            recall = tp_cumsum / (all_gt[c] + 1e-6)
            precisions.append(precision)
            recalls.append(recall)

        # AP using 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            p = 0
            for prec, rec in zip(precisions, recalls):
                if rec >= t:
                    p = max(p, prec)
            ap += p / 11

        aps[c] = ap

    mAP = np.mean(list(aps.values()))
    return mAP, aps


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device}")

    # Datasets - use balanced_4class with symlinks to combined_dataset/images
    train_ds = ImprovedDetectionDataset(
        'balanced_4class/annotations/instances_train.json',
        ['balanced_4class/images', 'combined_dataset/images', 'train2017'],
        img_size=(IMG_H, IMG_W),
        augment=True,
        mosaic_prob=0.3
    )
    val_ds = ImprovedDetectionDataset(
        'balanced_4class/annotations/instances_val_balanced.json',
        ['balanced_4class/val2017', 'coco/val2017', 'val2017', 'oxford_pets_coco/images'],
        img_size=(IMG_H, IMG_W),
        augment=False,
        mosaic_prob=0
    )

    print(f"Train: {len(train_ds)} images")
    print(f"Val: {len(val_ds)} images")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Model
    model = TinyDet(num_classes=NUM_CLASSES).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    # Loss
    criterion = DetectionLoss(train_ds.class_weights, device)

    # Optimizer with warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # Warmup + cosine annealing
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        else:
            return 0.5 * (1 + math.cos(math.pi * (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Output directory
    out_dir = Path('runs/localization_focus')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try to load pretrained weights from best previous model
    pretrained_path = Path('runs/improved_v2/tinydet_best.pth')
    if pretrained_path.exists():
        print(f"Loading pretrained weights from {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=True))
        print("  Loaded successfully - fine-tuning from improved_v2")

    best_mAP = 0.0
    best_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_obj_loss = 0.0
        train_box_loss = 0.0
        train_cls_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{EPOCHS}')
        for imgs, targets in pbar:
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)

            obj_loss, box_loss, cls_loss = criterion(outputs, targets, epoch)
            # Prioritize localization: 5*obj + 3*box + 0.5*cls
            loss = OBJ_LOSS_WEIGHT * obj_loss + BOX_LOSS_WEIGHT * box_loss + CLS_LOSS_WEIGHT * cls_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            train_loss += loss.item()
            train_obj_loss += obj_loss.item()
            train_box_loss += box_loss.item()
            train_cls_loss += cls_loss.item()

            pbar.set_postfix(loss=f'{loss.item():.4f}')

        train_loss /= len(train_loader)
        train_obj_loss /= len(train_loader)
        train_box_loss /= len(train_loader)
        train_cls_loss /= len(train_loader)

        scheduler.step()

        # Validation with mAP
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = model(imgs)

                obj_loss, box_loss, cls_loss = criterion(outputs, targets, epoch)
                val_loss += (OBJ_LOSS_WEIGHT * obj_loss + BOX_LOSS_WEIGHT * box_loss + CLS_LOSS_WEIGHT * cls_loss).item()

                # Decode predictions for mAP
                preds = decode_predictions(outputs, conf_thresh=0.1)
                all_preds.extend(preds)
                all_targets.extend([t.cpu() for t in targets])

        val_loss /= len(val_loader)
        mAP, class_aps = compute_metrics(all_preds, all_targets)

        lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch}: loss={train_loss:.4f} (obj={train_obj_loss:.3f}, box={train_box_loss:.3f}, cls={train_cls_loss:.3f})')
        print(f'         val_loss={val_loss:.4f}, mAP@0.5={mAP:.3f}, lr={lr:.6f}')
        print(f'         APs: ' + ', '.join([f'{CLASS_NAMES[c]}={class_aps[c]:.3f}' for c in range(NUM_CLASSES)]))

        # Save checkpoints
        if epoch % 10 == 0:
            torch.save(model.state_dict(), out_dir / f'tinydet_e{epoch}.pth')

        if mAP > best_mAP:
            best_mAP = mAP
            torch.save(model.state_dict(), out_dir / 'tinydet_best_mAP.pth')
            print(f'  New best mAP: {mAP:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), out_dir / 'tinydet_best.pth')
            print(f'  New best loss: {val_loss:.4f}')

    torch.save(model.state_dict(), out_dir / 'tinydet_final.pth')
    print(f'Training complete. Best mAP: {best_mAP:.4f}, Best loss: {best_loss:.4f}')


if __name__ == '__main__':
    train()