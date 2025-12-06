"""
TinyDet Training Script

Train the TinyDet model on filtered COCO (person/dog/cat).
Supports multi-GPU training with DataParallel.

Usage:
  # Single GPU
  python train.py --data ./coco_3class --epochs 100 --batch-size 64

  # Multi-GPU (uses all available)
  python train.py --data ./coco_3class --epochs 100 --batch-size 128 --multi-gpu

  # Specific GPUs
  CUDA_VISIBLE_DEVICES=0,1 python train.py --data ./coco_3class --batch-size 128 --multi-gpu
"""

import os
import argparse
import json
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageEnhance
import numpy as np
from tqdm import tqdm

from tinydet import TinyDet, INPUT_W, INPUT_H, GRID_W, GRID_H, STRIDE


# Constants
NUM_CLASSES = 3
CLASS_NAMES = ['person', 'cat', 'dog']
# Class weights: stronger weighting for rare classes
# COCO: person=262K, cat=4.7K, dog=5.5K annotations
# Use much stronger weights to overcome imbalance
CLASS_WEIGHTS = torch.tensor([1.0, 55.0, 48.0])  # person, cat, dog (inverse frequency)


def augment_image(image, anns, orig_w, orig_h, p_flip=0.5, p_color=0.3):
    """Apply data augmentation to image and adjust bboxes accordingly."""
    augmented_anns = []

    # Horizontal flip
    flip = random.random() < p_flip
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # Color jitter (brightness, contrast, saturation)
    if random.random() < p_color:
        factor = random.uniform(0.8, 1.2)
        image = ImageEnhance.Brightness(image).enhance(factor)
    if random.random() < p_color:
        factor = random.uniform(0.8, 1.2)
        image = ImageEnhance.Contrast(image).enhance(factor)
    if random.random() < p_color:
        factor = random.uniform(0.9, 1.1)
        image = ImageEnhance.Color(image).enhance(factor)

    # Adjust bboxes for flip
    for ann in anns:
        bbox = ann['bbox'].copy() if isinstance(ann['bbox'], list) else list(ann['bbox'])
        if flip:
            # Flip x coordinate: new_x = orig_w - x - w
            bbox[0] = orig_w - bbox[0] - bbox[2]
        augmented_anns.append({'bbox': bbox, 'category_id': ann['category_id']})

    return image, augmented_anns


class COCODetectionDataset(Dataset):
    """COCO detection dataset with augmentation for TinyDet training."""

    def __init__(self, data_dir, split='train2017', transform=None, augment=False):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.augment = augment

        # Load annotations
        ann_file = self.data_dir / 'annotations' / f'instances_{split}.json'
        with open(ann_file, 'r') as f:
            data = json.load(f)

        self.images = {img['id']: img for img in data['images']}

        # Group annotations by image and track classes per image
        self.annotations = {}
        self.image_classes = {}  # Track which classes each image contains
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
                self.image_classes[img_id] = set()
            self.annotations[img_id].append(ann)
            self.image_classes[img_id].add(ann['category_id'])

        self.image_ids = list(self.annotations.keys())

        # Compute class distribution for sampling weights
        self._compute_class_stats()
        print(f"Loaded {len(self.image_ids)} images from {split}")

    def _compute_class_stats(self):
        """Compute per-image sampling weights to balance classes."""
        # Count images per class
        class_counts = {0: 0, 1: 0, 2: 0}  # person, cat, dog
        for img_id, classes in self.image_classes.items():
            for c in classes:
                class_counts[c] = class_counts.get(c, 0) + 1

        print(f"  Class distribution: person={class_counts[0]}, cat={class_counts[1]}, dog={class_counts[2]}")

        # Compute sampling weights: rare classes get higher weight
        max_count = max(class_counts.values())
        self.sample_weights = []
        for img_id in self.image_ids:
            classes = self.image_classes[img_id]
            # Weight is max of weights for all classes in the image
            # This ensures cat/dog images are sampled more often
            weight = max(max_count / class_counts.get(c, max_count) for c in classes)
            self.sample_weights.append(weight)

    def get_sampler(self):
        """Return a weighted sampler that oversamples rare classes."""
        from torch.utils.data import WeightedRandomSampler
        return WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self.sample_weights),
            replacement=True
        )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        anns = self.annotations[img_id]

        # Load image
        img_path = self.data_dir / self.split / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size

        # Apply augmentation (training only)
        if self.augment:
            image, anns = augment_image(image, anns, orig_w, orig_h)

        # Resize to INPUT_W x INPUT_H (320x192, close to 16:9)
        image = image.resize((INPUT_W, INPUT_H), Image.BILINEAR)

        # Convert to tensor and normalize
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # Build target grid (20x12 for 320x192 input with stride 16)
        target = torch.zeros(5 + NUM_CLASSES, GRID_H, GRID_W)

        # Process annotations
        for ann in anns:
            bbox = ann['bbox']  # [x, y, w, h] in pixels
            cat_id = ann['category_id']  # Already remapped to 0-2

            # Skip tiny boxes (< 1% of image area)
            if bbox[2] * bbox[3] < 0.0001 * orig_w * orig_h:
                continue

            # Convert bbox to normalized center coords
            cx = (bbox[0] + bbox[2] / 2) / orig_w
            cy = (bbox[1] + bbox[3] / 2) / orig_h
            bw = bbox[2] / orig_w
            bh = bbox[3] / orig_h

            # Clamp to valid range
            cx = max(0.001, min(0.999, cx))
            cy = max(0.001, min(0.999, cy))
            bw = max(0.01, min(0.99, bw))
            bh = max(0.01, min(0.99, bh))

            # Find grid cell (separate x and y grid sizes)
            gx = int(cx * GRID_W)
            gy = int(cy * GRID_H)
            gx = min(max(gx, 0), GRID_W - 1)
            gy = min(max(gy, 0), GRID_H - 1)

            # Only assign if cell is empty (first object wins)
            if target[0, gy, gx] == 0:
                target[0, gy, gx] = 1.0  # objectness
                # Box coords relative to cell
                target[1, gy, gx] = cx * GRID_W - gx  # offset x (0-1)
                target[2, gy, gx] = cy * GRID_H - gy  # offset y (0-1)
                target[3, gy, gx] = bw  # width (normalized to image)
                target[4, gy, gx] = bh  # height (normalized to image)
                target[5 + cat_id, gy, gx] = 1.0  # class one-hot

        return image, target


def compute_ciou(pred_box, target_box, grid_coords, eps=1e-7):
    """
    Compute Complete IoU (CIoU) loss for bounding boxes.

    Args:
        pred_box: [N, 4] predicted boxes (x_off, y_off, w, h) - RAW outputs
                  x_off, y_off: raw offsets (will be clamped to 0-1)
                  w, h: raw width/height (will be clamped to 0-1)
        target_box: [N, 4] target boxes (same format, already in 0-1 range)
        grid_coords: [N, 2] grid cell coordinates (gx, gy) for each box

    Returns:
        ciou_loss: scalar CIoU loss (1 - CIoU)
    """
    # Convert from (x_off, y_off, w, h) to image-normalized (cx, cy, w, h)
    # Then to (x1, y1, x2, y2)
    gx = grid_coords[:, 0]
    gy = grid_coords[:, 1]

    # Clamp predictions to valid range (0-1) for offset and size
    pred_x_off = pred_box[:, 0].clamp(0, 1)
    pred_y_off = pred_box[:, 1].clamp(0, 1)
    pred_w = pred_box[:, 2].clamp(0.001, 1)  # min size to avoid zero area
    pred_h = pred_box[:, 3].clamp(0.001, 1)

    # Center = (grid_cell + offset) / grid_size -> normalized (0-1)
    pred_cx = (gx + pred_x_off) / GRID_W
    pred_cy = (gy + pred_y_off) / GRID_H

    target_cx = (gx + target_box[:, 0]) / GRID_W
    target_cy = (gy + target_box[:, 1]) / GRID_H
    target_w = target_box[:, 2]
    target_h = target_box[:, 3]

    # Convert to x1, y1, x2, y2
    pred_x1 = pred_cx - pred_w / 2
    pred_y1 = pred_cy - pred_h / 2
    pred_x2 = pred_cx + pred_w / 2
    pred_y2 = pred_cy + pred_h / 2

    target_x1 = target_cx - target_w / 2
    target_y1 = target_cy - target_h / 2
    target_x2 = target_cx + target_w / 2
    target_y2 = target_cy + target_h / 2

    # Intersection
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Union
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    union_area = pred_area + target_area - inter_area + eps

    # IoU
    iou = inter_area / union_area

    # Enclosing box
    enclose_x1 = torch.min(pred_x1, target_x1)
    enclose_y1 = torch.min(pred_y1, target_y1)
    enclose_x2 = torch.max(pred_x2, target_x2)
    enclose_y2 = torch.max(pred_y2, target_y2)

    # Distance term (center distance / diagonal distance)
    pred_cx = (pred_x1 + pred_x2) / 2
    pred_cy = (pred_y1 + pred_y2) / 2
    target_cx = (target_x1 + target_x2) / 2
    target_cy = (target_y1 + target_y2) / 2

    center_dist_sq = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
    enclose_diag_sq = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps

    # Aspect ratio term
    pred_w = pred_box[:, 2]
    pred_h = pred_box[:, 3]
    target_w = target_box[:, 2]
    target_h = target_box[:, 3]

    v = (4 / (np.pi ** 2)) * ((torch.atan(target_w / (target_h + eps)) -
                               torch.atan(pred_w / (pred_h + eps))) ** 2)
    alpha = v / (1 - iou + v + eps)

    # CIoU = IoU - distance_term - aspect_ratio_term
    ciou = iou - center_dist_sq / enclose_diag_sq - alpha * v

    return (1 - ciou).mean()


def compute_loss(pred, target):
    """
    Compute detection loss with CIoU for boxes.

    pred/target layout: [B, 8, H, W]
      [0]: objectness (logits -> sigmoid)
      [1:5]: box (x_off, y_off, w, h)
      [5:8]: class scores (logits -> sigmoid)
    """
    device = pred.device
    obj_mask = target[:, 0:1] > 0  # [B, 1, H, W]
    num_pos = obj_mask.sum().clamp(min=1)

    # Objectness loss (focal loss)
    obj_pred = pred[:, 0:1]
    obj_target = target[:, 0:1]

    bce = nn.functional.binary_cross_entropy_with_logits(
        obj_pred, obj_target, reduction='none'
    )
    prob = torch.sigmoid(obj_pred)
    # Focal weighting with gamma=2
    focal_weight = torch.where(obj_target > 0, (1 - prob) ** 2, prob ** 2)
    obj_loss = (focal_weight * bce).mean()

    # Box loss (only on positive cells)
    # NOTE: We use RAW outputs for box regression (no sigmoid) to avoid vanishing gradients
    # Using smooth L1 loss for stable training
    if obj_mask.sum() > 0:
        # Get predicted and target boxes for positive cells
        # NO sigmoid on box predictions - use raw outputs
        pred_box = pred[:, 1:5]
        target_box = target[:, 1:5]

        # Flatten and select positive samples
        mask_flat = obj_mask[:, 0].flatten()  # [B*H*W]
        pred_flat = pred_box.permute(0, 2, 3, 1).reshape(-1, 4)  # [B*H*W, 4]
        target_flat = target_box.permute(0, 2, 3, 1).reshape(-1, 4)

        pred_pos = pred_flat[mask_flat]
        target_pos = target_flat[mask_flat]

        # Use smooth L1 loss for box regression (stable gradients with raw outputs)
        box_loss = nn.functional.smooth_l1_loss(pred_pos, target_pos)
    else:
        box_loss = torch.tensor(0.0, device=device)

    # Class loss with label smoothing and class weighting
    if obj_mask.sum() > 0:
        cls_pred = pred[:, 5:][obj_mask.expand(-1, NUM_CLASSES, -1, -1)]
        cls_target = target[:, 5:][obj_mask.expand(-1, NUM_CLASSES, -1, -1)]

        # Label smoothing: soften targets slightly
        cls_target_smooth = cls_target * 0.95 + 0.025

        # Apply class weights to handle imbalance (person:cat:dog = 1:15:12)
        # Reshape cls_pred/target from flat to [N, NUM_CLASSES]
        cls_pred_reshaped = cls_pred.view(-1, NUM_CLASSES)
        cls_target_reshaped = cls_target_smooth.view(-1, NUM_CLASSES)

        # Per-class BCE with weights
        class_weights = CLASS_WEIGHTS.to(device)
        bce_per_class = nn.functional.binary_cross_entropy_with_logits(
            cls_pred_reshaped, cls_target_reshaped, reduction='none'
        )
        weighted_bce = bce_per_class * class_weights.unsqueeze(0)
        cls_loss = weighted_bce.sum() / num_pos
    else:
        cls_loss = torch.tensor(0.0, device=device)

    # Weighted sum (CIoU loss is already well-scaled)
    total_loss = obj_loss + 5.0 * box_loss + 1.0 * cls_loss
    return total_loss, obj_loss, box_loss, cls_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./coco_3class')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--output', type=str, default='./checkpoints')
    parser.add_argument('--multi-gpu', action='store_true', help='Use all available GPUs')
    parser.add_argument('--workers', type=int, default=8, help='DataLoader workers')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    # Setup device(s)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        num_gpus = torch.cuda.device_count()
        print(f"CUDA available: {num_gpus} GPU(s)")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        device = torch.device('cpu')
        num_gpus = 0
        print("Using CPU (this will be slow!)")

    # Data transforms with augmentation for training
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Datasets (augmentation enabled for training only)
    train_dataset = COCODetectionDataset(args.data, 'train2017', train_transform, augment=True)
    val_dataset = COCODetectionDataset(args.data, 'val2017', val_transform, augment=False)

    # Scale batch size for multi-GPU
    effective_batch = args.batch_size * max(1, num_gpus) if args.multi_gpu else args.batch_size
    print(f"Effective batch size: {effective_batch}")

    # Use weighted sampler to oversample rare classes (cat/dog)
    train_sampler = train_dataset.get_sampler()
    train_loader = DataLoader(train_dataset, batch_size=effective_batch,
                              sampler=train_sampler, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=effective_batch,
                            shuffle=False, num_workers=args.workers, pin_memory=True)

    # Model
    model = TinyDet(num_classes=NUM_CLASSES)
    print(f"Model params: {model.count_params():,}")

    # Multi-GPU with DataParallel
    if args.multi_gpu and num_gpus > 1:
        print(f"Using DataParallel on {num_gpus} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Optimizer with scaled LR for larger batch
    scaled_lr = args.lr * (effective_batch / 64)  # Linear scaling rule
    optimizer = optim.AdamW(model.parameters(), lr=scaled_lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_val_loss = float('inf')

    # Resume from checkpoint
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_val_loss = ckpt.get('val_loss', float('inf'))
        else:
            state_dict = ckpt

        # Handle DataParallel prefix mismatch
        model_is_dp = hasattr(model, 'module')
        ckpt_is_dp = any(k.startswith('module.') for k in state_dict.keys())

        if model_is_dp and not ckpt_is_dp:
            # Add 'module.' prefix to checkpoint keys
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
        elif not model_is_dp and ckpt_is_dp:
            # Remove 'module.' prefix from checkpoint keys
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)

    os.makedirs(args.output, exist_ok=True)

    # Training loop
    print(f"\nStarting training from epoch {start_epoch+1}")
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss, obj_l, box_l, cls_l = compute_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss, _, _, _ = compute_loss(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        scheduler.step()
        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{args.epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, LR: {lr:.6f}, Time: {elapsed:.1f}s")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(state, f"{args.output}/tinydet_best.pth")
            print(f"  -> New best! Saved tinydet_best.pth")

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, f"{args.output}/tinydet_epoch{epoch+1}.pth")

    # Save final
    state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(state, f"{args.output}/tinydet_final.pth")
    print(f"\nTraining complete! Best val_loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()

