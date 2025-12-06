#!/usr/bin/env python3
"""
TinyDet Training Script

Clean, configurable training for 4-class object detection.
Supports float32 and quantization-aware training modes.

Usage:
  python scripts/train.py --config params.yaml
  python scripts/train.py --config configs/tinydet_base.yaml
"""

import os
import sys
import json
import math
import random
import argparse
from pathlib import Path
from collections import defaultdict

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

# Quantization imports
try:
    from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, convert
    from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
    HAS_QUANTIZATION = True
except ImportError:
    HAS_QUANTIZATION = False

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from tinydet import TinyDet


def load_config(config_path: str) -> dict:
    """Load and merge configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Loss Functions
# ============================================================================

def compute_ciou_loss(pred_box, target_box, eps=1e-7):
    """Compute CIoU loss for box regression."""
    # pred/target: [N, 4] as [cx, cy, w, h] normalized
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
    c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps
    rho2 = ((pred_box[:, 0] - target_box[:, 0]) ** 2 + 
            (pred_box[:, 1] - target_box[:, 1]) ** 2)

    # Aspect ratio term
    v = (4 / (math.pi ** 2)) * torch.pow(
        torch.atan(target_box[:, 2] / (target_box[:, 3] + eps)) -
        torch.atan(pred_box[:, 2] / (pred_box[:, 3] + eps)), 2
    )
    alpha = v / (1 - iou + v + eps)

    ciou = iou - rho2 / c2 - alpha * v
    return (1 - ciou).mean()


class DetectionLoss(nn.Module):
    """Combined detection loss with proper weighting."""
    
    def __init__(self, num_classes, class_weights, device, label_smoothing=0.02):
        super().__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights.to(device)
        self.device = device
        self.label_smoothing = label_smoothing

    def forward(self, pred, target, grid_h, grid_w):
        """
        pred: [B, 5+C, H, W] - raw logits
        target: [B, 5+C, H, W] - ground truth
        """
        pos_mask = target[:, 0] > 0.5
        neg_mask = ~pos_mask
        num_pos = pos_mask.sum().item()

        if num_pos == 0:
            obj_loss = F.binary_cross_entropy_with_logits(
                pred[:, 0][neg_mask], target[:, 0][neg_mask]
            )
            return obj_loss, torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        # Objectness loss - weighted BCE
        pos_weight = neg_mask.sum().float() / (pos_mask.sum().float() + 1e-6)
        pos_weight = torch.clamp(pos_weight, 1.0, 50.0)
        
        obj_loss = F.binary_cross_entropy_with_logits(
            pred[:, 0].flatten(), target[:, 0].flatten(),
            pos_weight=pos_weight
        )

        # Box loss using CIoU
        pos_indices = pos_mask.nonzero()  # [N, 3]
        
        pred_offset_x = torch.sigmoid(pred[:, 1][pos_mask])
        pred_offset_y = torch.sigmoid(pred[:, 2][pos_mask])
        pred_w = torch.sigmoid(pred[:, 3][pos_mask])
        pred_h = torch.sigmoid(pred[:, 4][pos_mask])

        gx = pos_indices[:, 2].float()
        gy = pos_indices[:, 1].float()
        pred_cx = (gx + pred_offset_x) / grid_w
        pred_cy = (gy + pred_offset_y) / grid_h
        pred_boxes = torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=1)

        target_cx = (gx + target[:, 1][pos_mask]) / grid_w
        target_cy = (gy + target[:, 2][pos_mask]) / grid_h
        target_boxes = torch.stack([
            target_cx, target_cy, 
            target[:, 3][pos_mask], target[:, 4][pos_mask]
        ], dim=1)

        box_loss = compute_ciou_loss(pred_boxes, target_boxes)

        # Class loss with label smoothing
        class_pred = pred[:, 5:].permute(0, 2, 3, 1)[pos_mask]
        class_target = target[:, 5:].permute(0, 2, 3, 1)[pos_mask]
        class_target = class_target * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes

        class_loss = torch.tensor(0.0, device=self.device)
        for c in range(self.num_classes):
            c_loss = F.binary_cross_entropy_with_logits(class_pred[:, c], class_target[:, c])
            class_loss = class_loss + self.class_weights[c] * c_loss
        class_loss = class_loss / self.class_weights.sum()

        return obj_loss, box_loss, class_loss


# ============================================================================
# Dataset
# ============================================================================

class DetectionDataset(Dataset):
    """Detection dataset with augmentation."""

    def __init__(self, ann_file, img_dirs, img_size, num_classes, augment=True, mosaic_prob=0.3):
        with open(ann_file) as f:
            data = json.load(f)

        self.images = {img['id']: img for img in data['images']}
        self.img_size = img_size  # (H, W)
        self.num_classes = num_classes
        self.augment = augment
        self.mosaic_prob = mosaic_prob if augment else 0
        self.img_dirs = img_dirs if isinstance(img_dirs, list) else [img_dirs]
        self.grid_h = img_size[0] // 16
        self.grid_w = img_size[1] // 16

        # Group annotations by image
        self.img_anns = defaultdict(list)
        for ann in data['annotations']:
            self.img_anns[ann['image_id']].append(ann)

        self.img_ids = list(self.img_anns.keys())

        # Compute class weights
        class_counts = [0] * num_classes
        for anns in self.img_anns.values():
            for ann in anns:
                if ann['category_id'] < num_classes:
                    class_counts[ann['category_id']] += 1

        total = sum(class_counts) + 1e-6
        self.class_weights = torch.tensor([
            (total / (num_classes * (c + 1))) ** 0.5 for c in class_counts
        ])

    def __len__(self):
        return len(self.img_ids)

    def find_image(self, img_info):
        """Find image file in possible directories."""
        filename = img_info.get('file_name', '')

        # Handle Oxford Pets absolute paths
        if img_info.get('source') == 'oxford_pets':
            for img_dir in self.img_dirs:
                path = Path(img_dir) / filename
                if path.exists():
                    return path

        # Standard COCO structure
        for img_dir in self.img_dirs:
            path = Path(img_dir) / filename
            if path.exists():
                return path
        return None

    def load_image_and_boxes(self, idx):
        """Load image and bounding boxes."""
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        img_path = self.find_image(img_info)

        if img_path is None:
            return None, []

        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size

        boxes = []
        for ann in self.img_anns[img_id]:
            x, y, w, h = ann['bbox']
            boxes.append({
                'cx': (x + w/2) / orig_w,
                'cy': (y + h/2) / orig_h,
                'w': w / orig_w,
                'h': h / orig_h,
                'class': ann['category_id']
            })

        return img, boxes

    def augment_image(self, img, boxes):
        """Apply augmentation pipeline."""
        if not self.augment:
            return img, boxes

        # Horizontal flip
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            boxes = [{'cx': 1 - b['cx'], 'cy': b['cy'], 'w': b['w'], 'h': b['h'], 'class': b['class']}
                     for b in boxes]

        # Color augmentation
        if random.random() > 0.3:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.4))
        if random.random() > 0.3:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.6, 1.4))
        if random.random() > 0.3:
            img = ImageEnhance.Color(img).enhance(random.uniform(0.5, 1.5))

        # Clip boxes
        valid_boxes = []
        for b in boxes:
            x1, y1 = max(0, b['cx'] - b['w']/2), max(0, b['cy'] - b['h']/2)
            x2, y2 = min(1, b['cx'] + b['w']/2), min(1, b['cy'] + b['h']/2)
            new_w, new_h = x2 - x1, y2 - y1
            if new_w > 0.03 and new_h > 0.03:
                valid_boxes.append({
                    'cx': (x1 + x2) / 2, 'cy': (y1 + y2) / 2,
                    'w': new_w, 'h': new_h, 'class': b['class']
                })

        return img, valid_boxes

    def create_target(self, boxes):
        """Create target tensor from boxes."""
        target = torch.zeros(5 + self.num_classes, self.grid_h, self.grid_w)

        for b in boxes:
            gx = int(b['cx'] * self.grid_w)
            gy = int(b['cy'] * self.grid_h)
            gx = min(max(gx, 0), self.grid_w - 1)
            gy = min(max(gy, 0), self.grid_h - 1)

            # Keep larger box if collision
            if target[0, gy, gx] > 0:
                existing_area = target[3, gy, gx] * target[4, gy, gx]
                if b['w'] * b['h'] < existing_area:
                    continue

            target[0, gy, gx] = 1.0
            target[1, gy, gx] = b['cx'] * self.grid_w - gx
            target[2, gy, gx] = b['cy'] * self.grid_h - gy
            target[3, gy, gx] = b['w']
            target[4, gy, gx] = b['h']
            if b['class'] < self.num_classes:
                target[5 + b['class'], gy, gx] = 1.0

        return target

    def __getitem__(self, idx):
        img, boxes = self.load_image_and_boxes(idx)
        if img is None:
            return torch.zeros(3, *self.img_size), torch.zeros(5 + self.num_classes, self.grid_h, self.grid_w)

        img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        img, boxes = self.augment_image(img, boxes)

        # Normalize
        img_np = np.array(img).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = (img_np - mean) / std
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()

        target = self.create_target(boxes)
        return img_tensor, target


# ============================================================================
# Training Loop
# ============================================================================

def train(config: dict, qat: bool = False):
    """Main training function.

    Args:
        config: Training configuration dict
        qat: If True, use Quantization-Aware Training
    """
    # Extract config
    model_cfg = config.get('model', {})
    train_cfg = config.get('train', {})
    prep_cfg = config.get('prepare', {})

    if qat and not HAS_QUANTIZATION:
        raise RuntimeError("QAT requested but PyTorch quantization not available")

    num_classes = model_cfg.get('num_classes', 4)
    img_h = model_cfg.get('input_height', 192)
    img_w = model_cfg.get('input_width', 320)
    grid_h = model_cfg.get('grid_height', 12)
    grid_w = model_cfg.get('grid_width', 20)

    epochs = train_cfg.get('epochs', 100)
    batch_size = train_cfg.get('batch_size', 32)
    lr = train_cfg.get('learning_rate', 0.001)
    weight_decay = train_cfg.get('weight_decay', 0.0001)
    warmup_epochs = train_cfg.get('warmup_epochs', 5)
    obj_weight = train_cfg.get('obj_loss_weight', 5.0)
    box_weight = train_cfg.get('box_loss_weight', 2.0)
    cls_weight = train_cfg.get('cls_loss_weight', 1.0)
    mosaic_prob = train_cfg.get('mosaic_prob', 0.3)
    label_smoothing = train_cfg.get('label_smoothing', 0.02)
    save_every = train_cfg.get('save_every', 10)
    output_dir = Path(train_cfg.get('output_dir', 'runs/current'))
    num_workers = train_cfg.get('num_workers', 4)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Datasets
    data_dir = Path(prep_cfg.get('output_dir', 'combined_dataset'))
    train_ann = data_dir / 'annotations' / 'instances_train2017.json'
    val_ann = data_dir / 'annotations' / 'instances_val2017.json'

    img_dirs = [
        data_dir / 'images',
        'combined_dataset/images',
        'train2017',
        'coco/train2017',
        'coco/val2017',
        'oxford_pets/images',
    ]

    print(f"Loading training data from {train_ann}")
    train_ds = DetectionDataset(
        str(train_ann), img_dirs, (img_h, img_w), num_classes,
        augment=True, mosaic_prob=mosaic_prob
    )
    val_ds = DetectionDataset(
        str(val_ann), img_dirs, (img_h, img_w), num_classes,
        augment=False, mosaic_prob=0
    )

    print(f"Train: {len(train_ds)} images, Val: {len(val_ds)} images")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # Model
    model = TinyDet(num_classes=num_classes)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {param_count:,} parameters")

    # QAT setup - must be done before moving to GPU
    if qat:
        print("Enabling Quantization-Aware Training (QAT)...")
        # Use FBGEMM backend for x86 (training), will export to qnnpack-friendly format
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
        # Switch to eval mode for fusion (required by fuse_modules)
        model.eval()
        # Fuse Conv+BN+ReLU before QAT
        torch.ao.quantization.fuse_modules(model.backbone.stem, [['0.conv', '0.bn', '0.relu'], ['1.conv', '1.bn', '1.relu']], inplace=True)
        torch.ao.quantization.fuse_modules(model.backbone.s1, [['0.conv', '0.bn', '0.relu'], ['1.conv', '1.bn', '1.relu']], inplace=True)
        torch.ao.quantization.fuse_modules(model.backbone.s2, [['0.conv', '0.bn', '0.relu'], ['1.conv', '1.bn', '1.relu']], inplace=True)
        torch.ao.quantization.fuse_modules(model.backbone.s3, [['0.conv', '0.bn', '0.relu'], ['1.conv', '1.bn', '1.relu']], inplace=True)
        torch.ao.quantization.fuse_modules(model.head.head, [['0.conv', '0.bn', '0.relu'], ['1.conv', '1.bn', '1.relu']], inplace=True)
        # Switch back to train mode for QAT preparation
        model.train()
        # Prepare for QAT
        model = torch.ao.quantization.prepare_qat(model, inplace=False)
        print("  QAT preparation complete - fake quantization enabled")

    model = model.to(device)

    # Loss
    criterion = DetectionLoss(
        num_classes, train_ds.class_weights, device, label_smoothing
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # LR schedule: warmup + cosine
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Output
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path('metrics')
    metrics_dir.mkdir(exist_ok=True)

    best_loss = float('inf')
    train_history = []

    print(f"\nStarting training for {epochs} epochs...")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_obj = 0.0
        epoch_box = 0.0
        epoch_cls = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}')
        for imgs, targets in pbar:
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)

            obj_loss, box_loss, cls_loss = criterion(outputs, targets, grid_h, grid_w)
            loss = obj_weight * obj_loss + box_weight * box_loss + cls_weight * cls_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_obj += obj_loss.item()
            epoch_box += box_loss.item()
            epoch_cls += cls_loss.item()

            pbar.set_postfix(loss=f'{loss.item():.4f}')

        epoch_loss /= len(train_loader)
        epoch_obj /= len(train_loader)
        epoch_box /= len(train_loader)
        epoch_cls /= len(train_loader)

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = model(imgs)
                obj_loss, box_loss, cls_loss = criterion(outputs, targets, grid_h, grid_w)
                val_loss += (obj_weight * obj_loss + box_weight * box_loss + cls_weight * cls_loss).item()
        val_loss /= len(val_loader)

        lr_current = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch}: loss={epoch_loss:.4f} (obj={epoch_obj:.3f}, box={epoch_box:.3f}, cls={epoch_cls:.3f}), val={val_loss:.4f}, lr={lr_current:.6f}')

        train_history.append({
            'epoch': epoch, 'loss': epoch_loss, 'val_loss': val_loss,
            'obj_loss': epoch_obj, 'box_loss': epoch_box, 'cls_loss': epoch_cls, 'lr': lr_current
        })

        # Save checkpoints
        if epoch % save_every == 0:
            torch.save(model.state_dict(), output_dir / f'tinydet_e{epoch}.pth')

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), output_dir / 'tinydet_best.pth')
            print(f'  New best: {val_loss:.4f}')

    # Save final
    torch.save(model.state_dict(), output_dir / 'tinydet_final.pth')

    # Convert QAT model to quantized model and save
    if qat:
        print("\nConverting QAT model to quantized format...")
        model.eval()
        model_cpu = model.cpu()
        model_int8 = torch.ao.quantization.convert(model_cpu, inplace=False)
        torch.save(model_int8.state_dict(), output_dir / 'tinydet_int8.pth')

        # Also save a scripted version for easier deployment
        try:
            example_input = torch.randn(1, 3, img_h, img_w)
            scripted = torch.jit.trace(model_int8, example_input)
            scripted.save(str(output_dir / 'tinydet_int8.pt'))
            print(f"  Saved quantized model: tinydet_int8.pth, tinydet_int8.pt")
        except Exception as e:
            print(f"  Warning: Could not save scripted model: {e}")
            print(f"  Saved quantized model: tinydet_int8.pth")

    # Save metrics
    with open(metrics_dir / 'train_metrics.json', 'w') as f:
        json.dump({'best_loss': best_loss, 'epochs': epochs, 'final_loss': epoch_loss, 'qat': qat}, f, indent=2)

    # Save loss curve CSV for DVC plots
    with open(metrics_dir / 'loss_curve.csv', 'w') as f:
        f.write('epoch,loss,val_loss,obj_loss,box_loss,cls_loss\n')
        for h in train_history:
            f.write(f"{h['epoch']},{h['loss']:.6f},{h['val_loss']:.6f},{h['obj_loss']:.6f},{h['box_loss']:.6f},{h['cls_loss']:.6f}\n")

    print("\n" + "=" * 60)
    print(f"Training complete! Best val loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    if qat:
        print("QAT mode: INT8 quantized model saved")


def main():
    parser = argparse.ArgumentParser(description='Train TinyDet')
    parser.add_argument('--config', type=str, default='params.yaml')
    parser.add_argument('--seed', type=int, default=42)
    # Config overrides for experiment runner
    parser.add_argument('--epochs', type=int, help='Override epochs')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--learning-rate', type=float, help='Override learning rate')
    parser.add_argument('--weight-decay', type=float, help='Override weight decay')
    parser.add_argument('--warmup-epochs', type=int, help='Override warmup epochs')
    parser.add_argument('--output-dir', type=str, help='Override output directory')
    parser.add_argument('--base-channels', type=int, help='Override model base channels')
    parser.add_argument('--lr-schedule', type=str, choices=['step', 'cosine'], help='LR schedule type')
    parser.add_argument('--use-focal-loss', type=bool, help='Use focal loss')
    parser.add_argument('--focal-alpha', type=float, default=0.25, help='Focal loss alpha')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--qat', action='store_true', help='Enable Quantization-Aware Training')
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config(args.config)

    # Apply command-line overrides
    if args.epochs:
        config['train']['epochs'] = args.epochs
    if args.batch_size:
        config['train']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['train']['learning_rate'] = args.learning_rate
    if args.weight_decay:
        config['train']['weight_decay'] = args.weight_decay
    if args.warmup_epochs:
        config['train']['warmup_epochs'] = args.warmup_epochs
    if args.output_dir:
        config['train']['output_dir'] = args.output_dir
    if args.base_channels:
        config['model']['base_channels'] = args.base_channels
    if args.lr_schedule:
        config['train']['lr_schedule'] = args.lr_schedule
    if args.use_focal_loss:
        config['train']['use_focal_loss'] = args.use_focal_loss
        config['train']['focal_alpha'] = args.focal_alpha
        config['train']['focal_gamma'] = args.focal_gamma

    train(config, qat=args.qat)


if __name__ == '__main__':
    main()
