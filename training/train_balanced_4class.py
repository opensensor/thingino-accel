#!/usr/bin/env python3
"""
Training script for balanced 4-class detection model with focal loss.
Classes: person, vehicle, cat, dog
"""
import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from tinydet import TinyDet

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
IMG_SIZE = (192, 320)  # H, W
NUM_CLASSES = 4
CLASS_NAMES = ['person', 'vehicle', 'cat', 'dog']

# Focal loss parameters
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# Class weights (inverse frequency) - will be computed from data
CLASS_WEIGHTS = None

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

class DetectionDataset(Dataset):
    def __init__(self, ann_file, img_dirs, img_size=(192, 320), augment=True):
        with open(ann_file) as f:
            data = json.load(f)
        self.images = {img['id']: img for img in data['images']}
        self.img_size = img_size
        self.augment = augment
        self.img_dirs = img_dirs if isinstance(img_dirs, list) else [img_dirs]
        
        # Group annotations by image
        self.img_anns = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_anns:
                self.img_anns[img_id] = []
            self.img_anns[img_id].append(ann)
        
        self.img_ids = list(self.img_anns.keys())
        
        # Compute class weights
        class_counts = [0] * NUM_CLASSES
        for anns in self.img_anns.values():
            for ann in anns:
                class_counts[ann['category_id']] += 1
        total = sum(class_counts)
        self.class_weights = torch.tensor([total / (NUM_CLASSES * c) if c > 0 else 1.0 for c in class_counts])
        print(f"  Class distribution: {dict(zip(CLASS_NAMES, class_counts))}")
        print(f"  Class weights: {self.class_weights.tolist()}")
    
    def __len__(self):
        return len(self.img_ids)
    
    def find_image(self, filename):
        for img_dir in self.img_dirs:
            path = Path(img_dir) / filename
            if path.exists():
                return path
        return None
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        img_path = self.find_image(img_info['file_name'])
        
        if img_path is None:
            # Return dummy data if image not found
            return torch.zeros(3, *self.img_size), torch.zeros(5 + NUM_CLASSES, self.img_size[0]//16, self.img_size[1]//16)
        
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        
        # Resize
        img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        
        # Augment
        if self.augment:
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # To tensor and normalize
        img_np = np.array(img).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = (img_np - mean) / std
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
        
        # Create target grid
        grid_h, grid_w = self.img_size[0] // 16, self.img_size[1] // 16
        target = torch.zeros(5 + NUM_CLASSES, grid_h, grid_w)
        
        for ann in self.img_anns[img_id]:
            x, y, w, h = ann['bbox']
            # Scale to new size
            x = x * self.img_size[1] / orig_w
            y = y * self.img_size[0] / orig_h
            w = w * self.img_size[1] / orig_w
            h = h * self.img_size[0] / orig_h
            
            # Center
            cx, cy = x + w/2, y + h/2
            
            # Grid cell
            gx, gy = int(cx / 16), int(cy / 16)
            gx = min(max(gx, 0), grid_w - 1)
            gy = min(max(gy, 0), grid_h - 1)
            
            # Set target
            target[0, gy, gx] = 1.0  # objectness
            target[1, gy, gx] = (cx % 16) / 16  # x offset
            target[2, gy, gx] = (cy % 16) / 16  # y offset
            target[3, gy, gx] = w / self.img_size[1]  # width
            target[4, gy, gx] = h / self.img_size[0]  # height
            target[5 + ann['category_id'], gy, gx] = 1.0  # class
        
        return img_tensor, target

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device}")

    # Dataset
    train_ds = DetectionDataset(
        'balanced_4class/annotations/instances_train.json',
        ['balanced_4class/train2017', 'balanced_4class/images', 'coco/train2017', 'combined_dataset/images'],
        augment=True
    )
    val_ds = DetectionDataset(
        'balanced_4class/annotations/instances_val.json',
        ['balanced_4class/val2017', 'coco/val2017'],
        augment=False
    )

    print(f"Train: {len(train_ds)} images")
    print(f"Val: {len(val_ds)} images")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = TinyDet(num_classes=NUM_CLASSES).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters())} params")

    # Losses
    focal_loss = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    class_weights = train_ds.class_weights.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    # Output dir
    out_dir = Path('runs/balanced_4class')
    out_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float('inf')

    # Negative sampling ratio (how many negatives per positive)
    NEG_RATIO = 3

    def compute_loss(outputs, targets, class_weights, focal_loss, neg_ratio=NEG_RATIO):
        """Compute loss with YOLO-style sampling - only train on positive cells + sampled negatives"""
        batch_size = outputs.shape[0]

        # Masks
        pos_mask = targets[:, 0] > 0.5  # [B, H, W]
        neg_mask = targets[:, 0] < 0.5

        num_pos = pos_mask.sum().item()
        num_neg_sample = int(num_pos * neg_ratio)

        if num_pos == 0:
            # No objects in batch - just compute objectness loss on some negatives
            neg_indices = neg_mask.nonzero()
            if len(neg_indices) > 100:
                perm = torch.randperm(len(neg_indices))[:100]
                neg_indices = neg_indices[perm]
            obj_loss = F.binary_cross_entropy_with_logits(
                outputs[:, 0][neg_mask][:100] if neg_mask.sum() > 100 else outputs[:, 0][neg_mask],
                targets[:, 0][neg_mask][:100] if neg_mask.sum() > 100 else targets[:, 0][neg_mask]
            )
            return obj_loss, torch.tensor(0.0, device=outputs.device), torch.tensor(0.0, device=outputs.device)

        # OBJECTNESS LOSS: All positives + sampled negatives
        # Positive objectness
        pos_obj_out = outputs[:, 0][pos_mask]
        pos_obj_tgt = targets[:, 0][pos_mask]

        # Sample negatives
        neg_indices = neg_mask.nonzero()
        if len(neg_indices) > num_neg_sample:
            perm = torch.randperm(len(neg_indices))[:num_neg_sample]
            neg_indices = neg_indices[perm]

        neg_obj_out = outputs[:, 0][neg_indices[:, 0], neg_indices[:, 1], neg_indices[:, 2]]
        neg_obj_tgt = targets[:, 0][neg_indices[:, 0], neg_indices[:, 1], neg_indices[:, 2]]

        # Combined objectness loss with positive weighting
        all_obj_out = torch.cat([pos_obj_out, neg_obj_out])
        all_obj_tgt = torch.cat([pos_obj_tgt, neg_obj_tgt])
        obj_loss = focal_loss(all_obj_out, all_obj_tgt)

        # BOX LOSS: Only on positive cells
        box_out = outputs[:, 1:5].permute(0, 2, 3, 1)[pos_mask]  # [num_pos, 4]
        box_tgt = targets[:, 1:5].permute(0, 2, 3, 1)[pos_mask]  # [num_pos, 4]
        box_loss = F.smooth_l1_loss(box_out, box_tgt)

        # CLASS LOSS: Only on positive cells (where objects exist)
        class_out = outputs[:, 5:].permute(0, 2, 3, 1)[pos_mask]  # [num_pos, 4]
        class_tgt = targets[:, 5:].permute(0, 2, 3, 1)[pos_mask]  # [num_pos, 4]

        # Weighted class loss
        class_loss = torch.tensor(0.0, device=outputs.device)
        for c in range(NUM_CLASSES):
            c_loss = focal_loss(class_out[:, c], class_tgt[:, c])
            class_loss = class_loss + class_weights[c] * c_loss
        class_loss = class_loss / NUM_CLASSES

        return obj_loss, box_loss, class_loss

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for imgs, targets in tqdm(train_loader, desc=f'Epoch {epoch}/{EPOCHS}'):
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)  # [B, 9, H, W]

            obj_loss, box_loss, class_loss = compute_loss(outputs, targets, class_weights, focal_loss)
            loss = obj_loss + box_loss + class_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = model(imgs)

                obj_loss, box_loss, class_loss = compute_loss(outputs, targets, class_weights, focal_loss)
                val_loss += (obj_loss + box_loss + class_loss).item()

        val_loss /= len(val_loader)

        print(f'Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}')

        # Save checkpoints
        if epoch % 20 == 0:
            torch.save(model.state_dict(), out_dir / f'tinydet_e{epoch}.pth')

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), out_dir / 'tinydet_best.pth')
            print(f'  New best: {val_loss:.4f}')

    torch.save(model.state_dict(), out_dir / 'tinydet_final.pth')
    print(f'Training complete. Best val_loss: {best_loss:.4f}')

if __name__ == '__main__':
    train()

