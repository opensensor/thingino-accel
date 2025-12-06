#!/usr/bin/env python3
"""
Train TinyDet on security dataset (person, vehicle, cat, dog)
100 epochs, balanced dataset, proper evaluation
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image, ImageEnhance
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from tinydet import TinyDet, INPUT_W, INPUT_H, GRID_W, GRID_H

NUM_CLASSES = 4
CLASS_NAMES = ['person', 'vehicle', 'cat', 'dog']
# Inverse frequency weights (person/vehicle have more data)
CLASS_WEIGHTS = torch.tensor([1.0, 1.0, 10.0, 10.0])


class SecurityDataset(Dataset):
    """Dataset for security detection with balanced sampling."""
    
    def __init__(self, data_dir, split='train', transform=None, augment=False):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.augment = augment
        
        # Determine image directory (train2017 or val2017)
        self.img_dir = self.data_dir / ('train2017' if split == 'train' else 'val2017')
        
        ann_file = self.data_dir / 'annotations' / f'instances_{split}.json'
        with open(ann_file) as f:
            data = json.load(f)
        
        self.images = {img['id']: img for img in data['images']}
        self.annotations = defaultdict(list)
        self.image_classes = defaultdict(set)
        
        for ann in data['annotations']:
            img_id = ann['image_id']
            self.annotations[img_id].append(ann)
            self.image_classes[img_id].add(ann['category_id'])
        
        self.image_ids = list(self.annotations.keys())
        self._compute_weights()
        print(f"Loaded {len(self.image_ids)} images from {split}")
    
    def _compute_weights(self):
        """Compute sampling weights to balance classes."""
        class_counts = defaultdict(int)
        for img_id, classes in self.image_classes.items():
            for c in classes:
                class_counts[c] += 1
        
        print(f"  Class distribution: {dict(class_counts)}")
        max_count = max(class_counts.values())
        
        self.sample_weights = []
        for img_id in self.image_ids:
            weight = max(max_count / class_counts.get(c, max_count) 
                        for c in self.image_classes[img_id])
            self.sample_weights.append(weight)
    
    def get_sampler(self):
        return WeightedRandomSampler(self.sample_weights, len(self), replacement=True)
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        anns = self.annotations[img_id]
        
        img_path = self.img_dir / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        
        # Augmentation
        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            for ann in anns:
                ann['bbox'][0] = orig_w - ann['bbox'][0] - ann['bbox'][2]
        
        image = image.resize((INPUT_W, INPUT_H), Image.BILINEAR)
        if self.transform:
            image = self.transform(image)
        
        # Build target grid
        target = torch.zeros(5 + NUM_CLASSES, GRID_H, GRID_W)
        for ann in anns:
            bbox = ann['bbox']
            cat_id = ann['category_id']
            
            if bbox[2] * bbox[3] < 0.0001 * orig_w * orig_h:
                continue
            
            cx = (bbox[0] + bbox[2]/2) / orig_w
            cy = (bbox[1] + bbox[3]/2) / orig_h
            bw, bh = bbox[2] / orig_w, bbox[3] / orig_h
            
            cx = max(0.001, min(0.999, cx))
            cy = max(0.001, min(0.999, cy))
            
            gx = min(max(int(cx * GRID_W), 0), GRID_W - 1)
            gy = min(max(int(cy * GRID_H), 0), GRID_H - 1)
            
            if target[0, gy, gx] == 0:
                target[0, gy, gx] = 1.0
                target[1, gy, gx] = cx * GRID_W - gx
                target[2, gy, gx] = cy * GRID_H - gy
                target[3, gy, gx] = bw
                target[4, gy, gx] = bh
                target[5 + cat_id, gy, gx] = 1.0
        
        return image, target


def compute_loss(pred, target):
    """Detection loss with focal objectness and weighted class loss."""
    device = pred.device
    obj_mask = target[:, 0:1] > 0
    num_pos = obj_mask.sum().clamp(min=1)
    
    # Focal loss for objectness
    obj_pred, obj_target = pred[:, 0:1], target[:, 0:1]
    bce = nn.functional.binary_cross_entropy_with_logits(obj_pred, obj_target, reduction='none')
    prob = torch.sigmoid(obj_pred)
    focal_weight = torch.where(obj_target > 0, (1-prob)**2, prob**2)
    obj_loss = (focal_weight * bce).mean()
    
    # Box loss (smooth L1)
    if obj_mask.sum() > 0:
        mask_flat = obj_mask[:, 0].flatten()
        pred_box = pred[:, 1:5].permute(0,2,3,1).reshape(-1, 4)[mask_flat]
        target_box = target[:, 1:5].permute(0,2,3,1).reshape(-1, 4)[mask_flat]
        box_loss = nn.functional.smooth_l1_loss(pred_box, target_box)
    else:
        box_loss = torch.tensor(0.0, device=device)
    
    # Weighted class loss
    if obj_mask.sum() > 0:
        cls_pred = pred[:, 5:][obj_mask.expand(-1, NUM_CLASSES, -1, -1)].view(-1, NUM_CLASSES)
        cls_target = target[:, 5:][obj_mask.expand(-1, NUM_CLASSES, -1, -1)].view(-1, NUM_CLASSES)
        weights = CLASS_WEIGHTS.to(device)
        bce = nn.functional.binary_cross_entropy_with_logits(cls_pred, cls_target, reduction='none')
        cls_loss = (bce * weights).sum() / num_pos
    else:
        cls_loss = torch.tensor(0.0, device=device)
    
    return obj_loss + 5.0 * box_loss + cls_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./security_dataset')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--output', default='./runs/security')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = SecurityDataset(args.data, 'train', transform, augment=True)
    val_ds = SecurityDataset(args.data, 'val', transform, augment=False)

    train_loader = DataLoader(train_ds, args.batch_size, sampler=train_ds.get_sampler(),
                              num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=4)

    model = TinyDet(num_classes=NUM_CLASSES).to(device)
    print(f"Model: {model.count_params():,} params, {NUM_CLASSES} classes: {CLASS_NAMES}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    os.makedirs(args.output, exist_ok=True)
    best_val = float('inf')

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        train_loss = 0
        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = compute_loss(model(imgs), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                val_loss += compute_loss(model(imgs), targets).item()
        val_loss /= len(val_loader)

        scheduler.step()
        print(f"Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}, time={time.time()-t0:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"{args.output}/tinydet_best.pth")
            print("  -> Saved best model")

        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), f"{args.output}/tinydet_e{epoch+1}.pth")

    torch.save(model.state_dict(), f"{args.output}/tinydet_final.pth")
    print(f"Done! Best val loss: {best_val:.4f}")


if __name__ == '__main__':
    main()

