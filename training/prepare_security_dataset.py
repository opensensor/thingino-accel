#!/usr/bin/env python3
"""
Prepare a balanced dataset for security/monitoring use case.
Classes: person, vehicle (car+truck+bus+motorcycle), cat, dog
"""

import json
import random
import os
import shutil
from collections import defaultdict
from pathlib import Path

# Target classes - merge vehicles into one class for simplicity
COCO_TO_NEW = {
    1: 0,   # person -> 0
    3: 1,   # car -> 1 (vehicle)
    8: 1,   # truck -> 1 (vehicle)
    6: 1,   # bus -> 1 (vehicle)
    4: 1,   # motorcycle -> 1 (vehicle)
    17: 2,  # cat -> 2
    18: 3,  # dog -> 3
}

CLASS_NAMES = ['person', 'vehicle', 'cat', 'dog']

def main():
    random.seed(42)
    
    # Load COCO annotations
    print("Loading COCO annotations...")
    with open('coco/annotations/instances_train2017.json') as f:
        train_data = json.load(f)
    with open('coco/annotations/instances_val2017.json') as f:
        val_data = json.load(f)
    
    # Create output directory
    out_dir = Path('security_dataset')
    out_dir.mkdir(exist_ok=True)
    (out_dir / 'annotations').mkdir(exist_ok=True)
    
    for split_name, data in [('train', train_data), ('val', val_data)]:
        print(f"\nProcessing {split_name}...")
        
        # Group annotations by image and filter to target classes
        img_to_anns = defaultdict(list)
        class_counts = defaultdict(int)
        
        for ann in data['annotations']:
            old_cat = ann['category_id']
            if old_cat in COCO_TO_NEW:
                new_cat = COCO_TO_NEW[old_cat]
                img_to_anns[ann['image_id']].append({
                    'id': ann['id'],
                    'image_id': ann['image_id'],
                    'category_id': new_cat,
                    'bbox': ann['bbox'],
                    'area': ann['area'],
                    'iscrowd': ann.get('iscrowd', 0)
                })
                class_counts[new_cat] += 1
        
        print(f"  Raw counts: {dict(class_counts)}")
        
        # For training, balance by undersampling
        if split_name == 'train':
            # Target: ~15k per class (match cat/dog scale with some oversampling)
            target_per_class = 15000
            
            # Group images by their "primary" class (most annotations)
            class_to_images = defaultdict(list)
            for img_id, anns in img_to_anns.items():
                # Find dominant class in this image
                cls_count = defaultdict(int)
                for ann in anns:
                    cls_count[ann['category_id']] += 1
                primary_cls = max(cls_count, key=cls_count.get)
                class_to_images[primary_cls].append(img_id)
            
            # Sample images per class
            selected_images = set()
            for cls_id in range(len(CLASS_NAMES)):
                imgs = class_to_images[cls_id]
                # For small classes, take all; for large, subsample
                n_to_take = min(len(imgs), target_per_class // 2)  # ~2 anns per image avg
                selected = random.sample(imgs, n_to_take) if len(imgs) > n_to_take else imgs
                selected_images.update(selected)
                print(f"  {CLASS_NAMES[cls_id]}: {len(imgs)} images -> {len(selected)} selected")
            
            # Filter annotations to selected images
            new_anns = []
            new_class_counts = defaultdict(int)
            for img_id in selected_images:
                for ann in img_to_anns[img_id]:
                    new_anns.append(ann)
                    new_class_counts[ann['category_id']] += 1
            
            print(f"  Balanced counts: {dict(new_class_counts)}")
        else:
            # For val, keep all
            new_anns = [ann for anns in img_to_anns.values() for ann in anns]
            selected_images = set(img_to_anns.keys())
        
        # Filter images
        img_id_set = selected_images
        new_images = [img for img in data['images'] if img['id'] in img_id_set]
        
        # Create new dataset
        new_data = {
            'images': new_images,
            'annotations': new_anns,
            'categories': [{'id': i, 'name': name} for i, name in enumerate(CLASS_NAMES)]
        }
        
        out_file = out_dir / 'annotations' / f'instances_{split_name}.json'
        with open(out_file, 'w') as f:
            json.dump(new_data, f)
        
        print(f"  Saved {len(new_images)} images, {len(new_anns)} annotations to {out_file}")
    
    # Create symlinks to COCO images
    for split in ['train2017', 'val2017']:
        src = Path('coco') / split
        dst = out_dir / split
        if src.exists() and not dst.exists():
            dst.symlink_to(src.resolve())
            print(f"Created symlink: {dst} -> {src}")

if __name__ == '__main__':
    main()

