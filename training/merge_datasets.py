#!/usr/bin/env python3
"""Merge Oxford Pets with COCO filtered dataset."""

import json
import os
import shutil
from pathlib import Path

def merge_datasets(coco_dir, oxford_dir, output_dir):
    """Merge COCO and Oxford Pets datasets."""
    coco_dir = Path(coco_dir)
    oxford_dir = Path(oxford_dir)
    output_dir = Path(output_dir)

    # Create output
    out_images = output_dir / 'images'
    out_images.mkdir(parents=True, exist_ok=True)

    # Load COCO annotations (already filtered with our category IDs: 0=person, 1=cat, 2=dog)
    with open(coco_dir / 'annotations' / 'instances_train2017.json') as f:
        coco = json.load(f)
    
    # Load Oxford annotations
    with open(oxford_dir / 'annotations.json') as f:
        oxford = json.load(f)
    
    print(f"COCO: {len(coco['images'])} images, {len(coco['annotations'])} annotations")
    print(f"Oxford: {len(oxford['images'])} images, {len(oxford['annotations'])} annotations")
    
    # Merged structure
    merged = {
        'images': [],
        'annotations': [],
        'categories': coco['categories']
    }
    
    # Build image ID set from COCO annotations
    coco_img_ids = set(ann['image_id'] for ann in coco['annotations'])
    coco_img_map = {img['id']: img for img in coco['images'] if img['id'] in coco_img_ids}

    # Copy COCO images and annotations
    print("Copying COCO images...")
    for img_id, img in coco_img_map.items():
        src = coco_dir / 'train2017' / img['file_name']
        dst = out_images / img['file_name']
        if src.exists() and not dst.exists():
            os.symlink(src.resolve(), dst)
        merged['images'].append(img)

    for ann in coco['annotations']:
        merged['annotations'].append(ann)
    
    # Copy Oxford images and annotations
    print("Copying Oxford Pets images...")
    for img in oxford['images']:
        src = oxford_dir / 'images' / img['file_name']
        dst = out_images / img['file_name']
        if src.exists() and not dst.exists():
            os.symlink(src.resolve(), dst)
        merged['images'].append(img)
    
    for ann in oxford['annotations']:
        merged['annotations'].append(ann)
    
    # Count by class
    class_counts = {0: 0, 1: 0, 2: 0}
    for ann in merged['annotations']:
        class_counts[ann['category_id']] += 1
    
    print(f"\nMerged dataset:")
    print(f"  Total images: {len(merged['images'])}")
    print(f"  Total annotations: {len(merged['annotations'])}")
    print(f"  Person: {class_counts[0]}")
    print(f"  Cat: {class_counts[1]}")
    print(f"  Dog: {class_counts[2]}")
    
    # Save
    with open(output_dir / 'annotations.json', 'w') as f:
        json.dump(merged, f)
    
    print(f"\nSaved to: {output_dir}")
    return merged

if __name__ == '__main__':
    merge_datasets('coco_filtered', 'oxford_pets_coco', 'combined_dataset')

