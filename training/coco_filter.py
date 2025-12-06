"""
Filter COCO dataset to person/dog/cat classes only.

COCO class IDs:
  1 = person
  17 = cat  
  18 = dog

Usage:
  python coco_filter.py --coco-dir /path/to/coco --output-dir ./coco_3class
"""

import json
import os
import shutil
from pathlib import Path
from collections import defaultdict
import argparse


# COCO class IDs we want to keep
KEEP_CLASSES = {
    1: 'person',
    17: 'cat',
    18: 'dog'
}

# Remap to 0-indexed for training
CLASS_REMAP = {
    1: 0,   # person -> 0
    17: 1,  # cat -> 1
    18: 2   # dog -> 2
}


def filter_annotations(input_json, output_json):
    """Filter COCO annotations to keep only target classes."""
    print(f"Loading {input_json}...")
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    # Filter categories
    new_categories = []
    for cat in data['categories']:
        if cat['id'] in KEEP_CLASSES:
            new_cat = cat.copy()
            new_cat['id'] = CLASS_REMAP[cat['id']]
            new_cat['name'] = KEEP_CLASSES[cat['id']]
            new_categories.append(new_cat)
    
    # Filter annotations and track which images have valid annotations
    new_annotations = []
    image_ids_with_annotations = set()
    
    for ann in data['annotations']:
        if ann['category_id'] in KEEP_CLASSES:
            new_ann = ann.copy()
            new_ann['category_id'] = CLASS_REMAP[ann['category_id']]
            new_annotations.append(new_ann)
            image_ids_with_annotations.add(ann['image_id'])
    
    # Filter images to only those with at least one target class
    new_images = [img for img in data['images'] 
                  if img['id'] in image_ids_with_annotations]
    
    # Build new dataset
    new_data = {
        'info': data.get('info', {}),
        'licenses': data.get('licenses', []),
        'categories': new_categories,
        'images': new_images,
        'annotations': new_annotations
    }
    
    print(f"  Original: {len(data['images'])} images, {len(data['annotations'])} annotations")
    print(f"  Filtered: {len(new_images)} images, {len(new_annotations)} annotations")
    
    # Count per class
    class_counts = defaultdict(int)
    for ann in new_annotations:
        class_counts[ann['category_id']] += 1
    for cid, name in enumerate(['person', 'cat', 'dog']):
        print(f"    {name}: {class_counts[cid]:,}")
    
    # Save
    print(f"Saving {output_json}...")
    with open(output_json, 'w') as f:
        json.dump(new_data, f)
    
    return new_images


def main():
    parser = argparse.ArgumentParser(description='Filter COCO to 3 classes')
    parser.add_argument('--coco-dir', type=str, required=True,
                        help='Path to COCO dataset (contains annotations/, train2017/, val2017/)')
    parser.add_argument('--output-dir', type=str, default='./coco_3class',
                        help='Output directory for filtered dataset')
    parser.add_argument('--copy-images', action='store_true',
                        help='Copy images to output dir (otherwise just create annotations)')
    args = parser.parse_args()
    
    coco_dir = Path(args.coco_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directories
    (output_dir / 'annotations').mkdir(parents=True, exist_ok=True)
    
    # Process train and val
    for split in ['train2017', 'val2017']:
        ann_file = coco_dir / 'annotations' / f'instances_{split}.json'
        if not ann_file.exists():
            print(f"Skipping {split} (not found)")
            continue
        
        out_ann_file = output_dir / 'annotations' / f'instances_{split}.json'
        images = filter_annotations(ann_file, out_ann_file)
        
        # Optionally copy images
        if args.copy_images:
            img_out_dir = output_dir / split
            img_out_dir.mkdir(exist_ok=True)
            print(f"Copying {len(images)} images to {img_out_dir}...")
            for img in images:
                src = coco_dir / split / img['file_name']
                dst = img_out_dir / img['file_name']
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
    
    print("\nDone! Filtered dataset created at:", output_dir)
    print("\nTo use without copying images, symlink the image directories:")
    print(f"  ln -s {coco_dir}/train2017 {output_dir}/train2017")
    print(f"  ln -s {coco_dir}/val2017 {output_dir}/val2017")


if __name__ == '__main__':
    main()

