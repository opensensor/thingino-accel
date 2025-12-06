#!/usr/bin/env python3
"""
Prepare balanced 4-class detection dataset.

Combines and balances:
- COCO 2017: person, car/truck/bus (→ vehicle), cat, dog
- Oxford-IIIT Pets: cats and dogs with bounding boxes

Usage:
  python scripts/prepare_data.py --config params.yaml
"""

import os
import sys
import json
import random
import argparse
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import yaml

# COCO category mappings
COCO_TO_CLASS = {
    1: 0,    # person → 0
    3: 1,    # car → vehicle (1)
    6: 1,    # bus → vehicle (1)
    8: 1,    # truck → vehicle (1)
    17: 2,   # cat → 2
    18: 3,   # dog → 3
}

CLASS_NAMES = ['person', 'vehicle', 'cat', 'dog']


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def filter_coco_annotations(
    coco_path: Path,
    split: str,
    max_per_class: Dict[int, int],
    seed: int = 42
) -> Tuple[List[dict], List[dict]]:
    """Filter COCO annotations to target classes with balancing."""
    ann_file = coco_path / 'annotations' / f'instances_{split}.json'
    
    print(f"Loading {ann_file}...")
    with open(ann_file) as f:
        coco = json.load(f)
    
    # Group annotations by image and class
    img_to_anns = defaultdict(list)
    img_classes = defaultdict(set)
    
    for ann in coco['annotations']:
        cat_id = ann['category_id']
        if cat_id in COCO_TO_CLASS:
            img_to_anns[ann['image_id']].append(ann)
            img_classes[ann['image_id']].add(COCO_TO_CLASS[cat_id])
    
    # Count per-class images
    class_images = defaultdict(list)
    for img_id, classes in img_classes.items():
        for cls in classes:
            class_images[cls].append(img_id)
    
    print(f"  COCO {split} class distribution:")
    for cls_id, name in enumerate(CLASS_NAMES):
        print(f"    {name}: {len(class_images[cls_id])} images")
    
    # Balance by undersampling majority classes
    random.seed(seed)
    selected_images = set()
    
    for cls_id in range(len(CLASS_NAMES)):
        imgs = class_images[cls_id]
        random.shuffle(imgs)
        max_count = max_per_class.get(cls_id, len(imgs))
        selected_images.update(imgs[:max_count])
    
    # Filter images and annotations
    img_id_to_info = {img['id']: img for img in coco['images']}
    filtered_images = [img_id_to_info[img_id] for img_id in selected_images 
                       if img_id in img_id_to_info]
    
    filtered_anns = []
    for ann in coco['annotations']:
        if ann['image_id'] in selected_images and ann['category_id'] in COCO_TO_CLASS:
            new_ann = ann.copy()
            new_ann['category_id'] = COCO_TO_CLASS[ann['category_id']]
            filtered_anns.append(new_ann)
    
    print(f"  Selected {len(filtered_images)} images, {len(filtered_anns)} annotations")
    return filtered_images, filtered_anns


def parse_oxford_pets(pets_path: Path) -> Tuple[List[dict], List[dict]]:
    """Parse Oxford-IIIT Pets dataset to COCO format."""
    from PIL import Image
    import xml.etree.ElementTree as ET
    
    # Cat breeds from Oxford Pets
    cat_breeds = {
        'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair',
        'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue',
        'Siamese', 'Sphynx'
    }
    
    xmls_dir = pets_path / 'annotations' / 'xmls'
    images_dir = pets_path / 'images'
    
    if not xmls_dir.exists():
        print(f"  Warning: Oxford Pets xmls not found at {xmls_dir}")
        return [], []
    
    images = []
    annotations = []
    img_id = 200000  # Start high to avoid COCO ID conflicts
    ann_id = 2000000
    
    for xml_file in xmls_dir.glob('*.xml'):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            filename = root.find('filename').text
            if not filename.endswith('.jpg'):
                filename += '.jpg'

            img_path = images_dir / filename
            if not img_path.exists():
                continue

            # Get image size
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            # Determine cat vs dog from breed name
            breed = filename.rsplit('_', 1)[0]
            is_cat = any(cat in breed for cat in cat_breeds)
            category_id = 2 if is_cat else 3  # cat=2, dog=3

            # Get bounding box
            obj = root.find('object')
            if obj is None:
                continue
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            bbox = [xmin, ymin, xmax - xmin, ymax - ymin]  # COCO format

            images.append({
                'id': img_id,
                'file_name': filename,
                'width': width,
                'height': height,
                'source': 'oxford_pets'
            })

            annotations.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': category_id,
                'bbox': bbox,
                'area': bbox[2] * bbox[3],
                'iscrowd': 0
            })

            img_id += 1
            ann_id += 1

        except Exception as e:
            continue

    # Count classes
    cat_count = sum(1 for a in annotations if a['category_id'] == 2)
    dog_count = sum(1 for a in annotations if a['category_id'] == 3)
    print(f"  Oxford Pets: {len(images)} images (cats: {cat_count}, dogs: {dog_count})")

    return images, annotations


def create_val_split(
    images: List[dict],
    annotations: List[dict],
    val_ratio: float,
    seed: int
) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    """Split data into train and validation sets."""
    random.seed(seed)

    # Group by class for stratified split
    img_to_classes = defaultdict(set)
    for ann in annotations:
        img_to_classes[ann['image_id']].add(ann['category_id'])

    class_images = defaultdict(list)
    for img in images:
        for cls in img_to_classes[img['id']]:
            class_images[cls].append(img['id'])

    # Select val images from each class
    val_img_ids = set()
    for cls_id in range(len(CLASS_NAMES)):
        imgs = list(set(class_images[cls_id]))
        random.shuffle(imgs)
        n_val = max(1, int(len(imgs) * val_ratio))
        val_img_ids.update(imgs[:n_val])

    # Split
    train_images = [img for img in images if img['id'] not in val_img_ids]
    val_images = [img for img in images if img['id'] in val_img_ids]
    train_anns = [ann for ann in annotations if ann['image_id'] not in val_img_ids]
    val_anns = [ann for ann in annotations if ann['image_id'] in val_img_ids]

    return train_images, train_anns, val_images, val_anns


def save_coco_format(
    images: List[dict],
    annotations: List[dict],
    output_path: Path,
    split_name: str
):
    """Save annotations in COCO format."""
    categories = [
        {'id': i, 'name': name, 'supercategory': 'object'}
        for i, name in enumerate(CLASS_NAMES)
    ]

    dataset = {
        'info': {'description': f'TinyDet {split_name} dataset', 'version': '1.0'},
        'licenses': [],
        'categories': categories,
        'images': images,
        'annotations': annotations
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(dataset, f)

    print(f"  Saved {len(images)} images, {len(annotations)} annotations to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Prepare TinyDet training data')
    parser.add_argument('--config', type=str, default='params.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    prep = config.get('prepare', {})

    coco_dir = Path(prep.get('coco_dir', 'datasets/coco'))
    pets_dir = Path(prep.get('oxford_pets_dir', 'datasets/oxford_pets'))
    output_dir = Path(prep.get('output_dir', 'datasets/prepared'))
    max_person = prep.get('max_person_images', 10000)
    max_vehicle = prep.get('max_vehicle_images', 10000)
    val_split = prep.get('val_split', 0.1)
    seed = prep.get('random_seed', 42)

    print("=" * 60)
    print("TinyDet Data Preparation")
    print("=" * 60)

    all_images = []
    all_annotations = []

    # Process COCO
    if coco_dir.exists():
        print(f"\n1. Processing COCO from {coco_dir}")
        max_per_class = {0: max_person, 1: max_vehicle, 2: 100000, 3: 100000}

        coco_images, coco_anns = filter_coco_annotations(
            coco_dir, 'train2017', max_per_class, seed
        )
        all_images.extend(coco_images)
        all_annotations.extend(coco_anns)
    else:
        print(f"\nWarning: COCO directory not found: {coco_dir}")

    # Process Oxford Pets
    if pets_dir.exists():
        print(f"\n2. Processing Oxford Pets from {pets_dir}")
        pets_images, pets_anns = parse_oxford_pets(pets_dir)
        all_images.extend(pets_images)
        all_annotations.extend(pets_anns)
    else:
        print(f"\nWarning: Oxford Pets directory not found: {pets_dir}")

    if not all_images:
        print("\nError: No data found!")
        sys.exit(1)

    # Create train/val split
    print(f"\n3. Creating train/val split (val_ratio={val_split})")
    train_imgs, train_anns, val_imgs, val_anns = create_val_split(
        all_images, all_annotations, val_split, seed
    )

    # Save datasets
    print(f"\n4. Saving to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use train2017/val2017 naming to match what train.py expects
    save_coco_format(train_imgs, train_anns,
                     output_dir / 'annotations' / 'instances_train2017.json', 'train')
    save_coco_format(val_imgs, val_anns,
                     output_dir / 'annotations' / 'instances_val2017.json', 'val')

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total images: {len(all_images)}")
    print(f"  Train: {len(train_imgs)}")
    print(f"  Val: {len(val_imgs)}")

    # Per-class breakdown
    train_class_counts = defaultdict(int)
    val_class_counts = defaultdict(int)
    for ann in train_anns:
        train_class_counts[ann['category_id']] += 1
    for ann in val_anns:
        val_class_counts[ann['category_id']] += 1

    print("\nPer-class annotations:")
    for cls_id, name in enumerate(CLASS_NAMES):
        print(f"  {name}: train={train_class_counts[cls_id]}, val={val_class_counts[cls_id]}")

    print(f"\nOutput directory: {output_dir}")
    print("Done!")


if __name__ == '__main__':
    main()

