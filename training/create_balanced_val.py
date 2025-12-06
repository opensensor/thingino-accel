#!/usr/bin/env python3
"""
Create a balanced validation set with more cat/dog samples.
Combines COCO val with Oxford Pets for better class balance.
"""
import json
import os
import random
from pathlib import Path
from collections import defaultdict

# Target: ~500 samples per class for validation
TARGET_PER_CLASS = 500

def main():
    # Load COCO val annotations
    with open('balanced_4class/annotations/instances_val.json') as f:
        coco_val = json.load(f)
    
    # Load Oxford Pets annotations
    with open('oxford_pets_coco/annotations.json') as f:
        pets = json.load(f)
    
    print("Original COCO val:")
    coco_class_counts = defaultdict(int)
    for ann in coco_val['annotations']:
        coco_class_counts[ann['category_id']] += 1
    for cid, count in sorted(coco_class_counts.items()):
        print(f"  Class {cid}: {count}")
    
    print("\nOxford Pets:")
    pets_class_counts = defaultdict(int)
    for ann in pets['annotations']:
        pets_class_counts[ann['category_id']] += 1
    for cid, count in sorted(pets_class_counts.items()):
        print(f"  Class {cid}: {count}")
    
    # Group images by class
    coco_images_by_class = defaultdict(list)
    coco_img_anns = defaultdict(list)
    for ann in coco_val['annotations']:
        coco_img_anns[ann['image_id']].append(ann)
    
    for img in coco_val['images']:
        anns = coco_img_anns[img['id']]
        classes = set(ann['category_id'] for ann in anns)
        for c in classes:
            coco_images_by_class[c].append(img['id'])
    
    pets_images_by_class = defaultdict(list)
    pets_img_anns = defaultdict(list)
    for ann in pets['annotations']:
        pets_img_anns[ann['image_id']].append(ann)
    
    for img in pets['images']:
        anns = pets_img_anns[img['id']]
        classes = set(ann['category_id'] for ann in anns)
        for c in classes:
            pets_images_by_class[c].append(img['id'])
    
    # Build balanced validation set
    # For person/vehicle: sample from COCO
    # For cat/dog: combine COCO + Oxford Pets
    
    selected_coco_ids = set()
    selected_pets_ids = set()
    
    # Person (class 0): sample TARGET_PER_CLASS from COCO
    person_ids = list(set(coco_images_by_class[0]))
    random.shuffle(person_ids)
    selected_coco_ids.update(person_ids[:TARGET_PER_CLASS])
    
    # Vehicle (class 1): sample TARGET_PER_CLASS from COCO
    vehicle_ids = list(set(coco_images_by_class[1]))
    random.shuffle(vehicle_ids)
    selected_coco_ids.update(vehicle_ids[:TARGET_PER_CLASS])
    
    # Cat (class 2 in balanced_4class, class 1 in oxford_pets): all from COCO + sample from Oxford Pets
    cat_coco_ids = list(set(coco_images_by_class[2]))
    selected_coco_ids.update(cat_coco_ids)
    cat_pets_ids = list(set(pets_images_by_class[1]))  # Oxford Pets uses class 1 for cat
    random.shuffle(cat_pets_ids)
    needed = max(0, TARGET_PER_CLASS - len(cat_coco_ids))
    selected_pets_ids.update(cat_pets_ids[:needed])

    # Dog (class 3 in balanced_4class, class 2 in oxford_pets): all from COCO + sample from Oxford Pets
    dog_coco_ids = list(set(coco_images_by_class[3]))
    selected_coco_ids.update(dog_coco_ids)
    dog_pets_ids = list(set(pets_images_by_class[2]))  # Oxford Pets uses class 2 for dog
    random.shuffle(dog_pets_ids)
    needed = max(0, TARGET_PER_CLASS - len(dog_coco_ids))
    selected_pets_ids.update(dog_pets_ids[:needed])
    
    # Build new annotation file
    coco_id_to_img = {img['id']: img for img in coco_val['images']}
    pets_id_to_img = {img['id']: img for img in pets['images']}
    
    new_images = []
    new_annotations = []
    ann_id = 1
    
    # Add COCO images
    for img_id in selected_coco_ids:
        img = coco_id_to_img[img_id].copy()
        new_images.append(img)
        for ann in coco_img_anns[img_id]:
            new_ann = ann.copy()
            new_ann['id'] = ann_id
            new_annotations.append(new_ann)
            ann_id += 1
    
    # Add Oxford Pets images (need to remap IDs to avoid conflicts)
    # Also remap category IDs: Oxford Pets cat=1->2, dog=2->3
    pets_to_balanced_class = {1: 2, 2: 3}  # cat: 1->2, dog: 2->3
    max_coco_id = max(img['id'] for img in coco_val['images']) + 1
    for img_id in selected_pets_ids:
        img = pets_id_to_img[img_id].copy()
        new_img_id = max_coco_id + img_id
        img['id'] = new_img_id
        new_images.append(img)
        for ann in pets_img_anns[img_id]:
            new_ann = ann.copy()
            new_ann['id'] = ann_id
            new_ann['image_id'] = new_img_id
            # Remap category ID
            if new_ann['category_id'] in pets_to_balanced_class:
                new_ann['category_id'] = pets_to_balanced_class[new_ann['category_id']]
            new_annotations.append(new_ann)
            ann_id += 1
    
    # Create output
    output = {
        'images': new_images,
        'annotations': new_annotations,
        'categories': coco_val['categories']
    }
    
    # Save
    os.makedirs('balanced_4class/annotations', exist_ok=True)
    with open('balanced_4class/annotations/instances_val_balanced.json', 'w') as f:
        json.dump(output, f)
    
    # Print stats
    print("\nNew balanced validation set:")
    print(f"  Images: {len(new_images)}")
    print(f"  Annotations: {len(new_annotations)}")
    class_counts = defaultdict(int)
    for ann in new_annotations:
        class_counts[ann['category_id']] += 1
    for cid, count in sorted(class_counts.items()):
        print(f"  Class {cid}: {count}")
    
    print("\nSaved to balanced_4class/annotations/instances_val_balanced.json")

if __name__ == '__main__':
    random.seed(42)
    main()

