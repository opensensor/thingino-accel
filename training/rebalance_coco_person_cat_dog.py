#!/usr/bin/env python3
"""
Rebalance COCO dataset by undersampling person class to match cat/dog counts.
"""
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description='Rebalance COCO annotations')
    parser.add_argument('input', help='Input COCO JSON file')
    parser.add_argument('-o', '--output', help='Output COCO JSON file')
    parser.add_argument('--person-ratio', type=float, default=2.0,
                        help='Ratio of person annotations to max(cat, dog) count (default: 2.0)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Load annotations
    print(f"Loading {args.input}...")
    with open(args.input) as f:
        data = json.load(f)
    
    # Build category name->id mapping
    cat_name_to_id = {c['name']: c['id'] for c in data['categories']}
    cat_id_to_name = {c['id']: c['name'] for c in data['categories']}
    
    print(f"Categories: {cat_name_to_id}")
    
    # Get IDs for our target classes
    person_id = cat_name_to_id.get('person')
    cat_id = cat_name_to_id.get('cat')
    dog_id = cat_name_to_id.get('dog')
    
    if None in (person_id, cat_id, dog_id):
        print(f"ERROR: Missing category. person={person_id}, cat={cat_id}, dog={dog_id}")
        return
    
    # Group annotations by category
    by_category = defaultdict(list)
    for ann in data['annotations']:
        by_category[ann['category_id']].append(ann)
    
    print("\nBefore balancing:")
    for cat_id_iter, anns in sorted(by_category.items()):
        name = cat_id_to_name.get(cat_id_iter, f"unknown_{cat_id_iter}")
        print(f"  {name}: {len(anns)}")
    
    # Calculate target person count
    cat_count = len(by_category[cat_id])
    dog_count = len(by_category[dog_id])
    person_count = len(by_category[person_id])
    
    max_pet_count = max(cat_count, dog_count)
    target_person = int(max_pet_count * args.person_ratio)
    
    print(f"\nTarget person count: {target_person} ({args.person_ratio}x of max({cat_count}, {dog_count}))")
    
    # Undersample persons
    if person_count > target_person:
        by_category[person_id] = random.sample(by_category[person_id], target_person)
        print(f"Undersampled person: {person_count} -> {target_person}")
    
    # Combine all annotations
    balanced_annotations = []
    for cat_id_iter in by_category:
        balanced_annotations.extend(by_category[cat_id_iter])
    
    # Get set of image IDs still in use
    used_image_ids = set(ann['image_id'] for ann in balanced_annotations)
    
    # Filter images to only those with remaining annotations
    filtered_images = [img for img in data['images'] if img['id'] in used_image_ids]
    
    print("\nAfter balancing:")
    final_counts = defaultdict(int)
    for ann in balanced_annotations:
        final_counts[ann['category_id']] += 1
    for cat_id_iter, count in sorted(final_counts.items()):
        name = cat_id_to_name.get(cat_id_iter, f"unknown_{cat_id_iter}")
        print(f"  {name}: {count}")
    
    print(f"\nTotal annotations: {len(balanced_annotations)}")
    print(f"Total images: {len(filtered_images)}")
    
    # Update data
    data['annotations'] = balanced_annotations
    data['images'] = filtered_images
    
    # Output path
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input)
        output_path = input_path.parent / f"{input_path.stem}_balanced.json"
    
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(data, f)
    
    print("Done!")


if __name__ == '__main__':
    main()

