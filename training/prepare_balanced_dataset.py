"""
Prepare a balanced dataset for cat/dog/person detection.

Combines:
1. COCO 2017 - undersampled to balance classes
2. Oxford-IIIT Pets - cats and dogs with bounding boxes

This creates a dataset with roughly equal representation of each class.
"""

import os
import json
import random
import urllib.request
import tarfile
from pathlib import Path
from collections import defaultdict

# Target counts per class (aim for balance)
TARGET_PER_CLASS = 15000  # 15K images per class

def download_oxford_pets(output_dir):
    """Download Oxford-IIIT Pets dataset."""
    pets_dir = Path(output_dir) / "oxford_pets"
    pets_dir.mkdir(parents=True, exist_ok=True)
    
    images_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    annots_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
    
    images_tar = pets_dir / "images.tar.gz"
    annots_tar = pets_dir / "annotations.tar.gz"
    
    if not (pets_dir / "images").exists():
        print("Downloading Oxford Pets images...")
        urllib.request.urlretrieve(images_url, images_tar)
        print("Extracting...")
        with tarfile.open(images_tar, "r:gz") as tar:
            tar.extractall(pets_dir)
    
    if not (pets_dir / "annotations").exists():
        print("Downloading Oxford Pets annotations...")
        urllib.request.urlretrieve(annots_url, annots_tar)
        print("Extracting...")
        with tarfile.open(annots_tar, "r:gz") as tar:
            tar.extractall(pets_dir)
    
    return pets_dir

def parse_oxford_pets(pets_dir):
    """Parse Oxford Pets into COCO format annotations."""
    from PIL import Image
    import xml.etree.ElementTree as ET
    
    annotations = []
    images = []
    
    # Cat breeds (first 12 are cats in alphabetical order)
    cat_breeds = {
        'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair',
        'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue',
        'Siamese', 'Sphynx'
    }
    
    xmls_dir = pets_dir / "annotations" / "xmls"
    images_dir = pets_dir / "images"
    
    img_id = 100000  # Start high to avoid COCO ID conflicts
    ann_id = 1000000
    
    for xml_file in xmls_dir.glob("*.xml"):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            filename = root.find("filename").text
            if not filename.endswith(".jpg"):
                filename += ".jpg"
            
            img_path = images_dir / filename
            if not img_path.exists():
                continue
            
            # Get image size
            size = root.find("size")
            width = int(size.find("width").text)
            height = int(size.find("height").text)
            
            # Determine if cat or dog based on breed name
            breed = filename.rsplit("_", 1)[0]
            is_cat = any(cat in breed for cat in cat_breeds)
            category_id = 1 if is_cat else 2  # 1=cat, 2=dog (will remap later)
            
            # Get bounding box
            obj = root.find("object")
            if obj is None:
                continue
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            
            bbox = [xmin, ymin, xmax - xmin, ymax - ymin]  # COCO format: [x, y, w, h]
            
            images.append({
                "id": img_id,
                "file_name": str(img_path),
                "width": width,
                "height": height
            })
            
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0
            })
            
            img_id += 1
            ann_id += 1
            
        except Exception as e:
            continue
    
    return images, annotations

def balance_coco(coco_dir, max_person=10000):
    """Load COCO and undersample person class."""
    ann_file = Path(coco_dir) / "annotations" / "instances_train2017.json"
    
    print(f"Loading COCO annotations from {ann_file}...")
    with open(ann_file) as f:
        coco = json.load(f)
    
    # COCO category IDs: person=1, cat=17, dog=18
    # Group images by which classes they contain
    img_classes = defaultdict(set)
    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        if cat_id in [1, 17, 18]:
            img_classes[ann["image_id"]].add(cat_id)
    
    # Separate images by class
    person_only = [img_id for img_id, cats in img_classes.items() if cats == {1}]
    cat_imgs = [img_id for img_id, cats in img_classes.items() if 17 in cats]
    dog_imgs = [img_id for img_id, cats in img_classes.items() if 18 in cats]
    
    print(f"COCO class distribution:")
    print(f"  Person-only images: {len(person_only)}")
    print(f"  Cat images: {len(cat_imgs)}")
    print(f"  Dog images: {len(dog_imgs)}")
    
    # Undersample person, keep all cats/dogs
    random.shuffle(person_only)
    selected_person = set(person_only[:max_person])
    selected_imgs = selected_person | set(cat_imgs) | set(dog_imgs)
    
    # Filter images and annotations
    filtered_images = [img for img in coco["images"] if img["id"] in selected_imgs]
    filtered_anns = [ann for ann in coco["annotations"] 
                     if ann["image_id"] in selected_imgs and ann["category_id"] in [1, 17, 18]]
    
    return filtered_images, filtered_anns, coco["categories"]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-dir", default="./coco", help="COCO dataset directory")
    parser.add_argument("--output-dir", default="./balanced_dataset", help="Output directory")
    parser.add_argument("--max-person", type=int, default=10000, help="Max person images")
    args = parser.parse_args()
    
    print("Step 1: Download Oxford Pets...")
    pets_dir = download_oxford_pets(args.output_dir)
    
    print("\nStep 2: Parse Oxford Pets annotations...")
    pets_images, pets_anns = parse_oxford_pets(pets_dir)
    print(f"  Found {len(pets_images)} pet images")
    
    print("\nStep 3: Balance COCO dataset...")
    coco_images, coco_anns, categories = balance_coco(args.coco_dir, args.max_person)
    print(f"  Selected {len(coco_images)} COCO images")
    
    print("\nDataset preparation complete!")
    print(f"  Total images: {len(pets_images) + len(coco_images)}")

