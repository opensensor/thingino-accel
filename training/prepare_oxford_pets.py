#!/usr/bin/env python3
"""Convert Oxford-IIIT Pets dataset to COCO format and merge with existing COCO data."""

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
from PIL import Image

def parse_oxford_xml(xml_path):
    """Parse Oxford Pets XML annotation."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text.lower()  # 'cat' or 'dog'
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        objects.append({
            'class': name,
            'bbox': [xmin, ymin, xmax - xmin, ymax - ymin]  # COCO format: x, y, w, h
        })
    
    return width, height, objects

def convert_oxford_to_coco(oxford_dir, output_dir):
    """Convert Oxford Pets to COCO format."""
    oxford_dir = Path(oxford_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    images_dir = output_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Class mapping: cat=1, dog=2 (matching our TinyDet classes: 0=person, 1=cat, 2=dog)
    class_map = {'cat': 1, 'dog': 2}
    
    # COCO format structure
    coco = {
        'images': [],
        'annotations': [],
        'categories': [
            {'id': 0, 'name': 'person'},
            {'id': 1, 'name': 'cat'},
            {'id': 2, 'name': 'dog'}
        ]
    }
    
    xml_dir = oxford_dir / 'annotations' / 'xmls'
    img_dir = oxford_dir / 'images'
    
    ann_id = 1
    img_id = 100000  # Start high to avoid collision with COCO IDs
    
    cat_count = 0
    dog_count = 0
    
    for xml_file in sorted(xml_dir.glob('*.xml')):
        img_name = xml_file.stem + '.jpg'
        img_path = img_dir / img_name
        
        if not img_path.exists():
            continue
        
        try:
            width, height, objects = parse_oxford_xml(xml_file)
        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")
            continue
        
        if not objects:
            continue
        
        # Copy image to output
        dst_img = images_dir / img_name
        if not dst_img.exists():
            shutil.copy(img_path, dst_img)
        
        # Add image entry
        coco['images'].append({
            'id': img_id,
            'file_name': img_name,
            'width': width,
            'height': height
        })
        
        # Add annotations
        for obj in objects:
            cls = obj['class']
            if cls not in class_map:
                continue
            
            cat_id = class_map[cls]
            x, y, w, h = obj['bbox']
            
            coco['annotations'].append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': cat_id,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0
            })
            
            if cls == 'cat':
                cat_count += 1
            else:
                dog_count += 1
            
            ann_id += 1
        
        img_id += 1
    
    # Save annotations
    ann_file = output_dir / 'annotations.json'
    with open(ann_file, 'w') as f:
        json.dump(coco, f)
    
    print(f"Converted {len(coco['images'])} images with {len(coco['annotations'])} annotations")
    print(f"  Cats: {cat_count}, Dogs: {dog_count}")
    print(f"Saved to: {output_dir}")
    
    return coco

if __name__ == '__main__':
    oxford_dir = 'oxford_pets'
    output_dir = 'oxford_pets_coco'
    
    coco = convert_oxford_to_coco(oxford_dir, output_dir)

