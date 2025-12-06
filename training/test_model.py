#!/usr/bin/env python3
"""
Test script: Export trained model to ONNX/Mars and test on cat image.
Usage: python test_model.py [checkpoint_path] [--test-image path]
"""
import os
import sys
import argparse
import subprocess
import numpy as np
import torch
from pathlib import Path

# Add training dir to path
sys.path.insert(0, str(Path(__file__).parent))
from tinydet import TinyDet

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

def export_onnx(checkpoint_path, output_path, num_classes=4):
    """Export PyTorch checkpoint to ONNX"""
    print(f"Loading checkpoint: {checkpoint_path}")
    model = TinyDet(num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    
    dummy_input = torch.randn(1, 3, 192, 320)
    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model, dummy_input, output_path,
        input_names=['input'], output_names=['output'],
        opset_version=11
    )
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    return model

def compile_mars(onnx_path, mars_path):
    """Compile ONNX to Mars format"""
    base_dir = Path(__file__).parent.parent
    compiler_script = base_dir / "mars-compiler" / "onnx2mars.py"
    mars_compiler = base_dir / "mars-compiler" / "target" / "release" / "mars"
    
    # Step 1: Convert ONNX to JSON+BIN
    json_path = onnx_path.replace('.onnx', '.json')
    print(f"Converting ONNX to JSON: {json_path}")
    result = subprocess.run(['python3', str(compiler_script), onnx_path, '-o', mars_path],
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(result.stdout)
    
    # Step 2: Compile JSON to Mars
    print(f"Compiling to Mars: {mars_path}")
    result = subprocess.run([str(mars_compiler), '--input', json_path, '--output', mars_path],
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(result.stdout)
    return True

def test_onnx(onnx_path, test_image_path):
    """Test ONNX model on image"""
    import onnxruntime as ort
    from PIL import Image
    
    print(f"\nTesting ONNX model on: {test_image_path}")
    
    # Load model
    sess = ort.InferenceSession(onnx_path)
    
    # Load and preprocess image
    img = Image.open(test_image_path).convert('RGB')
    img = img.resize((320, 192), Image.BILINEAR)
    img_np = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = (img_np - mean) / std
    img_np = img_np.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    
    # Run inference
    output = sess.run(None, {'input': img_np})[0][0]  # [9, 12, 20]
    
    # Parse output
    obj = sigmoid(output[0])
    classes = sigmoid(output[5:9])
    class_names = ['person', 'vehicle', 'cat', 'dog']
    
    print(f"  Output shape: {output.shape}")
    print(f"  Objectness: max={obj.max():.3f}, mean={obj.mean():.3f}")
    
    # Find best detections
    detections = []
    for y in range(output.shape[1]):
        for x in range(output.shape[2]):
            obj_p = obj[y, x]
            for c in range(4):
                class_p = classes[c, y, x]
                conf = obj_p * class_p
                if conf > 0.1:
                    detections.append((conf, class_names[c], y, x))
    
    detections.sort(reverse=True)
    print(f"\n  Top detections (conf > 10%):")
    for conf, name, y, x in detections[:10]:
        print(f"    [{y},{x}] {name}: {conf*100:.1f}%")
    
    if not detections:
        print("    No detections above 10% confidence")
        # Show best anyway
        best_conf = 0
        best_info = None
        for y in range(output.shape[1]):
            for x in range(output.shape[2]):
                obj_p = obj[y, x]
                for c in range(4):
                    class_p = classes[c, y, x]
                    conf = obj_p * class_p
                    if conf > best_conf:
                        best_conf = conf
                        best_info = (y, x, c, class_names[c])
        if best_info:
            y, x, c, name = best_info
            print(f"    Best: [{y},{x}] {name}: {best_conf*100:.1f}%")

def main():
    parser = argparse.ArgumentParser(description='Test trained model')
    parser.add_argument('checkpoint', nargs='?', default='runs/balanced_4class/tinydet_best.pth',
                       help='Path to checkpoint')
    parser.add_argument('--test-image', default=None, help='Test image path')
    parser.add_argument('--output-dir', default='.', help='Output directory')
    args = parser.parse_args()
    
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"Error: Checkpoint not found: {checkpoint}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    name = checkpoint.stem
    onnx_path = str(output_dir / f"{name}.onnx")
    mars_path = str(output_dir / f"{name}.mars")
    
    # Export and compile
    model = export_onnx(str(checkpoint), onnx_path)
    compile_mars(onnx_path, mars_path)
    
    # Test if image provided
    test_image = args.test_image
    if not test_image:
        # Try default locations
        for path in ['/home/matteius/nfs/cat1_320x192.ppm', 
                     '/home/matteius/nfs/cat1_320x192.jpg',
                     'cat1_320x192.ppm']:
            if os.path.exists(path):
                test_image = path
                break
    
    if test_image and os.path.exists(test_image):
        test_onnx(onnx_path, test_image)
    
    print(f"\n✓ Output files:")
    print(f"  ONNX: {onnx_path}")
    print(f"  Mars: {mars_path}")

if __name__ == '__main__':
    main()

