# TinyDet Training

Custom 4-class object detector (person/vehicle/cat/dog) designed from scratch for the Ingenic T41 NNA, optimized for home security camera use cases.

**Key Features:**
- Compact architecture (~202K params) fits in device memory
- NHWC format for efficient MXU vectorization
- Standard Conv2D only (no depthwise/separable)
- Single-scale output for fast inference
- Achieves **1.75s inference** with MXU+ORAM optimization

## Architecture

```
Input: 320x192x3 RGB (NHWC)

Backbone (8 conv layers):
  Stem:  3x3 s2 -> 160x96x16
  S1:    3x3 s1 -> 160x96x16
  S2:    3x3 s2 -> 80x48x32
  S3:    3x3 s1 -> 80x48x32
  S4:    3x3 s2 -> 40x24x64
  S5:    3x3 s1 -> 40x24x64
  S6:    3x3 s2 -> 20x12x64
  S7:    3x3 s1 -> 20x12x64

Head (3 conv layers):
  Conv:  3x3 s1 -> 20x12x64
  Conv:  3x3 s1 -> 20x12x32
  Out:   1x1 s1 -> 20x12x9

Output: 20x12 grid × 9 channels
  [0]   = objectness
  [1:5] = box (x_off, y_off, w, h)
  [5:9] = class scores (person, vehicle, cat, dog)
```

- **Parameters**: ~202K
- **Target inference**: ~1.8s on T41 (CPU MXU path)

## Quick Start

```bash
# 1. Setup environment
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision onnx onnxruntime pillow tqdm

# 2. Train balanced 4-class model (uses existing dataset)
python3 train_balanced_4class.py

# 3. Export to ONNX and compile to Mars
python3 test_model.py runs/balanced_4class/tinydet_best.pth

# 4. Deploy to device
cp tinydet_best.mars ~/nfs/
# On device: ./mars_detect model.mars input.jpg out.jpg
```

## Training Features

- **YOLO-style sampling**: Only train on cells with objects + sampled negatives (3:1 ratio)
- **Focal loss**: α=0.25, γ=2.0 for class imbalance
- **Class weighting**: Inverse frequency weighting
- **Cosine annealing LR**: 100 epochs, starting LR=1e-3

## Dataset

Balanced 4-class dataset from COCO + Oxford Pets:
```
Training (28,966 annotations, 19,877 images):
  person:  8,000 (27.6%)
  vehicle: 7,003 (24.2%)
  cat:     5,957 (20.6%)
  dog:     8,006 (27.6%)

Validation: COCO val2017 subset
```

## Key Files

| File | Description |
|------|-------------|
| `tinydet.py` | Model architecture |
| `train_balanced_4class.py` | Main training with YOLO-style sampling |
| `test_model.py` | Export to ONNX/Mars and test on image |
| `prepare_balanced_dataset.py` | Dataset preparation from COCO |

## Monitoring Training

```bash
# Watch training progress
tail -f train_balanced_v2.log

# Check current best loss
grep "New best" train_balanced_v2.log

# Test current best model
python3 test_model.py
```

## Output

After training, files are saved to `runs/balanced_4class/`:
- `tinydet_best.pth` - Best validation loss checkpoint
- `tinydet_final.pth` - Final epoch checkpoint

After running `test_model.py`:
- `tinydet_best.onnx` - ONNX format for reference
- `tinydet_best.mars` - Mars format for T41 device

## Comparison with YOLOv5

| Model | Params | Input | Inference (T41) | Classes |
|-------|--------|-------|-----------------|---------|
| **TinyDet** | 202K | 320×192 | 1.75s | 4 |
| YOLOv5n | 1.9M | 640×640 | ~30s+ | 80 |
| YOLOv5s | 7.2M | 640×640 | OOM | 80 |

TinyDet is purpose-built for T41's constrained memory and MXU architecture, achieving practical inference times where standard YOLO models are too slow.

## Files in This Directory

| Script | Purpose |
|--------|---------|
| `tinydet.py` | Model architecture definition |
| `train_improved.py` | Advanced training with CIoU, mosaic, focal loss |
| `train_balanced_4class.py` | Main training script for 4-class detector |
| `export_onnx.py` | Export PyTorch model to ONNX with Conv+BN fusion |
| `test_model.py` | Test model and export to Mars format |
| `visualize_detections.py` | Draw detection boxes on images |
| `coco_filter.py` | Filter COCO dataset for target classes |
| `prepare_*.py` | Dataset preparation scripts |

