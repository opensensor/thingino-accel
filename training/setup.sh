#!/bin/bash
# Setup training environment for TinyDet
# Requires: Python 3.8+, CUDA 11.x or 12.x

set -e

echo "=== TinyDet Training Environment Setup ==="

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing PyTorch with CUDA support..."
# PyTorch 2.1+ with CUDA 12.1 (adjust if needed)
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "Installing additional dependencies..."
pip install \
    numpy \
    pillow \
    onnx \
    onnxruntime \
    tqdm \
    pycocotools

echo ""
echo "=== Verifying GPU setup ==="
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Device count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    props = torch.cuda.get_device_properties(i)
    print(f'    Memory: {props.total_memory / 1024**3:.1f} GB')
"

echo ""
echo "=== Setup complete ==="
echo "Activate with: source venv/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Download COCO dataset (or symlink existing)"
echo "  2. Filter to 3 classes: python coco_filter.py --coco-dir /path/to/coco"
echo "  3. Train: python train.py --data ./coco_3class --epochs 100"

