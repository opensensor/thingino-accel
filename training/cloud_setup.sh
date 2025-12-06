#!/bin/bash
# Cloud GPU Setup Script for TinyDet Training
# Run this after SSH into the cloud GPU instance
#
# Usage:
#   git clone git@github.com:opensensor/thingino-accel.git
#   cd thingino-accel/training
#   ./cloud_setup.sh

set -e

echo "=== TinyDet Cloud Training Setup ==="
echo ""

# Check for GPU
echo "Checking GPU..."
nvidia-smi || { echo "ERROR: No GPU detected!"; exit 1; }
echo ""

# Create and activate virtual environment
echo "Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pyyaml pillow tqdm onnx onnxruntime dvc[s3]

# Verify PyTorch CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Configure DVC remote (you'll need to set these env vars or edit)
echo ""
echo "Configuring DVC..."
if [ -z "$DO_SPACES_KEY" ] || [ -z "$DO_SPACES_SECRET" ]; then
    echo "WARNING: DO_SPACES_KEY and DO_SPACES_SECRET not set"
    echo "Set these environment variables, then run:"
    echo "  dvc remote modify --local do access_key_id \$DO_SPACES_KEY"
    echo "  dvc remote modify --local do secret_access_key \$DO_SPACES_SECRET"
else
    dvc remote modify --local do access_key_id "$DO_SPACES_KEY"
    dvc remote modify --local do secret_access_key "$DO_SPACES_SECRET"
    echo "DVC credentials configured"
fi

# Pull data
echo ""
echo "Pulling training data from DVC..."
dvc pull

# Create symlink for train2017 if needed
if [ ! -L coco/train2017 ] && [ -d train2017 ]; then
    echo "Creating coco/train2017 symlink..."
    mkdir -p coco
    ln -sf ../train2017 coco/train2017
fi

# Verify data
echo ""
echo "Verifying data..."
echo "Train images: $(ls coco/train2017/*.jpg 2>/dev/null | wc -l)"
echo "Val images: $(ls coco/val2017/*.jpg 2>/dev/null | wc -l)"
echo "Oxford pets: $(ls oxford_pets/images/*.jpg 2>/dev/null | wc -l)"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To run all experiments:"
echo "  source venv/bin/activate"
echo "  python run_experiments.py"
echo ""
echo "To run specific experiments (e.g., 1 and 3):"
echo "  python run_experiments.py --experiments 1 3"
echo ""
echo "To see what would run without executing:"
echo "  python run_experiments.py --dry-run"

