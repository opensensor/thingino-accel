#!/usr/bin/env python3
"""
Test the exported ONNX model with ONNX Runtime.

This script loads the ONNX model and runs inference with dummy data
to verify the model structure is valid.

Note: The model was extracted from a quantized MGK file, so the
graph structure may not be complete for actual inference.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("Error: onnxruntime not found. Install with: pip install onnxruntime")
    sys.exit(1)


def get_providers():
    """Get available execution providers."""
    available = ort.get_available_providers()
    print(f"Available providers: {available}")
    
    # Prefer GPU providers
    preferred = []
    if 'CUDAExecutionProvider' in available:
        preferred.append('CUDAExecutionProvider')
    if 'ROCMExecutionProvider' in available:
        preferred.append('ROCMExecutionProvider')
    preferred.append('CPUExecutionProvider')
    
    return preferred


def test_model(model_path: Path, num_runs: int = 10):
    """Test the ONNX model with dummy input."""
    print(f"\n=== Testing ONNX Model: {model_path} ===\n")
    
    # Load model
    providers = get_providers()
    print(f"Using providers: {providers}")
    
    try:
        session = ort.InferenceSession(str(model_path), providers=providers)
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    # Get input/output info
    print("\nModel Inputs:")
    inputs = {}
    for inp in session.get_inputs():
        shape = inp.shape
        dtype = inp.type
        print(f"  {inp.name}: shape={shape}, type={dtype}")
        
        # Create dummy input with appropriate shape
        # Replace dynamic dims with reasonable values
        concrete_shape = []
        for dim in shape:
            if isinstance(dim, str) or dim is None:
                # Dynamic dimension - use 1 for batch, 16 for sequence, 64 for features
                if 'batch' in str(dim).lower():
                    concrete_shape.append(1)
                elif 'seq' in str(dim).lower():
                    concrete_shape.append(16)  # AEC models typically use small windows
                else:
                    concrete_shape.append(64)  # Common hidden size
            else:
                concrete_shape.append(dim)
        
        # Create numpy array with appropriate dtype
        if 'float' in dtype.lower():
            inputs[inp.name] = np.random.randn(*concrete_shape).astype(np.float32)
        elif 'int8' in dtype.lower():
            inputs[inp.name] = np.random.randint(-128, 127, concrete_shape, dtype=np.int8)
        elif 'uint8' in dtype.lower():
            inputs[inp.name] = np.random.randint(0, 255, concrete_shape, dtype=np.uint8)
        else:
            inputs[inp.name] = np.random.randn(*concrete_shape).astype(np.float32)
        
        print(f"    Created dummy input: shape={inputs[inp.name].shape}, dtype={inputs[inp.name].dtype}")
    
    print("\nModel Outputs:")
    output_names = []
    for out in session.get_outputs():
        print(f"  {out.name}: shape={out.shape}, type={out.type}")
        output_names.append(out.name)
    
    # Run inference
    print(f"\n=== Running Inference ({num_runs} iterations) ===")
    
    try:
        # Warmup
        _ = session.run(output_names, inputs)
        
        # Timed runs
        times = []
        for i in range(num_runs):
            start = time.perf_counter()
            outputs = session.run(output_names, inputs)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms
        
        print(f"\nInference Results:")
        for name, output in zip(output_names, outputs):
            print(f"  {name}: shape={output.shape}, dtype={output.dtype}")
            print(f"    min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
        
        print(f"\nPerformance:")
        print(f"  Average: {np.mean(times):.3f} ms")
        print(f"  Std Dev: {np.std(times):.3f} ms")
        print(f"  Min: {np.min(times):.3f} ms")
        print(f"  Max: {np.max(times):.3f} ms")
        
        return True
        
    except Exception as e:
        print(f"\nError during inference: {e}")
        print("\nThis is expected - the exported model represents structure only.")
        print("The MGK model uses custom ops that may not have ONNX equivalents.")
        return False


def main():
    parser = argparse.ArgumentParser(description='Test ONNX model with ONNX Runtime')
    parser.add_argument('model', type=Path, help='Path to ONNX model')
    parser.add_argument('-n', '--num-runs', type=int, default=10, help='Number of inference runs')
    args = parser.parse_args()
    
    if not args.model.exists():
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)
    
    success = test_model(args.model, args.num_runs)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

