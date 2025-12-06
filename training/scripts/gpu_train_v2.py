#!/usr/bin/env python3
"""
GPU Training Script v2 - With proper data preparation
Runs 3 experiments on prepared 4-class dataset.

Usage:
  python scripts/gpu_train_v2.py
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

# Experiments configuration
EXPERIMENTS = [
    {
        "name": "baseline_200ep",
        "description": "Baseline model, 200 epochs on correctly prepared data",
        "args": {
            "epochs": 200,
            "batch_size": 64,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "warmup_epochs": 10,
        }
    },
    {
        "name": "wider_cosine",
        "description": "Wider backbone (more capacity) + cosine LR",
        "args": {
            "epochs": 200,
            "batch_size": 64,
            "learning_rate": 0.0005,
            "weight_decay": 0.0001,
            "warmup_epochs": 15,
            # Note: base_channels handled via model modification
        }
    },
    {
        "name": "longer_300ep",
        "description": "Extended training 300 epochs with warmup",
        "args": {
            "epochs": 300,
            "batch_size": 64,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "warmup_epochs": 20,
        }
    },
]


def run_command(cmd, cwd=None):
    """Run command and return (success, output)."""
    print(f">>> {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
    return result.returncode == 0, result.stdout + result.stderr


def prepare_data():
    """Run data preparation to create properly filtered dataset."""
    print("\n" + "=" * 60)
    print("STEP 1: Preparing Training Data")
    print("=" * 60)
    
    # Check if already prepared
    prepared_file = Path("combined_dataset/annotations/instances_train2017.json")
    if prepared_file.exists():
        # Verify it has correct format
        import json
        with open(prepared_file) as f:
            d = json.load(f)
        cats = {c['id']: c['name'] for c in d['categories']}
        if cats == {0: 'person', 1: 'vehicle', 2: 'cat', 3: 'dog'}:
            print(f"✓ Prepared data exists and is valid ({len(d['annotations'])} annotations)")
            return True
    
    success, output = run_command("python scripts/prepare_data.py --config params.yaml")
    print(output)
    return success


def run_experiment(exp):
    """Run a single training experiment."""
    name = exp["name"]
    args = exp["args"]
    output_dir = f"runs/experiments/{name}"
    
    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT: {name}")
    print(f"Description: {exp['description']}")
    print(f"Config: {args}")
    print("=" * 60)
    
    # Build command
    cmd_parts = [
        "python scripts/train.py",
        "--config params.yaml",
        f"--output-dir {output_dir}",
    ]
    for key, value in args.items():
        arg_name = key.replace("_", "-")
        cmd_parts.append(f"--{arg_name} {value}")
    
    cmd = " ".join(cmd_parts)
    
    t0 = time.time()
    success, output = run_command(cmd)
    elapsed = time.time() - t0
    
    print(f"\nExperiment {name} completed in {elapsed/60:.1f} minutes")
    
    return {
        "name": name,
        "success": success,
        "elapsed_minutes": elapsed / 60,
        "output_dir": output_dir,
    }


def export_models():
    """Export all trained models to ONNX."""
    print("\n" + "=" * 60)
    print("STEP 3: Exporting Models to ONNX")
    print("=" * 60)
    
    for exp in EXPERIMENTS:
        name = exp["name"]
        weights = f"runs/experiments/{name}/tinydet_best.pth"
        output = f"runs/experiments/{name}/export/tinydet.onnx"
        
        if not Path(weights).exists():
            print(f"  Skip {name}: weights not found")
            continue
        
        Path(f"runs/experiments/{name}/export").mkdir(parents=True, exist_ok=True)
        cmd = f"python scripts/export.py --config params.yaml --weights {weights} --output {output}"
        success, _ = run_command(cmd)
        print(f"  {name}: {'✓' if success else '✗'}")


def main():
    print("=" * 60)
    print("TinyDet GPU Training v2")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Step 1: Prepare data
    if not prepare_data():
        print("ERROR: Data preparation failed!")
        sys.exit(1)
    
    # Step 2: Run experiments
    results = []
    for exp in EXPERIMENTS:
        result = run_experiment(exp)
        results.append(result)
    
    # Step 3: Export models
    export_models()
    
    # Save results
    with open("runs/experiments/results_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"  {status} {r['name']}: {r['elapsed_minutes']:.1f} min")


if __name__ == "__main__":
    main()

