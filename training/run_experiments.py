#!/usr/bin/env python3
"""
Multi-Experiment Training Runner

Runs a series of training experiments sequentially, logging results.
Designed for cloud GPU batch training to maximize $/hour value.

Usage:
    python run_experiments.py                    # Run all experiments
    python run_experiments.py --experiments 1 3  # Run specific experiments
    python run_experiments.py --dry-run          # Show what would run
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

EXPERIMENTS = {
    1: {
        "name": "baseline_100ep",
        "description": "Baseline model, 100 epochs",
        "config": {
            "epochs": 100,
            "batch_size": 64,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
        }
    },
    2: {
        "name": "wider_backbone",
        "description": "2x wider backbone (64 base channels)",
        "config": {
            "epochs": 100,
            "batch_size": 64,
            "learning_rate": 0.001,
            "base_channels": 64,  # Default is 32
        }
    },
    3: {
        "name": "lower_lr_cosine",
        "description": "Lower LR with cosine annealing",
        "config": {
            "epochs": 100,
            "batch_size": 64,
            "learning_rate": 0.0005,
            "lr_schedule": "cosine",
            "warmup_epochs": 10,
        }
    },
    4: {
        "name": "focal_loss",
        "description": "Focal loss for class imbalance",
        "config": {
            "epochs": 100,
            "batch_size": 64,
            "learning_rate": 0.001,
            "use_focal_loss": True,
            "focal_alpha": 0.25,
            "focal_gamma": 2.0,
        }
    },
    5: {
        "name": "best_combo",
        "description": "Best settings from experiments 1-4",
        "config": {
            "epochs": 150,
            "batch_size": 64,
            "learning_rate": 0.0005,
            "lr_schedule": "cosine",
            "warmup_epochs": 10,
            "use_focal_loss": True,
            "base_channels": 48,  # Moderate increase
        }
    },
}


def run_experiment(exp_id: int, exp: dict, output_base: Path):
    """Run a single experiment."""
    exp_name = exp["name"]
    output_dir = output_base / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT {exp_id}: {exp_name}")
    print(f"Description: {exp['description']}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Build command with config overrides
    cmd = ["python", "scripts/train.py", f"--output-dir={output_dir}"]
    for key, value in exp["config"].items():
        cmd.append(f"--{key.replace('_', '-')}={value}")
    
    start_time = time.time()
    
    # Run training
    result = subprocess.run(cmd, capture_output=False)
    
    elapsed = time.time() - start_time
    
    # Log results
    log_entry = {
        "experiment_id": exp_id,
        "name": exp_name,
        "config": exp["config"],
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60,
        "returncode": result.returncode,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Try to read metrics
    metrics_file = output_dir / "train_metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            log_entry["metrics"] = json.load(f)
    
    return log_entry


def main():
    parser = argparse.ArgumentParser(description="Run training experiments")
    parser.add_argument("--experiments", nargs="+", type=int, 
                        help="Which experiments to run (1-5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing")
    parser.add_argument("--output-base", type=str, default="runs/experiments",
                        help="Base directory for experiment outputs")
    args = parser.parse_args()
    
    exp_ids = args.experiments or list(EXPERIMENTS.keys())
    output_base = Path(args.output_base)
    
    print(f"Running {len(exp_ids)} experiments: {exp_ids}")
    print(f"Output base: {output_base}\n")
    
    if args.dry_run:
        for exp_id in exp_ids:
            exp = EXPERIMENTS[exp_id]
            print(f"[{exp_id}] {exp['name']}: {exp['description']}")
            print(f"    Config: {exp['config']}\n")
        return
    
    output_base.mkdir(parents=True, exist_ok=True)
    results = []
    
    for exp_id in exp_ids:
        result = run_experiment(exp_id, EXPERIMENTS[exp_id], output_base)
        results.append(result)
        
        # Save running results
        with open(output_base / "results.json", "w") as f:
            json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = "✓" if r["returncode"] == 0 else "✗"
        metrics = r.get("metrics", {})
        mAP = metrics.get("mAP", "N/A")
        print(f"{status} [{r['experiment_id']}] {r['name']}: mAP={mAP}, time={r['elapsed_minutes']:.1f}min")


if __name__ == "__main__":
    main()

