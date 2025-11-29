#!/usr/bin/env python3
"""
Extract and apply INT8 weights from MGK file to create a functional model.

The AEC_T41_16K_NS_OUT_UC.mgk model appears to be an Audio Echo Cancellation
model using GRU layers with INT8 quantized weights.
"""

import argparse
import struct
import numpy as np
from pathlib import Path


def extract_weights(mgk_path: Path) -> dict:
    """Extract weight data from MGK file."""
    with open(mgk_path, 'rb') as f:
        # Weight data starts at 0x79294
        f.seek(0x79294)
        raw_weights = f.read(153644)
    
    # Dense weight sections (low sparsity)
    sections = {
        'section_0': (0x0, 12288),      # Dense, likely first layer input weights
        'section_2': (0x4400, 39936),   # Dense, main GRU weights
        'section_3': (0xe000, 15360),   # Dense, more weights
        'section_6': (0x17000, 36864),  # Dense, second GRU weights
    }
    
    weights = {}
    for name, (offset, size) in sections.items():
        data = raw_weights[offset:offset + size]
        # Convert to signed int8
        arr = np.frombuffer(data, dtype=np.int8).copy()
        weights[name] = arr
        print(f"  {name}: {len(arr):,} bytes, mean={arr.mean():.2f}, std={arr.std():.2f}")
    
    return weights, raw_weights


def create_gru_model(weights: dict, hidden_size: int = 96, input_size: int = 80):
    """Create a PyTorch GRU model with extracted weights."""
    import torch
    import torch.nn as nn

    class AECModel(nn.Module):
        """Simple AEC model with GRU layers."""

        def __init__(self, input_size, hidden_size, num_layers=2):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=False
            )
            self.fc = nn.Linear(hidden_size, input_size)
            self.register_buffer('weight_scale', torch.tensor(1.0 / 127.0))

        def forward(self, x, hidden=None):
            if hidden is None:
                hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            out, hidden = self.gru(x, hidden)
            out = self.fc(out)
            return out, hidden

    model = AECModel(input_size, hidden_size, num_layers=2)

    # Load INT8 weights into model
    print(f"\nLoading INT8 weights into model...")

    # Concatenate all weight sections
    all_weights = np.concatenate([weights['section_0'], weights['section_2'],
                                   weights['section_3'], weights['section_6']])

    # Expected sizes for 2-layer GRU
    gru_ih_l0 = 3 * hidden_size * input_size  # 23,040
    gru_hh_l0 = 3 * hidden_size * hidden_size  # 27,648
    gru_ih_l1 = 3 * hidden_size * hidden_size  # 27,648 (input is hidden_size for layer 1)
    gru_hh_l1 = 3 * hidden_size * hidden_size  # 27,648

    offset = 0
    scale = 1.0 / 127.0  # INT8 dequantization scale

    with torch.no_grad():
        # Layer 0: weight_ih
        size = gru_ih_l0
        if offset + size <= len(all_weights):
            w = all_weights[offset:offset + size].astype(np.float32) * scale
            model.gru.weight_ih_l0.copy_(torch.from_numpy(w.reshape(3 * hidden_size, input_size)))
            print(f"  Loaded weight_ih_l0: {size:,} bytes")
            offset += size

        # Layer 0: weight_hh
        size = gru_hh_l0
        if offset + size <= len(all_weights):
            w = all_weights[offset:offset + size].astype(np.float32) * scale
            model.gru.weight_hh_l0.copy_(torch.from_numpy(w.reshape(3 * hidden_size, hidden_size)))
            print(f"  Loaded weight_hh_l0: {size:,} bytes")
            offset += size

        # Layer 1: weight_ih
        size = gru_ih_l1
        if offset + size <= len(all_weights):
            w = all_weights[offset:offset + size].astype(np.float32) * scale
            model.gru.weight_ih_l1.copy_(torch.from_numpy(w.reshape(3 * hidden_size, hidden_size)))
            print(f"  Loaded weight_ih_l1: {size:,} bytes")
            offset += size

        # Layer 1: weight_hh
        size = gru_hh_l1
        if offset + size <= len(all_weights):
            w = all_weights[offset:offset + size].astype(np.float32) * scale
            model.gru.weight_hh_l1.copy_(torch.from_numpy(w.reshape(3 * hidden_size, hidden_size)))
            print(f"  Loaded weight_hh_l1: {size:,} bytes")
            offset += size

    print(f"  Total loaded: {offset:,} / {len(all_weights):,} bytes")

    return model


def test_model(model, input_size: int):
    """Test the model with random input."""
    import torch
    
    print("\n=== Testing Model ===")
    
    # Create dummy input
    batch_size = 1
    seq_len = 16
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Run forward pass
    model.eval()
    with torch.no_grad():
        out, hidden = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Hidden shape: {hidden.shape}")
    print(f"Output range: [{out.min():.4f}, {out.max():.4f}]")


def main():
    parser = argparse.ArgumentParser(description='Extract weights from MGK file')
    parser.add_argument('mgk_file', type=Path, help='Path to MGK file')
    parser.add_argument('--hidden', type=int, default=96, help='GRU hidden size')
    parser.add_argument('--input', type=int, default=80, help='Input feature size')
    args = parser.parse_args()
    
    print(f"=== Extracting weights from {args.mgk_file} ===\n")
    weights, raw = extract_weights(args.mgk_file)
    
    print(f"\n=== Creating GRU model (hidden={args.hidden}, input={args.input}) ===")
    model = create_gru_model(weights, args.hidden, args.input)
    
    test_model(model, args.input)
    
    # Save weights for later use
    np.savez('mgk_weights.npz', **weights)
    print(f"\nWeights saved to mgk_weights.npz")


if __name__ == '__main__':
    main()

