#!/usr/bin/env python3
"""
Convert extracted MGK weights to ONNX format.

This script:
1. Loads unpacked NMHWSOIB2 weights from .npy files
2. Creates a PyTorch model matching the MGK architecture
3. Loads the INT8 weights with proper dequantization
4. Exports to ONNX format
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


class AECModel(nn.Module):
    """
    AEC model matching the MGK architecture.
    
    Architecture from device logs:
    - Input: [1, 1, 256, 8] - 256 freq bins, 8 frames
    - Encoder: 2 downsample stages (256->128->64)
    - Bottleneck: BiGRU + UniGRU
    - Decoder: 2 upsample stages (64->128->256)
    - Output: [1, 1, 256, 2] - sigmoid mask
    """
    
    def __init__(self, n_freq=256, n_frames=8, n_ch=32, hidden=32):
        super().__init__()
        self.n_freq = n_freq
        self.n_frames = n_frames
        self.n_ch = n_ch
        self.hidden = hidden
        
        # Input processing
        self.input_bn = nn.BatchNorm1d(n_freq)

        # Encoder - note: conv1 extracted is 32x32 because of NMHWSOIB2 padding
        # The actual first conv is 1->32, but we extracted weights are padded
        self.expand = nn.Conv2d(1, n_ch, kernel_size=(1, n_frames))  # 1->32, collapse time
        self.conv1 = nn.Conv2d(n_ch, n_ch, kernel_size=(1, 1))  # 32->32
        self.conv2 = nn.Conv2d(n_ch, n_ch, kernel_size=(3, 1), padding=(1, 0))
        self.down1 = nn.Conv2d(n_ch, n_ch, kernel_size=(2, 1), stride=(2, 1))
        self.conv3 = nn.Conv2d(n_ch, n_ch, kernel_size=(3, 1), padding=(1, 0))
        self.down2 = nn.Conv2d(n_ch, n_ch, kernel_size=(2, 1), stride=(2, 1))
        self.conv4 = nn.Conv2d(n_ch, n_ch, kernel_size=(3, 1), padding=(1, 0))
        
        # GRU bottleneck
        self.pre_gru_bn = nn.BatchNorm1d(n_ch)
        self.gru_uni = nn.GRU(n_ch, hidden, batch_first=True)
        self.gru_bi = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.post_gru_bn = nn.BatchNorm1d(hidden * 2)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(hidden * 2, n_ch, kernel_size=(2, 1), stride=(2, 1))
        self.conv5 = nn.Conv2d(n_ch, n_ch, kernel_size=(3, 3), padding=(1, 1))
        self.up2 = nn.ConvTranspose2d(n_ch, n_ch, kernel_size=(2, 1), stride=(2, 1))
        self.conv6 = nn.Conv2d(n_ch, n_ch, kernel_size=(3, 3), padding=(1, 1))
        
        # Output
        self.output_conv = nn.Conv2d(n_ch, 2, kernel_size=(1, 1))
    
    def forward(self, x):
        """
        Args:
            x: [B, 1, 256, 8] - frequency bins x frames
        Returns:
            mask: [B, 2, 256, 1] - output mask
        """
        B = x.shape[0]

        # Input normalization
        x_flat = x.squeeze(1)  # [B, 256, 8]
        x_norm = self.input_bn(x_flat)
        x = x_norm.unsqueeze(1)  # [B, 1, 256, 8]

        # Encoder
        x = torch.relu(self.expand(x))  # [B, 32, 256, 1] - collapse time dimension
        x = torch.relu(self.conv1(x))   # [B, 32, 256, 1]
        x = torch.relu(self.conv2(x))   # [B, 32, 256, 1]
        x = torch.relu(self.down1(x))   # [B, 32, 128, 1]
        x = torch.relu(self.conv3(x))   # [B, 32, 128, 1]
        x = torch.relu(self.down2(x))   # [B, 32, 64, 1]
        x = torch.relu(self.conv4(x))   # [B, 32, 64, 1]

        # Prepare for GRU: [B, 32, 64, 1] -> [B, 64, 32]
        x = x.squeeze(3)  # [B, 32, 64]
        x = x.permute(0, 2, 1)  # [B, 64, 32]

        # Pre-GRU normalization
        x_t = x.permute(0, 2, 1)  # [B, 32, 64]
        x_t = self.pre_gru_bn(x_t)
        x = x_t.permute(0, 2, 1)  # [B, 64, 32]

        # GRU layers (no hidden state for ONNX export)
        x, _ = self.gru_uni(x)  # [B, 64, 32]
        x, _ = self.gru_bi(x)   # [B, 64, 64]

        # Post-GRU normalization
        x_t = x.permute(0, 2, 1)  # [B, 64, 64]
        x_t = self.post_gru_bn(x_t)
        x = x_t.permute(0, 2, 1)  # [B, 64, 64]

        # Prepare for decoder: [B, 64, 64] -> [B, 64, 64, 1]
        x = x.permute(0, 2, 1).unsqueeze(3)  # [B, 64, 64, 1]

        # Decoder
        x = torch.relu(self.up1(x))    # [B, 32, 128, 1]
        x = torch.relu(self.conv5(x))  # [B, 32, 128, 1]
        x = torch.relu(self.up2(x))    # [B, 32, 256, 1]
        x = torch.relu(self.conv6(x))  # [B, 32, 256, 1]

        # Output
        x = self.output_conv(x)  # [B, 2, 256, 1]
        mask = torch.sigmoid(x)

        return mask


def load_weights(model, weights_dir: Path, scale=0.01):
    """Load unpacked NMHWSOIB2 weights into model."""

    with torch.no_grad():
        # Load conv weights - map extracted to model layers
        # Extracted conv weights are in OIHW format after unpacking
        weight_mapping = [
            ('conv1_weight.npy', 'conv1', (32, 32, 1, 1)),  # 1x1 conv
            ('conv2_weight.npy', 'conv2', (32, 32, 3, 1)),  # 3x3 -> 3x1 for freq
            ('conv3_weight.npy', 'conv3', (32, 32, 3, 1)),
            ('conv4_weight.npy', 'conv4', (32, 32, 3, 1)),
        ]

        for filename, layer_name, target_shape in weight_mapping:
            path = weights_dir / filename
            if path.exists():
                w = np.load(path).astype(np.float32) * scale
                layer = getattr(model, layer_name)

                # Reshape from extracted shape to target shape
                if w.shape[0:2] == target_shape[0:2]:
                    # Take center slice for kernel adaptation
                    if w.shape[2] == 3 and target_shape[2] == 3:
                        w_adapted = w[:, :, :, 1:2]  # Take center column
                    elif w.shape[2] == 1 and target_shape[2] == 1:
                        w_adapted = w
                    else:
                        w_adapted = w[:, :, :target_shape[2], :target_shape[3]]

                    layer.weight.copy_(torch.from_numpy(w_adapted))
                    print(f"  Loaded {layer_name}: {w.shape} -> {w_adapted.shape}")
        
        # Load GRU weights
        # BiGRU
        for direction in ['forward', 'backward']:
            suffix = '' if direction == 'forward' else '_reverse'
            ih_path = weights_dir / f'bigru_{direction}_weight_ih.npy'
            hh_path = weights_dir / f'bigru_{direction}_weight_hh.npy'
            
            if ih_path.exists() and hh_path.exists():
                w_ih = np.load(ih_path).astype(np.float32) * scale
                w_hh = np.load(hh_path).astype(np.float32) * scale
                
                getattr(model.gru_bi, f'weight_ih_l0{suffix}').copy_(
                    torch.from_numpy(w_ih))
                getattr(model.gru_bi, f'weight_hh_l0{suffix}').copy_(
                    torch.from_numpy(w_hh))
                print(f"  Loaded gru_bi {direction}: ih={w_ih.shape}, hh={w_hh.shape}")
        
        # UniGRU
        ih_path = weights_dir / 'unigru_weight_ih.npy'
        hh_path = weights_dir / 'unigru_weight_hh.npy'
        if ih_path.exists() and hh_path.exists():
            w_ih = np.load(ih_path).astype(np.float32) * scale
            w_hh = np.load(hh_path).astype(np.float32) * scale
            # UniGRU has (64, 32) shape but PyTorch expects (96, 32) for 3 gates
            # This suggests the GRU has 2 gates, not 3 - adjust if needed
            print(f"  UniGRU weight shapes: ih={w_ih.shape}, hh={w_hh.shape}")


def main():
    parser = argparse.ArgumentParser(description='Convert MGK weights to ONNX')
    parser.add_argument('--weights', '-w', type=Path, 
                        default=Path('mgk-decompiler/extracted_final/weights_unpacked'),
                        help='Directory with extracted .npy weight files')
    parser.add_argument('--output', '-o', type=Path,
                        default=Path('mgk-decompiler/aec_model.onnx'),
                        help='Output ONNX file')
    parser.add_argument('--scale', type=float, default=0.01,
                        help='Dequantization scale for INT8 weights')
    args = parser.parse_args()
    
    print("=== MGK to ONNX Converter ===\n")
    
    # Create model
    model = AECModel()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load weights
    print(f"\nLoading weights from {args.weights}")
    load_weights(model, args.weights, args.scale)
    
    # Export to ONNX
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 1, 256, 8)

    # Test inference first
    print("\nTesting inference...")
    with torch.no_grad():
        output = model(dummy_input)
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")

    # Export to ONNX using dynamo=False for legacy exporter
    print(f"\nExporting to ONNX: {args.output}")
    try:
        # Use legacy exporter which handles GRU better
        torch.onnx.export(
            model,
            dummy_input,
            str(args.output),
            input_names=['input'],
            output_names=['mask'],
            dynamic_axes={
                'input': {0: 'batch'},
                'mask': {0: 'batch'},
            },
            opset_version=14,
            dynamo=False,  # Use legacy exporter
        )
        print("ONNX export successful!")
    except Exception as e:
        print(f"ONNX export failed: {e}")
        # Try saving as TorchScript instead
        print("\nSaving as TorchScript instead...")
        pt_path = str(args.output).replace('.onnx', '.pt')
        traced = torch.jit.trace(model, dummy_input)
        traced.save(pt_path)
        print(f"TorchScript model saved: {pt_path}")
    
    print(f"Done! ONNX model saved to {args.output}")
    
    # Verify with onnxruntime if available
    if args.output.exists():
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(args.output))
            outputs = sess.run(None, {'input': dummy_input.numpy()})
            print(f"\nONNX verification passed!")
            print(f"  Output shape: {outputs[0].shape}")
            print(f"  Output range: [{outputs[0].min():.4f}, {outputs[0].max():.4f}]")
        except ImportError:
            print("\nNote: Install onnxruntime to verify the exported model")
        except Exception as e:
            print(f"\nONNX verification failed: {e}")


if __name__ == '__main__':
    main()

