#!/usr/bin/env python3
"""
AEC Model extracted from MGK file based on device log analysis.

Architecture from log.txt:
- Input: [1,1,256,8] - 256 freq bins, 8 frames (FP32)
- Hidden: [64,1,1,32] - GRU hidden state (UINT8) = 64 time steps, 32 hidden
- Output: [1,1,256,2] - 256 freq bins, 2 channels (FP32)

Layer sequence from log:
1. layer_80_QuantizeBatchNorm: [1,1,256,8] - input normalization
2. layer_2_QuantizeFeature: [1,1,1,256,32] - expand to 32 channels
3. layer_4_QuantizeFeature: [1,4,1,128,32] - downsample to 128
4. layer_8_QuantizeFeature: [1,1,1,128,32]
5. layer_10_QuantizeFeature: [1,4,1,64,32] - downsample to 64
6. layer_14-28: Multiple [1,1,1,64,32] and [1,4,1,64,32] layers
7. layer_34_QuantizeBatchNorm: [1,1,1,64,32] - pre-GRU norm
8. layer_37_QuantizeGRU: [64,1,1,32] - first GRU
9. layer_43_QuantizeBatchNorm: [1,1,1,64,32] - post-GRU norm
10. layer_46_QuantizeGRU: [1,64,2,32] - second GRU (bidirectional?)
11. Decoder layers back to [1,1,256,2]
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


class AECModel(nn.Module):
    """AEC model matching the MGK architecture more closely."""

    def __init__(self, hidden_size=32, num_features=32):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_features = num_features

        # Input: [B, 1, 256, 8] - 256 freq bins, 8 frames
        # layer_80: Input BatchNorm
        self.input_bn = nn.BatchNorm2d(1)

        # Encoder path: 256 -> 128 -> 64 freq bins
        # Each layer expands to num_features (32) channels

        # layer_2: [1,1,256,8] -> [1,32,256,8] (expand channels)
        self.enc_expand = nn.Conv2d(1, num_features, kernel_size=1)

        # layer_4: [1,32,256,8] -> [1,32,128,8] (downsample freq)
        self.enc_down1 = nn.Conv2d(num_features, num_features, kernel_size=(2,1), stride=(2,1))

        # layer_10: [1,32,128,8] -> [1,32,64,8] (downsample freq)
        self.enc_down2 = nn.Conv2d(num_features, num_features, kernel_size=(2,1), stride=(2,1))

        # Multiple feature layers at 64 freq bins
        self.enc_conv = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(num_features, num_features, kernel_size=1),
            nn.ReLU(),
        )

        # layer_34: Pre-GRU BatchNorm
        self.pre_gru_bn = nn.BatchNorm2d(num_features)

        # layer_37: First GRU - processes 64 time steps with hidden_size=32
        # Input: [B, 64, 32] (64 freq bins, 32 features per bin)
        # Hidden: [1, B, 32]
        self.gru1 = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # layer_43: Post-GRU BatchNorm
        self.post_gru_bn = nn.BatchNorm1d(hidden_size)

        # layer_46: Second GRU - output [1,64,2,32] suggests bidirectional
        self.gru2 = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,  # Output is 2*hidden_size
        )

        # Decoder: 64 -> 128 -> 256 freq bins
        self.dec_up1 = nn.ConvTranspose2d(hidden_size*2, num_features, kernel_size=(2,1), stride=(2,1))
        self.dec_up2 = nn.ConvTranspose2d(num_features, num_features, kernel_size=(2,1), stride=(2,1))

        # Final output: [B, 32, 256, 8] -> [B, 2, 256, 8]
        self.output_conv = nn.Conv2d(num_features, 2, kernel_size=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x, hidden=None):
        """
        Forward pass.

        Args:
            x: Input tensor [B, 1, 256, 8]
            hidden: Tuple of (h1, h2) for two GRU layers

        Returns:
            output: Mask [B, 1, 256, 2]
            hidden: New hidden states
        """
        B = x.shape[0]

        # Input normalization
        x = self.input_bn(x)  # [B, 1, 256, 8]

        # Encoder
        x = self.enc_expand(x)  # [B, 32, 256, 8]
        x = torch.relu(x)
        x = self.enc_down1(x)   # [B, 32, 128, 8]
        x = torch.relu(x)
        x = self.enc_down2(x)   # [B, 32, 64, 8]
        x = self.enc_conv(x)    # [B, 32, 64, 8]

        # Pre-GRU norm
        x = self.pre_gru_bn(x)  # [B, 32, 64, 8]

        # Reshape for GRU: [B*8, 64, 32] (process each frame independently)
        x = x.permute(0, 3, 2, 1)  # [B, 8, 64, 32]
        B, T, F, C = x.shape
        x = x.reshape(B*T, F, C)  # [B*8, 64, 32]

        # First GRU
        if hidden is None:
            h1 = torch.zeros(1, B*T, self.hidden_size, device=x.device)
            h2 = torch.zeros(2, B*T, self.hidden_size, device=x.device)
        else:
            h1, h2 = hidden

        x, h1_new = self.gru1(x, h1)  # [B*8, 64, 32]

        # Post-GRU norm (apply per-feature)
        x = x.reshape(-1, self.hidden_size)
        x = self.post_gru_bn(x)
        x = x.reshape(B*T, F, self.hidden_size)

        # Second GRU (bidirectional)
        x, h2_new = self.gru2(x, h2)  # [B*8, 64, 64] (32*2 for bidirectional)

        # Reshape back: [B, 8, 64, 64] -> [B, 64, 64, 8]
        x = x.reshape(B, T, F, -1)
        x = x.permute(0, 3, 2, 1)  # [B, 64, 64, 8]

        # Decoder
        x = self.dec_up1(x)  # [B, 32, 128, 8]
        x = torch.relu(x)
        x = self.dec_up2(x)  # [B, 32, 256, 8]
        x = torch.relu(x)

        # Output
        x = self.output_conv(x)  # [B, 2, 256, 8]
        x = self.output_act(x)

        # Take last frame and reshape to [B, 1, 256, 2]
        mask = x[:, :, :, -1]  # [B, 2, 256]
        mask = mask.permute(0, 2, 1).unsqueeze(1)  # [B, 1, 256, 2]

        return mask, (h1_new, h2_new)


def load_weights_from_mgk(model: AECModel, mgk_path: Path):
    """Load INT8 weights from MGK file and dequantize."""
    with open(mgk_path, 'rb') as f:
        f.seek(0x79294)
        raw_weights = f.read(153644)

    weights = np.frombuffer(raw_weights, dtype=np.int8)
    print(f"Loaded {len(weights):,} bytes of INT8 weights")

    # Dequantize scale
    scale = 1.0 / 127.0

    # Calculate expected weight sizes based on model architecture
    # hidden_size=32, num_features=32
    H = model.hidden_size  # 32
    F = model.num_features  # 32

    # GRU weights: 3 gates (reset, update, new) * hidden * (input + hidden)
    # GRU1: input=32, hidden=32 -> 3 * 32 * (32 + 32) = 6144
    # GRU2 bidirectional: 2 * 3 * 32 * (32 + 32) = 12288

    gru1_size = 3 * H * (F + H)  # 6144
    gru2_size = 2 * 3 * H * (H + H)  # 12288

    # Conv weights
    enc_expand_size = 1 * F  # 1->32, kernel=1
    enc_down1_size = F * F * 2  # 32->32, kernel=(2,1)
    enc_down2_size = F * F * 2  # 32->32, kernel=(2,1)
    enc_conv_size = F * F * 2  # Two conv layers
    dec_up1_size = H * 2 * F * 2  # 64->32, kernel=(2,1)
    dec_up2_size = F * F * 2  # 32->32, kernel=(2,1)
    output_conv_size = F * 2  # 32->2, kernel=1

    print(f"Expected GRU1 weights: {gru1_size:,}")
    print(f"Expected GRU2 weights: {gru2_size:,}")

    # Try to map weights - this is approximate
    offset = 0
    with torch.no_grad():
        # Skip to GRU weights (they're the largest and most important)
        # Based on weight analysis, GRU weights start around offset 0x4400
        gru_offset = 0x4400

        # Load GRU1 weights
        if gru_offset + gru1_size <= len(weights):
            w = weights[gru_offset:gru_offset+gru1_size].astype(np.float32) * scale
            # GRU weight format: [3*hidden, input+hidden]
            w = w.reshape(3 * H, F + H)
            # Split into weight_ih and weight_hh
            weight_ih = w[:, :F].reshape(3 * H, F)
            weight_hh = w[:, F:].reshape(3 * H, H)
            model.gru1.weight_ih_l0.copy_(torch.from_numpy(weight_ih))
            model.gru1.weight_hh_l0.copy_(torch.from_numpy(weight_hh))
            print(f"  Loaded GRU1 weights: {gru1_size:,}")
            gru_offset += gru1_size

        # Load GRU2 weights (bidirectional)
        if gru_offset + gru2_size <= len(weights):
            w = weights[gru_offset:gru_offset+gru2_size].astype(np.float32) * scale
            # Split for forward and backward
            half = gru2_size // 2
            w_fwd = w[:half].reshape(3 * H, H + H)
            w_bwd = w[half:].reshape(3 * H, H + H)

            model.gru2.weight_ih_l0.copy_(torch.from_numpy(w_fwd[:, :H]))
            model.gru2.weight_hh_l0.copy_(torch.from_numpy(w_fwd[:, H:]))
            model.gru2.weight_ih_l0_reverse.copy_(torch.from_numpy(w_bwd[:, :H]))
            model.gru2.weight_hh_l0_reverse.copy_(torch.from_numpy(w_bwd[:, H:]))
            print(f"  Loaded GRU2 weights: {gru2_size:,}")

    return model


if __name__ == '__main__':
    # Test model
    model = AECModel()
    print("Model created")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(1, 1, 256, 8)
    mask, hidden = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {mask.shape}")
    print(f"Hidden shapes: h1={hidden[0].shape}, h2={hidden[1].shape}")

