#!/usr/bin/env python3
"""
AEC Model v2 - Matches the actual MGK architecture from log.txt analysis.

Architecture based on tensor shapes:
- Input: [1,1,256,8] - 256 freq bins, 8 time frames
- Encoder: Downsamples freq bins 256->128->64
- GRU: Processes 64 freq bins with hidden_size=32
- Decoder: Upsamples freq bins 64->128->256
- Output: [1,1,256,2] - sigmoid mask
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple


class AECModelV2(nn.Module):
    """AEC model matching the MGK architecture."""
    
    def __init__(self, extracted_dir: Optional[Path] = None):
        super().__init__()
        
        # Load scales if available
        self.scales = {}
        if extracted_dir:
            scale_path = extracted_dir / 'layer_scale_mapping.json'
            if scale_path.exists():
                with open(scale_path, 'r') as f:
                    self.scales = json.load(f)
        
        # Encoder - matches layer_2, layer_4, layer_8, layer_10, etc.
        # Input: [B, 8, 256, 1] after permute
        self.enc1 = nn.Conv2d(8, 32, kernel_size=(2, 1), stride=(2, 1))  # 256->128
        self.enc2 = nn.Conv2d(32, 32, kernel_size=(2, 1), stride=(2, 1))  # 128->64
        self.enc3 = nn.Conv2d(32, 32, kernel_size=1)  # 64->64
        self.enc4 = nn.Conv2d(32, 32, kernel_size=1)  # 64->64
        self.enc5 = nn.Conv2d(32, 32, kernel_size=1)  # 64->64
        
        # GRU layers
        self.gru1 = nn.GRU(32, 32, batch_first=True)
        self.gru2 = nn.GRU(32, 32, batch_first=True, bidirectional=True)
        
        # Decoder
        self.dec1 = nn.Conv2d(64, 32, kernel_size=1)  # After bidir GRU concat
        self.dec2 = nn.Conv2d(32, 32, kernel_size=1)
        self.dec3 = nn.ConvTranspose2d(32, 32, kernel_size=(2, 1), stride=(2, 1))  # 64->128
        self.dec4 = nn.ConvTranspose2d(32, 8, kernel_size=(2, 1), stride=(2, 1))  # 128->256
        
        # Output
        self.out_conv = nn.Conv2d(8, 2, kernel_size=1)
        
        self.relu = nn.ReLU()
        
        if extracted_dir:
            self._load_weights(extracted_dir)
    
    def _get_scale(self, layer_name: str) -> float:
        """Get scale for a layer."""
        if layer_name in self.scales:
            s = self.scales[layer_name]
            return s.get('weight_scale', s.get('scale_1', 0.05))
        return 0.05
    
    def _load_weights(self, extracted_dir: Path):
        """Load weights from extracted files."""
        layers_dir = extracted_dir / 'layers'
        
        # Map model layers to extracted files
        layer_files = {
            'enc1': ('layer_2_feature.bin', 'layer_2_QuantizeFeature'),
            'enc2': ('layer_4_feature.bin', 'layer_4_QuantizeFeature'),
            'enc3': ('layer_8_feature.bin', 'layer_8_QuantizeFeature'),
            'enc4': ('layer_10_feature.bin', 'layer_10_QuantizeFeature'),
            'enc5': ('layer_14_feature.bin', 'layer_14_QuantizeFeature'),
        }
        
        for layer_name, (file_name, scale_name) in layer_files.items():
            path = layers_dir / file_name
            if path.exists():
                data = np.fromfile(path, dtype=np.int8)
                scale = self._get_scale(scale_name)
                layer = getattr(self, layer_name)
                expected = layer.weight.numel()
                if len(data) >= expected:
                    w = data[:expected].astype(np.float32) * scale
                    layer.weight.data = torch.from_numpy(w.reshape(layer.weight.shape))
        
        # Load GRU weights
        # Bidirectional GRU (layer_46) - 12,864 bytes
        gru2_path = layers_dir / 'layer_46_gru_bidir.bin'
        if gru2_path.exists():
            data = np.fromfile(gru2_path, dtype=np.int8)
            scale = self._get_scale('layer_46_QuantizeGRU')

            # Structure: 12 x 1024 bytes = 12,288 bytes + padding
            # Each 1024 bytes = 768 bytes weight + 256 bytes (bias or more weight)
            # Forward: 6 x 1024 = 6144 bytes, Backward: 6 x 1024 = 6144 bytes

            # Forward direction
            if len(data) >= 6144:
                # weight_ih_l0: [96, 32]
                w = data[:3072].astype(np.float32) * scale
                self.gru2.weight_ih_l0.data = torch.from_numpy(w.reshape(96, 32))
                # weight_hh_l0: [96, 32]
                w = data[3072:6144].astype(np.float32) * scale
                self.gru2.weight_hh_l0.data = torch.from_numpy(w.reshape(96, 32))

            # Backward direction
            if len(data) >= 12288:
                # weight_ih_l0_reverse: [96, 32]
                w = data[6144:9216].astype(np.float32) * scale
                self.gru2.weight_ih_l0_reverse.data = torch.from_numpy(w.reshape(96, 32))
                # weight_hh_l0_reverse: [96, 32]
                w = data[9216:12288].astype(np.float32) * scale
                self.gru2.weight_hh_l0_reverse.data = torch.from_numpy(w.reshape(96, 32))

        # Unidirectional GRU (layer_37) - 4,096 bytes
        gru1_path = layers_dir / 'layer_37_gru.bin'
        if gru1_path.exists():
            data = np.fromfile(gru1_path, dtype=np.int8)
            scale = self._get_scale('layer_37_QuantizeGRU')

            # Structure: 4 x 1024 bytes
            # weight_ih: [96, 32] = 3072 bytes
            # weight_hh: [32, 32] = 1024 bytes (simplified?)
            if len(data) >= 3072:
                w = data[:3072].astype(np.float32) * scale
                self.gru1.weight_ih_l0.data = torch.from_numpy(w.reshape(96, 32))
            if len(data) >= 4096:
                # Use remaining 1024 bytes for weight_hh (only 32x32 instead of 96x32)
                w = data[3072:4096].astype(np.float32) * scale
                # Pad to 96x32 by repeating
                w_padded = np.zeros((96, 32), dtype=np.float32)
                w_padded[:32, :] = w.reshape(32, 32)
                self.gru1.weight_hh_l0.data = torch.from_numpy(w_padded)
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple] = None):
        """
        Forward pass.
        
        Args:
            x: Input [B, 1, 256, 8] - freq bins x time frames
            hidden: Optional GRU hidden states
        
        Returns:
            mask: Output [B, 1, 256, 2]
            hidden: Updated hidden states
        """
        B = x.size(0)
        
        # Reshape: [B, 1, 256, 8] -> [B, 8, 256, 1]
        x = x.permute(0, 3, 2, 1)
        
        # Encoder
        x = self.relu(self.enc1(x))  # [B, 32, 128, 1]
        x = self.relu(self.enc2(x))  # [B, 32, 64, 1]
        x = self.relu(self.enc3(x))  # [B, 32, 64, 1]
        x = self.relu(self.enc4(x))  # [B, 32, 64, 1]
        x = self.relu(self.enc5(x))  # [B, 32, 64, 1]
        
        # Reshape for GRU: [B, 32, 64, 1] -> [B, 64, 32]
        x = x.squeeze(-1).permute(0, 2, 1)
        
        # GRU layers
        h1 = hidden[0] if hidden else None
        h2 = hidden[1] if hidden else None
        
        x, h1_new = self.gru1(x, h1)  # [B, 64, 32]
        x, h2_new = self.gru2(x, h2)  # [B, 64, 64] (bidirectional)
        
        # Reshape back: [B, 64, 64] -> [B, 64, 64, 1]
        x = x.permute(0, 2, 1).unsqueeze(-1)
        
        # Decoder
        x = self.relu(self.dec1(x))  # [B, 32, 64, 1]
        x = self.relu(self.dec2(x))  # [B, 32, 64, 1]
        x = self.relu(self.dec3(x))  # [B, 32, 128, 1]
        x = self.relu(self.dec4(x))  # [B, 8, 256, 1]
        
        # Output
        x = self.out_conv(x)  # [B, 2, 256, 1]
        x = torch.sigmoid(x)
        
        # Reshape: [B, 2, 256, 1] -> [B, 1, 256, 2]
        x = x.squeeze(-1).permute(0, 2, 1).unsqueeze(1)
        
        return x, (h1_new, h2_new)

