#!/usr/bin/env python3
"""
MGK to PyTorch Model Converter

Converts extracted MGK weights to a PyTorch model for inference.
Uses the extracted weights and quantization scales from the MGK file.
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class QuantizedConv1x1(nn.Module):
    """Quantized 1x1 convolution layer."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 weight_scale: float = 0.05, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_scale = weight_scale
        
        # INT8 weights will be loaded and dequantized
        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.conv2d(x, self.weight, self.bias)
    
    def load_int8_weights(self, weight_data: np.ndarray, bias_data: Optional[np.ndarray] = None):
        """Load INT8 weights and dequantize."""
        # Reshape and dequantize
        w = weight_data.reshape(self.out_channels, self.in_channels, 1, 1)
        self.weight.data = torch.from_numpy(w.astype(np.float32) * self.weight_scale)
        
        if bias_data is not None and self.bias is not None:
            self.bias.data = torch.from_numpy(bias_data.astype(np.float32) * self.weight_scale)


class QuantizedGRU(nn.Module):
    """Quantized GRU layer."""
    
    def __init__(self, input_size: int, hidden_size: int, 
                 bidirectional: bool = False, weight_scale: float = 0.05):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.weight_scale = weight_scale
        
        # GRU has 3 gates: reset, update, new
        gate_size = 3 * hidden_size
        
        # Forward direction
        self.weight_ih = nn.Parameter(torch.zeros(gate_size, input_size))
        self.weight_hh = nn.Parameter(torch.zeros(gate_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.zeros(gate_size))
        self.bias_hh = nn.Parameter(torch.zeros(gate_size))
        
        if bidirectional:
            self.weight_ih_reverse = nn.Parameter(torch.zeros(gate_size, input_size))
            self.weight_hh_reverse = nn.Parameter(torch.zeros(gate_size, hidden_size))
            self.bias_ih_reverse = nn.Parameter(torch.zeros(gate_size))
            self.bias_hh_reverse = nn.Parameter(torch.zeros(gate_size))
    
    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        if h is None:
            h = torch.zeros(self.num_directions, batch_size, self.hidden_size, device=x.device)
        
        # Simple GRU implementation
        outputs = []
        h_fwd = h[0] if self.bidirectional else h.squeeze(0)
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Gates
            gates = x_t @ self.weight_ih.t() + self.bias_ih + h_fwd @ self.weight_hh.t() + self.bias_hh
            r, z, n = gates.chunk(3, dim=-1)
            r = torch.sigmoid(r)
            z = torch.sigmoid(z)
            n = torch.tanh(n)
            
            h_fwd = (1 - z) * n + z * h_fwd
            outputs.append(h_fwd)
        
        output = torch.stack(outputs, dim=1)
        
        if self.bidirectional:
            # Backward pass
            outputs_bwd = []
            h_bwd = h[1]
            
            for t in range(seq_len - 1, -1, -1):
                x_t = x[:, t, :]
                gates = x_t @ self.weight_ih_reverse.t() + self.bias_ih_reverse + \
                        h_bwd @ self.weight_hh_reverse.t() + self.bias_hh_reverse
                r, z, n = gates.chunk(3, dim=-1)
                r = torch.sigmoid(r)
                z = torch.sigmoid(z)
                n = torch.tanh(n)
                h_bwd = (1 - z) * n + z * h_bwd
                outputs_bwd.append(h_bwd)
            
            outputs_bwd = outputs_bwd[::-1]
            output_bwd = torch.stack(outputs_bwd, dim=1)
            output = torch.cat([output, output_bwd], dim=-1)
            h_out = torch.stack([h_fwd, h_bwd], dim=0)
        else:
            h_out = h_fwd.unsqueeze(0)
        
        return output, h_out
    
    def load_int8_weights(self, weight_data: np.ndarray):
        """Load INT8 weights from extracted data."""
        gate_size = 3 * self.hidden_size
        
        if self.bidirectional:
            # Forward direction
            offset = 0
            self.weight_ih.data = torch.from_numpy(
                weight_data[offset:offset + gate_size * self.input_size]
                .reshape(gate_size, self.input_size).astype(np.float32) * self.weight_scale
            )
            offset += gate_size * self.input_size
            
            self.weight_hh.data = torch.from_numpy(
                weight_data[offset:offset + gate_size * self.hidden_size]
                .reshape(gate_size, self.hidden_size).astype(np.float32) * self.weight_scale
            )
            offset += gate_size * self.hidden_size
            
            bias_size = gate_size * 2  # ih and hh biases
            bias_data = weight_data[offset:offset + bias_size].astype(np.float32) * self.weight_scale
            self.bias_ih.data = torch.from_numpy(bias_data[:gate_size])
            self.bias_hh.data = torch.from_numpy(bias_data[gate_size:])
            offset += bias_size
            
            # Backward direction
            self.weight_ih_reverse.data = torch.from_numpy(
                weight_data[offset:offset + gate_size * self.input_size]
                .reshape(gate_size, self.input_size).astype(np.float32) * self.weight_scale
            )
            offset += gate_size * self.input_size
            
            self.weight_hh_reverse.data = torch.from_numpy(
                weight_data[offset:offset + gate_size * self.hidden_size]
                .reshape(gate_size, self.hidden_size).astype(np.float32) * self.weight_scale
            )
            offset += gate_size * self.hidden_size
            
            bias_data = weight_data[offset:offset + bias_size].astype(np.float32) * self.weight_scale
            self.bias_ih_reverse.data = torch.from_numpy(bias_data[:gate_size])
            self.bias_hh_reverse.data = torch.from_numpy(bias_data[gate_size:])
        else:
            # Unidirectional - handle different weight formats
            # The 4096 byte format appears to be 4 x 32x32 matrices
            # This could be a simplified GRU or different gate structure

            total_size = len(weight_data)

            if total_size == 4096:
                # Special case: 4 x 32x32 matrices
                # Interpret as: weight_ih (32x32), weight_hh (32x32), and 2 more
                matrix_size = 32 * 32

                # Load as simplified structure
                self.weight_ih.data = torch.from_numpy(
                    weight_data[0:matrix_size * 3]
                    .reshape(gate_size, self.input_size).astype(np.float32) * self.weight_scale
                )

                self.weight_hh.data = torch.from_numpy(
                    weight_data[matrix_size:matrix_size + gate_size * self.hidden_size]
                    .reshape(gate_size, self.hidden_size).astype(np.float32) * self.weight_scale
                )

                # Remaining data for biases
                remaining = weight_data[matrix_size * 2:]
                if len(remaining) >= gate_size * 2:
                    self.bias_ih.data = torch.from_numpy(
                        remaining[:gate_size].astype(np.float32) * self.weight_scale
                    )
                    self.bias_hh.data = torch.from_numpy(
                        remaining[gate_size:gate_size*2].astype(np.float32) * self.weight_scale
                    )
            else:
                # Standard format
                offset = 0
                self.weight_ih.data = torch.from_numpy(
                    weight_data[offset:offset + gate_size * self.input_size]
                    .reshape(gate_size, self.input_size).astype(np.float32) * self.weight_scale
                )
                offset += gate_size * self.input_size

                self.weight_hh.data = torch.from_numpy(
                    weight_data[offset:offset + gate_size * self.hidden_size]
                    .reshape(gate_size, self.hidden_size).astype(np.float32) * self.weight_scale
                )
                offset += gate_size * self.hidden_size

                bias_size = gate_size * 2
                if offset + bias_size <= len(weight_data):
                    bias_data = weight_data[offset:offset + bias_size].astype(np.float32) * self.weight_scale
                    self.bias_ih.data = torch.from_numpy(bias_data[:gate_size])
                    self.bias_hh.data = torch.from_numpy(bias_data[gate_size:])


class AECModel(nn.Module):
    """AEC (Acoustic Echo Cancellation) model extracted from MGK."""

    def __init__(self, extracted_dir: Path):
        super().__init__()
        self.extracted_dir = extracted_dir

        # Load metadata
        model_json = extracted_dir / 'model.json'
        if model_json.exists():
            with open(model_json, 'r') as f:
                self.metadata = json.load(f)
        else:
            with open(extracted_dir / 'model_complete.json', 'r') as f:
                self.metadata = json.load(f)

        # Load layer-scale mapping if available
        scale_json = extracted_dir / 'layer_scale_mapping.json'
        if scale_json.exists():
            with open(scale_json, 'r') as f:
                self.scale_mapping = json.load(f)
        else:
            self.scale_mapping = {}

        # Model architecture based on log.txt analysis:
        # Input: [1, 1, 256, 8] - 256 freq bins, 8 time frames
        # Output: [1, 1, 256, 2] - sigmoid mask

        # Build layers
        self._build_layers()
        self._load_weights()

    def _build_layers(self):
        """Build the model layers."""
        # Encoder path
        self.conv_in = QuantizedConv1x1(8, 32)  # layer_2

        # Downsampling convs
        self.conv_down1 = QuantizedConv1x1(32, 32)  # layer_4
        self.conv_down2 = QuantizedConv1x1(32, 32)  # layer_10
        self.conv_down3 = QuantizedConv1x1(32, 32)  # layer_16
        self.conv_down4 = QuantizedConv1x1(32, 32)  # layer_22
        self.conv_down5 = QuantizedConv1x1(32, 32)  # layer_28

        # Feature convs
        self.conv_feat1 = QuantizedConv1x1(32, 32)  # layer_8
        self.conv_feat2 = QuantizedConv1x1(32, 32)  # layer_14
        self.conv_feat3 = QuantizedConv1x1(32, 32)  # layer_20
        self.conv_feat4 = QuantizedConv1x1(32, 32)  # layer_26
        self.conv_feat5 = QuantizedConv1x1(32, 32)  # layer_32

        # GRU layers
        self.gru1 = QuantizedGRU(32, 32, bidirectional=False)  # layer_37
        self.gru2 = QuantizedGRU(32, 32, bidirectional=True)   # layer_46

        # Decoder convs
        self.conv_dec1 = QuantizedConv1x1(32, 32)  # layer_35
        self.conv_dec2 = QuantizedConv1x1(32, 32)  # layer_41
        self.conv_dec3 = QuantizedConv1x1(32, 32)  # layer_44

        # Upsampling convs
        self.conv_up1 = QuantizedConv1x1(64, 32)  # layer_58 (after bidir GRU)
        self.conv_up2 = QuantizedConv1x1(32, 32)  # layer_63
        self.conv_up3 = QuantizedConv1x1(32, 32)  # layer_68
        self.conv_up4 = QuantizedConv1x1(32, 32)  # layer_73
        self.conv_up5 = QuantizedConv1x1(32, 32)  # layer_78

        # Output
        self.conv_out = QuantizedConv1x1(32, 2)  # Final output

        # Activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def _get_scale(self, layer_name: str) -> float:
        """Get the weight scale for a layer."""
        if layer_name in self.scale_mapping:
            scales = self.scale_mapping[layer_name]
            # Use weight_scale if available, otherwise scale_1
            return scales.get('weight_scale', scales.get('scale_1', 0.05))
        return 0.05  # Default scale

    def _load_weights(self):
        """Load weights from extracted files."""
        layers_dir = self.extracted_dir / 'layers'

        # Load GRU weights with scales
        gru_bidir_path = layers_dir / 'layer_46_gru_bidir.bin'
        if gru_bidir_path.exists():
            gru_data = np.fromfile(gru_bidir_path, dtype=np.int8)
            scale = self._get_scale('layer_46_QuantizeGRU')
            self.gru2.weight_scale = scale
            self.gru2.load_int8_weights(gru_data)

        gru_uni_path = layers_dir / 'layer_37_gru.bin'
        if gru_uni_path.exists():
            gru_data = np.fromfile(gru_uni_path, dtype=np.int8)
            scale = self._get_scale('layer_37_QuantizeGRU')
            self.gru1.weight_scale = scale
            self.gru1.load_int8_weights(gru_data)

        # Load conv weights
        conv_layer_mapping = {
            'conv_in': 'layer_2_feature',
            'conv_down1': 'layer_4_feature',
            'conv_feat1': 'layer_8_feature',
            'conv_down2': 'layer_10_feature',
            'conv_feat2': 'layer_14_feature',
            'conv_down3': 'layer_16_feature',
            'conv_feat3': 'layer_20_feature',
            'conv_down4': 'layer_22_feature',
            'conv_feat4': 'layer_26_feature',
            'conv_down5': 'layer_28_feature',
            'conv_feat5': 'layer_32_feature',
            'conv_dec1': 'layer_35_feature',
            'conv_dec2': 'layer_41_feature',
            'conv_dec3': 'layer_44_feature',
            'conv_up1': 'layer_58_feature',
            'conv_up2': 'layer_63_feature',
            'conv_up3': 'layer_68_feature',
            'conv_up4': 'layer_73_feature',
            'conv_up5': 'layer_78_feature',
        }

        for conv_name, layer_name in conv_layer_mapping.items():
            weight_path = layers_dir / f'{layer_name}.bin'
            if weight_path.exists() and hasattr(self, conv_name):
                weight_data = np.fromfile(weight_path, dtype=np.int8)
                scale_name = layer_name.replace('_feature', '_QuantizeFeature')
                scale = self._get_scale(scale_name)
                conv = getattr(self, conv_name)
                conv.weight_scale = scale
                # Only load if size matches
                expected_size = conv.in_channels * conv.out_channels
                if len(weight_data) >= expected_size:
                    conv.load_int8_weights(weight_data[:expected_size])

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, 1, freq_bins, time_frames]
            hidden: Hidden state for GRU

        Returns:
            mask: Output mask [batch, 1, freq_bins, 2]
            hidden: Updated hidden state
        """
        batch_size = x.size(0)

        # Reshape for processing
        # [B, 1, 256, 8] -> [B, 8, 256, 1]
        x = x.permute(0, 3, 2, 1)

        # Encoder
        x = self.relu(self.conv_in(x))

        # Downsampling path
        x = self.relu(self.conv_down1(x))
        x = self.relu(self.conv_feat1(x))

        x = self.relu(self.conv_down2(x))
        x = self.relu(self.conv_feat2(x))

        x = self.relu(self.conv_down3(x))
        x = self.relu(self.conv_feat3(x))

        x = self.relu(self.conv_down4(x))
        x = self.relu(self.conv_feat4(x))

        x = self.relu(self.conv_down5(x))
        x = self.relu(self.conv_feat5(x))

        # Reshape for GRU: [B, C, H, W] -> [B, H*W, C]
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)

        # GRU layers
        x, h1 = self.gru1(x, hidden)
        x, h2 = self.gru2(x)

        # Reshape back: [B, H*W, C] -> [B, C, H, W]
        x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2)

        # Decoder
        x = self.relu(self.conv_up1(x))
        x = self.relu(self.conv_dec1(x))

        x = self.relu(self.conv_up2(x))
        x = self.relu(self.conv_dec2(x))

        x = self.relu(self.conv_up3(x))
        x = self.relu(self.conv_dec3(x))

        x = self.relu(self.conv_up4(x))
        x = self.relu(self.conv_up5(x))

        # Output
        x = self.sigmoid(self.conv_out(x))

        # Reshape to output format
        # [B, 2, 256, 1] -> [B, 1, 256, 2]
        x = x.permute(0, 3, 2, 1)

        return x, h1


def main():
    parser = argparse.ArgumentParser(description='Convert MGK to PyTorch model')
    parser.add_argument('extracted_dir', type=Path, help='Directory with extracted weights')
    parser.add_argument('--test', action='store_true', help='Run a test inference')
    args = parser.parse_args()

    if not args.extracted_dir.exists():
        print(f"Error: {args.extracted_dir} not found")
        return 1

    # Create model
    model = AECModel(args.extracted_dir)
    print(f"Created AEC model with {sum(p.numel() for p in model.parameters()):,} parameters")

    if args.test:
        # Test inference
        model.eval()
        with torch.no_grad():
            x = torch.randn(1, 1, 256, 8)
            mask, hidden = model(x)
            print(f"Input shape: {x.shape}")
            print(f"Output shape: {mask.shape}")
            print(f"Mask range: [{mask.min():.4f}, {mask.max():.4f}]")

    return 0


if __name__ == '__main__':
    exit(main())

