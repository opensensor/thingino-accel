#!/usr/bin/env python3
"""
Run AEC inference using extracted weights from MGK file.

Based on log.txt analysis, the model architecture is:
- Input: [1,1,256,8] - 256 freq bins, 8 frames (FP32)
- Output: [1,1,256,2] - 256 freq bins, 2 channels (FP32)
- Hidden: [64,1,1,32] - GRU hidden state

The model uses:
- 512-point FFT (256 bins)
- 8-frame context window
- 2-channel output (likely magnitude mask for real/imag or stereo)
"""

import argparse
import json
import wave
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


def load_wav(path: Path):
    """Load WAV file as float32 array."""
    with wave.open(str(path), 'rb') as wf:
        sr = wf.getframerate()
        samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        return samples.astype(np.float32) / 32768.0, sr


def save_wav(path: Path, samples: np.ndarray, sr: int):
    """Save float32 array as WAV file."""
    samples_int = (samples * 32768.0).clip(-32768, 32767).astype(np.int16)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples_int.tobytes())


class AECModelV2(nn.Module):
    """
    AEC model matching MGK architecture from log.txt.

    Architecture from log.txt analysis:
    - Input: [1,1,256,8] FP32 - 256 freq bins, 8 frames
    - layer_80: BatchNorm, quantize to UINT8
    - layer_2: Expand 8 frames -> 32 channels [1,1,1,256,32]
    - layer_4: Downsample 256->128 freq [1,4,1,128,32]
    - layer_10: Downsample 128->64 freq [1,4,1,64,32]
    - layer_34: BatchNorm pre-GRU [1,1,1,64,32]
    - layer_37: GRU [64,1,1,32] - 64 time steps, 32 hidden
    - layer_43: BatchNorm post-GRU
    - layer_46: GRU bidirectional [1,64,2,32]
    - Output: [1,1,256,2] FP32 - sigmoid mask
    """

    def __init__(self, n_freq=256, n_frames=8, n_channels=32, hidden_size=32):
        super().__init__()
        self.n_freq = n_freq
        self.n_frames = n_frames
        self.n_channels = n_channels
        self.hidden_size = hidden_size

        # layer_80: Input BatchNorm
        self.input_bn = nn.BatchNorm1d(n_freq)

        # layer_2: Expand frames to channels (8 -> 32)
        self.expand = nn.Conv1d(n_frames, n_channels, kernel_size=1)

        # layer_4: Downsample 256 -> 128 freq bins
        self.down1 = nn.Conv1d(n_channels, n_channels, kernel_size=2, stride=2)

        # layer_8: Feature processing at 128 bins
        self.conv1 = nn.Conv1d(n_channels, n_channels, kernel_size=1)

        # layer_10: Downsample 128 -> 64 freq bins
        self.down2 = nn.Conv1d(n_channels, n_channels, kernel_size=2, stride=2)

        # Multiple feature layers at 64 bins (layers 14-32)
        self.feature_layers = nn.Sequential(
            nn.Conv1d(n_channels, n_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(n_channels, n_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(n_channels, n_channels, kernel_size=1),
            nn.ReLU(),
        )

        # layer_34: Pre-GRU BatchNorm
        self.pre_gru_bn = nn.BatchNorm1d(n_channels)

        # layer_37: First GRU - input is [64, 32] (64 freq bins, 32 channels)
        # Processes as sequence of 64 time steps with 32 features each
        self.gru1 = nn.GRU(n_channels, hidden_size, batch_first=True)

        # layer_43: Post-GRU BatchNorm
        self.post_gru_bn = nn.BatchNorm1d(hidden_size)

        # layer_46: Second GRU (bidirectional)
        self.gru2 = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)

        # Decoder: upsample back to 256 freq bins
        self.up1 = nn.ConvTranspose1d(hidden_size * 2, n_channels, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose1d(n_channels, n_channels, kernel_size=2, stride=2)

        # Output: 32 channels -> 2 channels
        self.output_conv = nn.Conv1d(n_channels, 2, kernel_size=1)

    def forward(self, x, hidden=None):
        """
        Args:
            x: [B, 1, n_freq, n_frames] or [B, n_freq, n_frames]
        Returns:
            mask: [B, n_freq, 2]
            hidden: tuple of hidden states
        """
        if x.dim() == 4:
            x = x.squeeze(1)  # [B, 256, 8]

        B, F, T = x.shape

        # Input BatchNorm (layer_80)
        x = self.input_bn(x)  # [B, 256, 8]

        # Transpose to [B, T, F] then expand channels
        x = x.permute(0, 2, 1)  # [B, 8, 256]
        x = self.expand(x)  # [B, 32, 256]
        x = torch.relu(x)

        # Downsample (layer_4, layer_10)
        x = self.down1(x)  # [B, 32, 128]
        x = torch.relu(x)
        x = self.conv1(x)  # [B, 32, 128]
        x = torch.relu(x)
        x = self.down2(x)  # [B, 32, 64]
        x = torch.relu(x)

        # Feature layers
        x = self.feature_layers(x)  # [B, 32, 64]

        # Pre-GRU BatchNorm (layer_34)
        x = self.pre_gru_bn(x)  # [B, 32, 64]

        # Reshape for GRU: [B, 64, 32] (64 time steps, 32 features)
        x = x.permute(0, 2, 1)  # [B, 64, 32]

        # GRU layers
        if hidden is None:
            h1 = torch.zeros(1, B, self.hidden_size, device=x.device)
            h2 = torch.zeros(2, B, self.hidden_size, device=x.device)
        else:
            h1, h2 = hidden

        x, h1_new = self.gru1(x, h1)  # [B, 64, 32]

        # Post-GRU BatchNorm (layer_43)
        x = x.permute(0, 2, 1)  # [B, 32, 64]
        x = self.post_gru_bn(x)
        x = x.permute(0, 2, 1)  # [B, 64, 32]

        x, h2_new = self.gru2(x, h2)  # [B, 64, 64] (bidirectional)

        # Reshape for decoder: [B, 64, 64] -> [B, 64, 64]
        x = x.permute(0, 2, 1)  # [B, 64, 64]

        # Upsample
        x = self.up1(x)  # [B, 32, 128]
        x = torch.relu(x)
        x = self.up2(x)  # [B, 32, 256]
        x = torch.relu(x)

        # Output
        x = self.output_conv(x)  # [B, 2, 256]
        mask = torch.sigmoid(x)

        # Reshape to [B, 256, 2]
        mask = mask.permute(0, 2, 1)  # [B, 256, 2]

        return mask, (h1_new, h2_new)


def load_extracted_weights(extracted_dir: Path) -> tuple:
    """Load weights and metadata from extracted directory."""
    metadata_path = extracted_dir / "metadata.json"
    weights_path = extracted_dir / "weights.bin"

    if not metadata_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"Extracted files not found in {extracted_dir}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    with open(weights_path, 'rb') as f:
        weights = np.frombuffer(f.read(), dtype=np.int8)

    return weights, metadata


def get_layer_scales(metadata: dict) -> list:
    """Extract layer-wise quantization scales from metadata."""
    # Group scales by proximity (scales for same layer are adjacent)
    raw_scales = metadata.get('quantization_scales', [])

    # Filter to reasonable scale values and group consecutive ones
    scales = []
    for entry in raw_scales:
        val = entry['value']
        if 0.005 < val < 0.06:  # Typical range for layer scales
            scales.append(val)

    # Return unique scales in order (dedup consecutive duplicates)
    unique_scales = []
    prev = None
    for s in scales:
        if prev is None or abs(s - prev) > 0.0001:
            unique_scales.append(s)
            prev = s

    return unique_scales if unique_scales else [1.0 / 127.0]  # Fallback


def load_weights_from_extracted(model, extracted_dir: Path):
    """Load INT8 weights from extracted directory."""
    weights, metadata = load_extracted_weights(extracted_dir)
    layer_scales = get_layer_scales(metadata)

    H = model.hidden_size  # 32
    C = model.n_channels  # 32

    print(f"Loading weights from {extracted_dir}")
    print(f"  Weight data: {len(weights):,} bytes")
    print(f"  Found {len(layer_scales)} quantization scales")

    offset = 0
    scale_idx = 0

    def get_scale():
        nonlocal scale_idx
        s = layer_scales[min(scale_idx, len(layer_scales) - 1)]
        scale_idx += 1
        return s

    with torch.no_grad():
        # expand: [32, 8, 1]
        size = C * 8
        scale = get_scale()
        if offset + size <= len(weights):
            w = weights[offset:offset+size].astype(np.float32) * scale
            model.expand.weight.copy_(torch.from_numpy(w.reshape(C, 8, 1)))
            offset += size
            print(f"  expand: {size} weights, scale={scale:.6f}")

        # down1: [32, 32, 2]
        size = C * C * 2
        scale = get_scale()
        if offset + size <= len(weights):
            w = weights[offset:offset+size].astype(np.float32) * scale
            model.down1.weight.copy_(torch.from_numpy(w.reshape(C, C, 2)))
            offset += size
            print(f"  down1: {size} weights, scale={scale:.6f}")

        # conv1: [32, 32, 1]
        size = C * C
        scale = get_scale()
        if offset + size <= len(weights):
            w = weights[offset:offset+size].astype(np.float32) * scale
            model.conv1.weight.copy_(torch.from_numpy(w.reshape(C, C, 1)))
            offset += size
            print(f"  conv1: {size} weights, scale={scale:.6f}")

        # down2: [32, 32, 2]
        size = C * C * 2
        scale = get_scale()
        if offset + size <= len(weights):
            w = weights[offset:offset+size].astype(np.float32) * scale
            model.down2.weight.copy_(torch.from_numpy(w.reshape(C, C, 2)))
            offset += size
            print(f"  down2: {size} weights, scale={scale:.6f}")

        # feature_layers: 3 conv layers
        for i, layer in enumerate([model.feature_layers[0], model.feature_layers[2], model.feature_layers[4]]):
            size = C * C
            scale = get_scale()
            if offset + size <= len(weights):
                w = weights[offset:offset+size].astype(np.float32) * scale
                layer.weight.copy_(torch.from_numpy(w.reshape(C, C, 1)))
                offset += size
                print(f"  feature_layer[{i}]: {size} weights, scale={scale:.6f}")

        # GRU1: weight_ih [3*H, C] and weight_hh [3*H, H]
        ih_size = 3 * H * C
        hh_size = 3 * H * H
        scale = get_scale()
        if offset + ih_size + hh_size <= len(weights):
            w_ih = weights[offset:offset+ih_size].astype(np.float32) * scale
            model.gru1.weight_ih_l0.copy_(torch.from_numpy(w_ih.reshape(3*H, C)))
            offset += ih_size

            w_hh = weights[offset:offset+hh_size].astype(np.float32) * scale
            model.gru1.weight_hh_l0.copy_(torch.from_numpy(w_hh.reshape(3*H, H)))
            offset += hh_size
            print(f"  gru1: {ih_size + hh_size} weights, scale={scale:.6f}")

        # GRU2 (bidirectional): 2 * (weight_ih + weight_hh)
        for suffix in ['', '_reverse']:
            ih_size = 3 * H * H
            hh_size = 3 * H * H
            scale = get_scale()
            if offset + ih_size + hh_size <= len(weights):
                w_ih = weights[offset:offset+ih_size].astype(np.float32) * scale
                getattr(model.gru2, f'weight_ih_l0{suffix}').copy_(
                    torch.from_numpy(w_ih.reshape(3*H, H)))
                offset += ih_size

                w_hh = weights[offset:offset+hh_size].astype(np.float32) * scale
                getattr(model.gru2, f'weight_hh_l0{suffix}').copy_(
                    torch.from_numpy(w_hh.reshape(3*H, H)))
                offset += hh_size
                print(f"  gru2{suffix}: {ih_size + hh_size} weights, scale={scale:.6f}")

        # up1: [32, 64, 2] (ConvTranspose: [in, out, kernel])
        size = C * (H * 2) * 2
        scale = get_scale()
        if offset + size <= len(weights):
            w = weights[offset:offset+size].astype(np.float32) * scale
            model.up1.weight.copy_(torch.from_numpy(w.reshape(H * 2, C, 2)))
            offset += size
            print(f"  up1: {size} weights, scale={scale:.6f}")

        # up2: [32, 32, 2]
        size = C * C * 2
        scale = get_scale()
        if offset + size <= len(weights):
            w = weights[offset:offset+size].astype(np.float32) * scale
            model.up2.weight.copy_(torch.from_numpy(w.reshape(C, C, 2)))
            offset += size
            print(f"  up2: {size} weights, scale={scale:.6f}")

        # output_conv: [2, 32, 1]
        size = 2 * C
        scale = get_scale()
        if offset + size <= len(weights):
            w = weights[offset:offset+size].astype(np.float32) * scale
            model.output_conv.weight.copy_(torch.from_numpy(w.reshape(2, C, 1)))
            offset += size
            print(f"  output_conv: {size} weights, scale={scale:.6f}")

    print(f"  Total loaded: {offset:,} / {len(weights):,} bytes")


def process_audio(model, mic: np.ndarray, lpb: np.ndarray, sr: int):
    """
    Process audio using the AEC model.

    Based on log.txt, the model expects:
    - Input: [1, 1, 256, 8] - 256 freq bins, 8 frames
    - Output: [1, 1, 256, 2] - mask for 2 channels

    We use 512-point FFT (256 bins) with hop=128 (8ms at 16kHz).
    """
    n_fft = 512
    hop = 128
    n_frames = 8  # Model expects 8 frames of context
    n_freq = 256

    # Pad audio to ensure we have enough samples
    pad_len = n_fft + (n_frames - 1) * hop
    mic_padded = np.pad(mic, (0, max(0, pad_len - len(mic))))
    _ = np.pad(lpb, (0, max(0, pad_len - len(lpb))))  # lpb_padded for future use

    total_frames = (len(mic_padded) - n_fft) // hop + 1
    print(f"Processing {total_frames} frames with 8-frame context windows...")

    model.eval()
    hidden = None

    # Output buffer - use same length as input
    output = np.zeros(len(mic), dtype=np.float32)
    window = np.hanning(n_fft)

    # Track mask statistics
    all_masks = []

    with torch.no_grad():
        for frame_idx in range(0, total_frames - n_frames + 1):
            # Build input: [1, 256, 8] - 8 frames of 256 freq bins
            input_tensor = np.zeros((1, n_freq, n_frames), dtype=np.float32)

            for j in range(n_frames):
                start = (frame_idx + j) * hop
                frame = mic_padded[start:start+n_fft] * window
                spectrum = np.fft.rfft(frame)
                # Use log magnitude as input (normalized)
                mag = np.abs(spectrum[:n_freq])
                input_tensor[0, :, j] = np.log1p(mag)

            # Run model
            x = torch.from_numpy(input_tensor)
            mask, hidden = model(x, hidden)
            mask = mask.numpy()[0]  # [256, 2]

            # Use channel 0 as the suppression mask (lower values = more suppression)
            # Channel 1 might be for a different purpose (e.g., noise estimation)
            suppression_mask = mask[:, 0]
            all_masks.append(np.mean(suppression_mask))

            # Apply mask to the last frame (frame_idx + n_frames - 1)
            out_frame_idx = frame_idx + n_frames - 1
            start = out_frame_idx * hop

            if start + n_fft <= len(mic):
                frame = mic[start:start+n_fft] * window
                spectrum = np.fft.rfft(frame)

                # Extend mask to full spectrum
                full_mask = np.ones(n_fft // 2 + 1, dtype=np.float32)
                full_mask[:n_freq] = suppression_mask

                # Apply mask to magnitude, keep phase
                mag = np.abs(spectrum)
                phase = np.angle(spectrum)
                mag_out = mag * full_mask
                spectrum_out = mag_out * np.exp(1j * phase)

                frame_out = np.fft.irfft(spectrum_out, n_fft)

                # Overlap-add with proper windowing
                output[start:start+n_fft] += frame_out * window

    # Normalize by window overlap
    # For Hann window with 75% overlap (hop=128, n_fft=512), normalization is ~1.5
    norm_factor = np.zeros(len(output))
    for i in range(0, total_frames - n_frames + 1):
        out_frame_idx = i + n_frames - 1
        start = out_frame_idx * hop
        if start + n_fft <= len(output):
            norm_factor[start:start+n_fft] += window ** 2

    # Apply normalization where we have overlap
    valid = norm_factor > 1e-8
    output[valid] /= norm_factor[valid]

    # Copy unprocessed portions from input
    output[~valid] = mic[~valid] if len(mic) == len(output) else 0

    # Report statistics
    avg_mask = np.mean(all_masks) if all_masks else 0.5
    print(f"  Average mask: {avg_mask:.3f}")
    print(f"  Processing complete")

    return output


def main():
    parser = argparse.ArgumentParser(description='AEC inference with MGK weights')
    parser.add_argument('--extracted', type=Path, required=True,
                        help='Directory with extracted weights (from extract_mgk_weights.py)')
    parser.add_argument('--mic', type=Path, required=True, help='Microphone input WAV')
    parser.add_argument('--lpb', type=Path, required=True, help='Loopback WAV')
    parser.add_argument('--output', '-o', type=Path, required=True, help='Output WAV')
    args = parser.parse_args()

    print("=== AEC Inference with Extracted MGK Weights ===\n")

    # Load audio
    mic, sr = load_wav(args.mic)
    lpb, _ = load_wav(args.lpb)
    print(f"Loaded mic: {len(mic)} samples, {len(mic)/sr:.2f}s @ {sr}Hz")
    print(f"Loaded lpb: {len(lpb)} samples")

    # Create and load model
    model = AECModelV2(n_freq=256, n_frames=8, hidden_size=32)
    load_weights_from_extracted(model, args.extracted)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Process
    output = process_audio(model, mic, lpb, sr)

    # Save
    save_wav(args.output, output, sr)
    print(f"\nSaved output: {args.output} ({len(output)/sr:.2f}s)")

    # Compare RMS
    mic_rms = np.sqrt(np.mean(mic**2))
    out_rms = np.sqrt(np.mean(output**2))
    print(f"\nRMS comparison:")
    print(f"  Input:  {mic_rms:.4f}")
    print(f"  Output: {out_rms:.4f}")
    print(f"  Change: {20*np.log10(out_rms/mic_rms):.1f} dB")


if __name__ == '__main__':
    main()

