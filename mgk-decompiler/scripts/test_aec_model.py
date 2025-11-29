#!/usr/bin/env python3
"""
Test the converted AEC model on audio files.

Usage:
    python test_aec_model.py --model aec_model.pt --mic mic.wav --lpb lpb.wav --output out.wav
"""

import argparse
import wave
import numpy as np
import torch
from pathlib import Path


def load_wav(path: Path):
    """Load WAV file as float32 array."""
    with wave.open(str(path), 'rb') as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        if n_channels == 2:
            samples = samples[::2]  # Take left channel
        return samples.astype(np.float32) / 32768.0, sr


def save_wav(path: Path, samples: np.ndarray, sr: int):
    """Save float32 array as WAV file."""
    samples_int = (samples * 32768.0).clip(-32768, 32767).astype(np.int16)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples_int.tobytes())


def process_audio(model, mic: np.ndarray, sr: int):
    """
    Process audio using AEC model.
    
    Model expects:
    - Input: [1, 1, 256, 8] - 256 freq bins, 8 frames
    - Output: [1, 2, 256, 1] - mask for 2 channels
    """
    n_fft = 512
    hop = 128
    n_frames = 8
    n_freq = 256
    
    # Pad audio
    pad_len = n_fft + (n_frames - 1) * hop
    mic_padded = np.pad(mic, (0, max(0, pad_len - len(mic))))
    
    total_frames = (len(mic_padded) - n_fft) // hop + 1
    print(f"Processing {total_frames} frames...")
    
    # Output buffer
    output = np.zeros(len(mic), dtype=np.float32)
    window = np.hanning(n_fft)
    
    # Hidden states
    h_uni = torch.zeros(1, 1, 32)
    h_bi = torch.zeros(2, 1, 32)
    
    all_masks = []
    
    with torch.no_grad():
        for frame_idx in range(0, total_frames - n_frames + 1, n_frames // 2):
            # Build input: [1, 1, 256, 8]
            input_tensor = np.zeros((1, 1, n_freq, n_frames), dtype=np.float32)
            
            for j in range(n_frames):
                start = (frame_idx + j) * hop
                frame = mic_padded[start:start+n_fft] * window
                spectrum = np.fft.rfft(frame)
                mag = np.abs(spectrum[:n_freq])
                input_tensor[0, 0, :, j] = np.log1p(mag)
            
            # Run model
            x = torch.from_numpy(input_tensor)
            mask, h_uni, h_bi = model(x, h_uni, h_bi)
            mask = mask.numpy()[0, 0, :, 0]  # [256]
            all_masks.append(np.mean(mask))
            
            # Apply mask to frames
            for j in range(n_frames):
                out_frame_idx = frame_idx + j
                start = out_frame_idx * hop
                
                if start + n_fft <= len(mic):
                    frame = mic[start:start+n_fft] * window
                    spectrum = np.fft.rfft(frame)
                    
                    # Extend mask
                    full_mask = np.ones(n_fft // 2 + 1, dtype=np.float32)
                    full_mask[:n_freq] = mask
                    
                    # Apply mask
                    mag = np.abs(spectrum) * full_mask
                    phase = np.angle(spectrum)
                    spectrum_out = mag * np.exp(1j * phase)
                    
                    frame_out = np.fft.irfft(spectrum_out, n_fft)
                    output[start:start+n_fft] += frame_out * window
    
    # Normalize overlap
    norm = np.zeros(len(output))
    for i in range(0, total_frames - n_frames + 1, n_frames // 2):
        for j in range(n_frames):
            start = (i + j) * hop
            if start + n_fft <= len(output):
                norm[start:start+n_fft] += window ** 2
    
    valid = norm > 1e-8
    output[valid] /= norm[valid]
    output[~valid] = mic[~valid] if len(mic) == len(output) else 0
    
    print(f"  Average mask: {np.mean(all_masks):.3f}")
    return output


def main():
    parser = argparse.ArgumentParser(description='Test AEC model on audio')
    parser.add_argument('--model', '-m', type=Path, 
                        default=Path('mgk-decompiler/aec_model.pt'),
                        help='TorchScript model file')
    parser.add_argument('--mic', type=Path, required=True, help='Microphone input WAV')
    parser.add_argument('--lpb', type=Path, help='Loopback WAV (optional)')
    parser.add_argument('--output', '-o', type=Path, required=True, help='Output WAV')
    args = parser.parse_args()
    
    print("=== AEC Model Test ===\n")
    
    # Load model
    print(f"Loading model: {args.model}")
    model = torch.jit.load(str(args.model))
    model.eval()
    
    # Load audio
    mic, sr = load_wav(args.mic)
    print(f"Loaded mic: {len(mic)} samples, {len(mic)/sr:.2f}s @ {sr}Hz")
    
    # Process
    output = process_audio(model, mic, sr)
    
    # Save
    save_wav(args.output, output, sr)
    print(f"\nSaved: {args.output}")
    
    # Stats
    mic_rms = np.sqrt(np.mean(mic**2))
    out_rms = np.sqrt(np.mean(output**2))
    print(f"\nRMS: input={mic_rms:.4f}, output={out_rms:.4f}")
    print(f"Change: {20*np.log10(out_rms/mic_rms + 1e-10):.1f} dB")


if __name__ == '__main__':
    main()

