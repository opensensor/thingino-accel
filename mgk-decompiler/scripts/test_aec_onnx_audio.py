#!/usr/bin/env python3
"""Test ONNX AEC model with real audio files."""

import argparse
import wave
import numpy as np
import onnxruntime as ort
from pathlib import Path
from scipy.signal import stft, istft

# Audio processing parameters
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 128
N_FREQ = 256  # N_FFT // 2
N_FRAMES = 8


def load_wav(path):
    """Load WAV file as float32 array."""
    with wave.open(str(path), 'rb') as wf:
        sr = wf.getframerate()
        samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        return samples.astype(np.float32) / 32768.0, sr


def save_wav(path, samples, sr=SAMPLE_RATE):
    """Save float32 array as WAV file."""
    samples_int = (samples * 32768.0).clip(-32768, 32767).astype(np.int16)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples_int.tobytes())


def compute_stft(audio):
    """Compute STFT magnitude and phase."""
    _, _, Zxx = stft(audio, fs=SAMPLE_RATE, nperseg=N_FFT, noverlap=N_FFT-HOP_LENGTH)
    return np.abs(Zxx), np.angle(Zxx)


def compute_istft(magnitude, phase):
    """Compute inverse STFT."""
    Zxx = magnitude * np.exp(1j * phase)
    _, audio = istft(Zxx, fs=SAMPLE_RATE, nperseg=N_FFT, noverlap=N_FFT-HOP_LENGTH)
    return audio.astype(np.float32)


def process_with_onnx(session, mic_audio, lpb_audio):
    """Process audio through ONNX AEC model."""
    # Compute STFT
    mic_mag, mic_phase = compute_stft(mic_audio)
    lpb_mag, _ = compute_stft(lpb_audio)
    
    n_frames_total = mic_mag.shape[1]
    output_mag = np.zeros_like(mic_mag)
    
    print(f"  STFT shape: {mic_mag.shape}")
    print(f"  Processing {n_frames_total // N_FRAMES} chunks...")
    
    # Process in chunks of N_FRAMES
    for start in range(0, n_frames_total - N_FRAMES + 1, N_FRAMES):
        # Get mic magnitude chunk [256, 8]
        mic_chunk = mic_mag[:N_FREQ, start:start+N_FRAMES]
        
        # Normalize to reasonable range
        mic_max = mic_chunk.max() + 1e-8
        mic_chunk_norm = mic_chunk / mic_max
        
        # Reshape for model: [batch, freq, frames]
        input_data = mic_chunk_norm[np.newaxis, :, :].astype(np.float32)
        
        # Run inference
        outputs = session.run(None, {'input': input_data})
        mask = outputs[0][0]  # [256, 2]
        
        # Apply mask (average of 2 output channels)
        mask_avg = mask.mean(axis=1)  # [256]
        output_mag[:N_FREQ, start:start+N_FRAMES] = mic_chunk * mask_avg[:, np.newaxis]
    
    # Reconstruct audio
    return compute_istft(output_mag, mic_phase)[:len(mic_audio)]


def main():
    parser = argparse.ArgumentParser(description="Test ONNX AEC model with audio")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--mic", required=True, help="Microphone audio WAV")
    parser.add_argument("--lpb", required=True, help="Loopback audio WAV")
    parser.add_argument("--output", required=True, help="Output audio WAV")
    args = parser.parse_args()

    print(f"=== ONNX AEC Audio Test ===")
    print(f"Model: {args.model}")
    
    # Load model
    session = ort.InferenceSession(args.model)
    print(f"Input: {session.get_inputs()[0].name}, shape={session.get_inputs()[0].shape}")
    print(f"Output: {session.get_outputs()[0].name}, shape={session.get_outputs()[0].shape}")
    
    # Load audio
    print(f"\nLoading audio...")
    mic_audio, mic_sr = load_wav(args.mic)
    lpb_audio, lpb_sr = load_wav(args.lpb)
    
    # Align lengths
    min_len = min(len(mic_audio), len(lpb_audio))
    mic_audio = mic_audio[:min_len]
    lpb_audio = lpb_audio[:min_len]
    
    print(f"  Mic: {len(mic_audio)} samples ({len(mic_audio)/SAMPLE_RATE:.2f}s)")
    print(f"  LPB: {len(lpb_audio)} samples ({len(lpb_audio)/SAMPLE_RATE:.2f}s)")
    
    # Process
    print(f"\nProcessing...")
    output_audio = process_with_onnx(session, mic_audio, lpb_audio)
    
    # Save
    save_wav(args.output, output_audio)
    print(f"\nSaved: {args.output}")
    
    # Compare RMS
    mic_rms = np.sqrt(np.mean(mic_audio**2))
    out_rms = np.sqrt(np.mean(output_audio**2))
    print(f"\nRMS comparison:")
    print(f"  Input:  {mic_rms:.4f}")
    print(f"  Output: {out_rms:.4f}")
    print(f"  Change: {20*np.log10(out_rms/(mic_rms+1e-8)):.1f} dB")


if __name__ == "__main__":
    main()

