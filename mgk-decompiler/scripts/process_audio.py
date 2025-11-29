#!/usr/bin/env python3
"""
Process audio files using the exported ONNX model.

This demonstrates running the MGK-derived model on real audio.
Note: Since the ONNX model is a simplified representation, the actual
AEC processing would require the original MGK weights which are INT8 quantized.
"""

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("Error: onnxruntime not found. Install with: pip install onnxruntime")
    sys.exit(1)


def load_wav(path: Path) -> tuple:
    """Load WAV file and return (samples, sample_rate)."""
    with wave.open(str(path), 'rb') as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)
        
        # Convert to numpy array
        samples = np.frombuffer(audio_data, dtype=np.int16)
        # Normalize to float32 [-1, 1]
        samples = samples.astype(np.float32) / 32768.0
        
    return samples, sample_rate


def save_wav(path: Path, samples: np.ndarray, sample_rate: int):
    """Save samples to WAV file."""
    # Convert back to int16
    samples_int = (samples * 32768.0).clip(-32768, 32767).astype(np.int16)
    
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples_int.tobytes())


def process_audio(model_path: Path, mic_path: Path, lpb_path: Path, output_path: Path):
    """Process audio through the ONNX model."""
    print(f"Loading model: {model_path}")
    session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    
    # Get model input info
    input_info = session.get_inputs()[0]
    print(f"Model input: {input_info.name}, shape={input_info.shape}")
    
    # Load audio files
    print(f"Loading mic audio: {mic_path}")
    mic_samples, mic_sr = load_wav(mic_path)
    print(f"  Samples: {len(mic_samples)}, Sample rate: {mic_sr} Hz")
    print(f"  Duration: {len(mic_samples) / mic_sr:.2f} seconds")
    
    print(f"Loading loopback audio: {lpb_path}")
    lpb_samples, lpb_sr = load_wav(lpb_path)
    print(f"  Samples: {len(lpb_samples)}, Sample rate: {lpb_sr} Hz")
    
    # For AEC, we typically need:
    # - mic signal (near-end + echo)
    # - loopback signal (far-end reference)
    # The model should output: near-end signal with echo removed
    
    # Process in frames (typical AEC uses 512 or 1024 sample frames)
    frame_size = 512  # 32ms at 16kHz
    hop_size = frame_size // 2
    
    # Our simplified model takes [batch, seq, features]
    # We'll treat audio frames as the sequence
    features = 64  # From our model config
    
    # Model expects [batch=1, seq=16, features=64] = 1024 samples per chunk
    seq_len = 16
    chunk_size = seq_len * features  # 16 * 64 = 1024 samples

    n_chunks = len(mic_samples) // chunk_size
    print(f"\nProcessing {n_chunks} chunks of {chunk_size} samples each...")

    output_samples = np.zeros_like(mic_samples)

    # Process audio in chunks
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size

        # Extract chunk and reshape to [1, 16, 64]
        mic_chunk = mic_samples[start:end]
        input_data = mic_chunk.reshape(1, seq_len, features).astype(np.float32)

        # Run inference
        output = session.run(None, {input_info.name: input_data})[0]

        # Reshape output back to 1D and store
        output_samples[start:end] = output.flatten()

    # Handle remaining samples (just copy them)
    remaining_start = n_chunks * chunk_size
    if remaining_start < len(mic_samples):
        output_samples[remaining_start:] = mic_samples[remaining_start:]
    
    # Save output
    print(f"\nSaving output: {output_path}")
    save_wav(output_path, output_samples, mic_sr)
    
    print("\nProcessing complete!")
    print(f"  Input duration: {len(mic_samples) / mic_sr:.2f}s")
    print(f"  Output duration: {len(output_samples) / mic_sr:.2f}s")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Process audio with MGK-derived ONNX model')
    parser.add_argument('--model', '-m', type=Path, required=True, help='ONNX model path')
    parser.add_argument('--mic', type=Path, required=True, help='Microphone input WAV')
    parser.add_argument('--lpb', type=Path, required=True, help='Loopback/far-end WAV')
    parser.add_argument('--output', '-o', type=Path, required=True, help='Output WAV path')
    args = parser.parse_args()
    
    for path, name in [(args.model, 'model'), (args.mic, 'mic'), (args.lpb, 'lpb')]:
        if not path.exists():
            print(f"Error: {name} file not found: {path}")
            sys.exit(1)
    
    success = process_audio(args.model, args.mic, args.lpb, args.output)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

