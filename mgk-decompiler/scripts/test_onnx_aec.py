#!/usr/bin/env python3
"""
Test the ONNX AEC model exported from MGK decompiler.
"""

import argparse
import wave
import numpy as np
import onnxruntime as ort
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


def print_model_info(session: ort.InferenceSession):
    """Print model input/output information."""
    print("=== Model Info ===")
    print("Inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.shape} ({inp.type})")
    print("Outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.shape} ({out.type})")


def process_audio(session: ort.InferenceSession, mic: np.ndarray, sr: int):
    """Process audio using the ONNX AEC model."""
    n_fft = 512
    hop = 128
    n_frames = 8  # Context window
    n_freq = 256

    # Get input/output names
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_name = session.get_outputs()[0].name

    print(f"\n=== Processing Audio ===")
    print(f"Input: {input_name} shape={input_shape}")
    print(f"Audio: {len(mic)} samples @ {sr}Hz")

    # Prepare input based on expected shape
    # Shape is likely [batch, channels, height, width] = [-1, 32, 64, 64]
    batch_size = 1
    channels = input_shape[1] if len(input_shape) > 1 and input_shape[1] != -1 else 32
    height = input_shape[2] if len(input_shape) > 2 and input_shape[2] != -1 else 64
    width = input_shape[3] if len(input_shape) > 3 and input_shape[3] != -1 else 64

    # Create a test input
    total_frames = (len(mic) - n_fft) // hop + 1
    print(f"Total frames: {total_frames}")

    # For now, just process with a dummy input to verify the model runs
    test_input = np.random.randn(batch_size, channels, height, width).astype(np.float32)
    print(f"Test input shape: {test_input.shape}")

    try:
        outputs = session.run([output_name], {input_name: test_input})
        output = outputs[0]
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
        print(f"Output mean: {output.mean():.4f}")
        return output
    except Exception as e:
        print(f"Error running model: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Test ONNX AEC model')
    parser.add_argument('--model', '-m', type=Path, required=True, help='ONNX model path')
    parser.add_argument('--mic', type=Path, help='Microphone input WAV (optional)')
    parser.add_argument('--output', '-o', type=Path, help='Output WAV (optional)')
    args = parser.parse_args()

    print("=== ONNX AEC Model Test ===\n")

    # Load model
    print(f"Loading model: {args.model}")
    session = ort.InferenceSession(str(args.model), providers=['CPUExecutionProvider'])
    print_model_info(session)

    # Load audio if provided
    if args.mic and args.mic.exists():
        mic, sr = load_wav(args.mic)
        print(f"\nLoaded audio: {args.mic}")
        output = process_audio(session, mic, sr)

        if output is not None and args.output:
            # For now just save a dummy output
            dummy_output = mic.copy()  # Just copy input
            save_wav(args.output, dummy_output, sr)
            print(f"\nSaved output: {args.output}")
    else:
        # Run with dummy input to verify model loads
        print("\nNo audio provided, running with dummy input...")
        input_shape = session.get_inputs()[0].shape
        batch_size = 1
        shape = [batch_size if d == -1 else d for d in input_shape]
        test_input = np.random.randn(*shape).astype(np.float32)
        
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        print(f"Input shape: {test_input.shape}")
        try:
            outputs = session.run([output_name], {input_name: test_input})
            output = outputs[0]
            print(f"Output shape: {output.shape}")
            print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
            print("\nModel runs successfully!")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == '__main__':
    main()

