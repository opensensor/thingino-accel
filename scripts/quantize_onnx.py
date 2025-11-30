#!/usr/bin/env python3
"""
Quantize ONNX model to INT8 using ONNX Runtime static quantization.
Produces a model with QDQ (QuantizeLinear/DequantizeLinear) operators
that contain explicit scales.
"""

import os
import sys
import glob
import numpy as np
from PIL import Image

import onnxruntime
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
)


class YOLOCalibrationDataReader(CalibrationDataReader):
    """Calibration data reader for YOLO models using real images."""
    
    def __init__(self, calibration_dir: str, input_name: str = "images",
                 input_shape: tuple = (1, 3, 640, 640), max_samples: int = 100,
                 dtype = np.float32):
        self.calibration_dir = calibration_dir
        self.input_name = input_name
        self.input_shape = input_shape
        self.max_samples = max_samples
        self.dtype = dtype
        
        # Find all images
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_files.extend(glob.glob(os.path.join(calibration_dir, ext)))
            self.image_files.extend(glob.glob(os.path.join(calibration_dir, ext.upper())))
        
        self.image_files = self.image_files[:max_samples]
        self.current_idx = 0
        
        print(f"Found {len(self.image_files)} calibration images")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for YOLO: resize, normalize, CHW format."""
        img = Image.open(image_path).convert('RGB')

        # Resize to input size
        _, _, h, w = self.input_shape
        img = img.resize((w, h), Image.BILINEAR)

        # Convert to numpy and normalize to [0, 1]
        img_np = np.array(img).astype(np.float32) / 255.0

        # HWC -> CHW
        img_np = img_np.transpose(2, 0, 1)

        # Add batch dimension
        img_np = np.expand_dims(img_np, axis=0)

        # Convert to float16 if needed
        if self.dtype == np.float16:
            img_np = img_np.astype(np.float16)

        return img_np
    
    def get_next(self):
        if self.current_idx >= len(self.image_files):
            return None
        
        image_path = self.image_files[self.current_idx]
        self.current_idx += 1
        
        try:
            img_data = self.preprocess_image(image_path)
            return {self.input_name: img_data}
        except Exception as e:
            print(f"Warning: Failed to load {image_path}: {e}")
            return self.get_next()  # Try next image
    
    def rewind(self):
        self.current_idx = 0


class RandomCalibrationDataReader(CalibrationDataReader):
    """Fallback: generate random calibration data."""

    def __init__(self, input_name: str = "images",
                 input_shape: tuple = (1, 3, 640, 640), num_samples: int = 50,
                 dtype = np.float32):
        self.input_name = input_name
        self.input_shape = input_shape
        self.num_samples = num_samples
        self.current_idx = 0
        self.dtype = dtype
        print(f"Using {num_samples} random calibration samples")

    def get_next(self):
        if self.current_idx >= self.num_samples:
            return None
        self.current_idx += 1

        # Generate random data in [0, 1] range (typical for normalized images)
        data = np.random.rand(*self.input_shape).astype(self.dtype)
        return {self.input_name: data}
    
    def rewind(self):
        self.current_idx = 0


def quantize_yolo(input_model: str, output_model: str, calibration_dir: str = None):
    """Quantize YOLO ONNX model to INT8 with QDQ format."""
    
    print(f"Input model: {input_model}")
    print(f"Output model: {output_model}")
    
    # Get input info from model
    sess = onnxruntime.InferenceSession(input_model, providers=['CPUExecutionProvider'])
    input_info = sess.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape
    
    # Handle dynamic dimensions
    input_shape = [1 if isinstance(d, str) else d for d in input_shape]
    input_shape = tuple(input_shape)
    
    print(f"Input: {input_name} {input_shape}")
    
    # Determine input dtype
    input_dtype_str = str(input_info.type)
    if 'float16' in input_dtype_str:
        input_dtype = np.float16
        print("Input dtype: float16")
    else:
        input_dtype = np.float32
        print("Input dtype: float32")

    # Create calibration data reader
    if calibration_dir and os.path.isdir(calibration_dir):
        data_reader = YOLOCalibrationDataReader(
            calibration_dir, input_name, input_shape, dtype=input_dtype
        )
    else:
        print("No calibration directory provided, using random data")
        data_reader = RandomCalibrationDataReader(input_name, input_shape, dtype=input_dtype)
    
    # Quantize with QDQ format (contains explicit scales)
    print("\nQuantizing model...")
    quantize_static(
        model_input=input_model,
        model_output=output_model,
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QDQ,  # QDQ format has explicit QuantizeLinear/DequantizeLinear
        per_channel=False,  # Per-tensor quantization (simpler)
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )
    
    print(f"\nQuantized model saved to: {output_model}")
    
    # Print model info
    import onnx
    model = onnx.load(output_model)
    
    # Count QDQ nodes
    qdq_count = sum(1 for node in model.graph.node 
                   if node.op_type in ['QuantizeLinear', 'DequantizeLinear'])
    print(f"QDQ nodes: {qdq_count}")
    
    # Print first few QuantizeLinear scales
    print("\nFirst few quantization scales:")
    count = 0
    for init in model.graph.initializer:
        if '_scale' in init.name and count < 5:
            scale = np.frombuffer(init.raw_data, dtype=np.float32)
            print(f"  {init.name}: {scale}")
            count += 1


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python quantize_onnx.py <input.onnx> <output.onnx> [calibration_dir]")
        print("\nExample:")
        print("  python quantize_onnx.py models/yolov5n.onnx models/yolov5n_int8_qdq.onnx test_images/")
        sys.exit(1)
    
    input_model = sys.argv[1]
    output_model = sys.argv[2]
    calibration_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    quantize_yolo(input_model, output_model, calibration_dir)

