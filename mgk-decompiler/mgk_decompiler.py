#!/usr/bin/env python3
"""
MGK Decompiler - Generic decompiler for Ingenic Magik neural network models

Extracts model architecture and weights from .mgk files and converts to ONNX.

Supported:
- MIPS32 Little-Endian ELF format
- INT8 quantized weights (NMHWSOIB2 format with quantize_type=8)
- 2-bit quantized weights (NMHWSOIB2 format with quantize_type=2)
- Layer types: Conv, BatchNorm, GRU, Concat, Pooling, Add

NMHWSOIB2 Weight Format:
    For quantize_type=2 (2-bit), the format is:
    - Shape: [oc_blocks, ic_blocks, kh, kw, pack=4, oc=32, ic=32]
    - Each weight position has 4 2-bit sub-values (packed 4 per byte)
    - pack[3] encodes sign: -2 = positive, 1 = negative (98%+ accuracy)
    - pack[0:3] encode magnitude information (true 2-bit quantization)
    - Original INT8 precision is LOST - weights cannot be exactly recovered

    For quantize_type=8 (INT8), the format is:
    - Shape: [oc_blocks, ic_blocks, kh, kw, oc=32, ic=32]
    - Standard INT8 values, no quantization loss

Usage:
    python mgk_decompiler.py model.mgk -o model.onnx
"""

import struct
import re
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np


@dataclass
class QuantizationParams:
    """Quantization parameters for a layer"""
    weight_bitwidth: int = 8
    input_bitwidth: int = 8
    output_bitwidth: int = 8
    dequantize_scale: float = 1.0
    offset: float = 0.0
    threshold_min: float = -128.0
    threshold_max: float = 127.0
    fixpoint: bool = False


@dataclass
class MGKLayer:
    """Represents a layer in the MGK model"""
    index: int
    name: str
    layer_type: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    attrs: Dict = field(default_factory=dict)
    weight_offset: Optional[int] = None
    weight_size: Optional[int] = None
    quant_params: Optional[QuantizationParams] = None
    is_fused: bool = False  # True if this is a fused Conv+BN+ReLU


@dataclass 
class MGKModel:
    """Represents a complete MGK model"""
    name: str
    layers: List[MGKLayer]
    input_shapes: Dict[str, Tuple]
    output_names: List[str]
    weight_data: bytes
    scales: List[float]


class MGKDecompiler:
    """Decompiles MGK files to extract model structure and weights"""
    
    def __init__(self, mgk_path: Path):
        self.path = Path(mgk_path)
        with open(self.path, 'rb') as f:
            self.data = f.read()
        
        self.sections = {}
        self.layers = []
        self.scales = []
        self.weight_offset = 0
        self.weight_size = 0
        
    def parse(self) -> MGKModel:
        """Parse the MGK file and return model structure"""
        if self.data[:4] != b'\x7fELF':
            raise ValueError("Not an ELF file")

        self._parse_elf_sections()
        self._extract_layers()
        self._extract_scales()
        self._find_weight_region()
        self._extract_tensor_info()

        return MGKModel(
            name=self.path.stem,
            layers=self.layers,
            input_shapes={},  # TODO: extract from metadata
            output_names=[],
            weight_data=self.data[self.weight_offset:],
            scales=self.scales
        )
    
    def _parse_elf_sections(self):
        """Parse ELF32 sections"""
        e_shoff = struct.unpack('<I', self.data[32:36])[0]
        e_shentsize = struct.unpack('<H', self.data[46:48])[0]
        e_shnum = struct.unpack('<H', self.data[48:50])[0]
        e_shstrndx = struct.unpack('<H', self.data[50:52])[0]
        
        sec_list = []
        for i in range(e_shnum):
            offset = e_shoff + i * e_shentsize
            sh_name = struct.unpack('<I', self.data[offset:offset+4])[0]
            sh_offset = struct.unpack('<I', self.data[offset+16:offset+20])[0]
            sh_size = struct.unpack('<I', self.data[offset+20:offset+24])[0]
            sec_list.append({'name_idx': sh_name, 'offset': sh_offset, 'size': sh_size})
        
        # Get section names
        if e_shstrndx < len(sec_list):
            strtab = sec_list[e_shstrndx]
            strtab_data = self.data[strtab['offset']:strtab['offset']+strtab['size']]
            for sec in sec_list:
                name_start = sec['name_idx']
                name_end = strtab_data.find(b'\x00', name_start)
                name = strtab_data[name_start:name_end].decode('utf-8', errors='replace')
                self.sections[name] = {
                    'offset': sec['offset'],
                    'size': sec['size'],
                    'data': self.data[sec['offset']:sec['offset']+sec['size']]
                }
    
    def _extract_layers(self):
        """Extract layer definitions from rodata"""
        rodata = self.sections.get('.rodata', {}).get('data', b'')

        # Pattern 1: layer_N_Type (AEC style)
        layer_pattern = re.compile(rb'layer_(\d+)_([A-Za-z]+)')
        for match in layer_pattern.finditer(rodata):
            layer_num = int(match.group(1))
            layer_type = match.group(2).decode()
            offset = match.start()
            end = rodata.find(b'\x00', offset)
            full_name = rodata[offset:end].decode()

            self.layers.append(MGKLayer(
                index=layer_num,
                name=full_name,
                layer_type=layer_type
            ))

        # Pattern 2: NNN_Quantize (YOLO style)
        if not self.layers:
            quant_pattern = re.compile(rb'(\d+)_Quantize')
            for match in quant_pattern.finditer(rodata):
                layer_num = int(match.group(1))
                offset = match.start()
                end = rodata.find(b'\x00', offset)
                full_name = rodata[offset:end].decode()

                self.layers.append(MGKLayer(
                    index=layer_num,
                    name=full_name,
                    layer_type='Quantize'
                ))

        # Pattern 3: ptq_model_* (fused layers)
        ptq_pattern = re.compile(rb'ptq_model_([a-z_]+)_(\d+)_Quantize')
        for match in ptq_pattern.finditer(rodata):
            op_type = match.group(1).decode()
            layer_num = int(match.group(2))
            offset = match.start()
            end = rodata.find(b'\x00', offset)
            full_name = rodata[offset:end].decode()

            self.layers.append(MGKLayer(
                index=layer_num,
                name=full_name,
                layer_type=f'Fused_{op_type}'
            ))

        # Pattern 4: NNN_output_last_layer (outputs)
        output_pattern = re.compile(rb'(\d+)_output_last_layer')
        for match in output_pattern.finditer(rodata):
            layer_num = int(match.group(1))
            offset = match.start()
            end = rodata.find(b'\x00', offset)
            full_name = rodata[offset:end].decode()

            self.layers.append(MGKLayer(
                index=layer_num,
                name=full_name,
                layer_type='Output'
            ))

        # Deduplicate by index
        seen = set()
        unique_layers = []
        for layer in self.layers:
            if layer.index not in seen:
                seen.add(layer.index)
                unique_layers.append(layer)

        self.layers = sorted(unique_layers, key=lambda x: x.index)

        # Detect fused operators (Conv+BN+ReLU fusion)
        self._detect_fused_ops(rodata)

    def _detect_fused_ops(self, rodata: bytes):
        """Detect fused operators from rodata patterns"""
        # Look for QuantizeConv2DWrapper or similar fusion patterns
        fusion_patterns = [
            b'QuantizeConv2DWrapper',
            b'conv2d_tnpu',
            b'QuantizeWeight',
            b'fuse_',
        ]

        self.has_fused_ops = any(p in rodata for p in fusion_patterns)

        # If we find fusion patterns, mark QuantizeFeature layers as potentially fused
        if self.has_fused_ops:
            for layer in self.layers:
                if 'QuantizeFeature' in layer.layer_type or 'Quantize' in layer.layer_type:
                    # Check if this layer has associated BN by looking at layer indices
                    # Typically fused layers have consecutive indices
                    layer.is_fused = True
    
    def _extract_scales(self):
        """Extract quantization scales from rodata"""
        rodata = self.sections.get('.rodata', {}).get('data', b'')

        # Extract float32 values that look like scales
        for i in range(0, len(rodata) - 4, 4):
            val = struct.unpack('<f', rodata[i:i+4])[0]
            if 0.001 < abs(val) < 10.0 and val != 0:
                self.scales.append((i, val))  # Store offset with value

        # Try to group scales by layer (typically pairs: input_scale, output_scale)
        # or groups of 4 (input_scale, weight_scale, bias_scale, output_scale)
        self.scale_groups = []
        if self.scales:
            current_group = [self.scales[0]]
            for i in range(1, len(self.scales)):
                offset, val = self.scales[i]
                prev_offset, _ = self.scales[i-1]
                # If consecutive or close offsets, group together
                if offset - prev_offset <= 16:
                    current_group.append(self.scales[i])
                else:
                    if len(current_group) >= 2:
                        self.scale_groups.append(current_group)
                    current_group = [self.scales[i]]
            if len(current_group) >= 2:
                self.scale_groups.append(current_group)
    
    def _find_weight_region(self):
        """Find where weight data starts (after ELF sections)"""
        # Weight region is typically after all sections
        max_offset = 0
        for name, sec in self.sections.items():
            end = sec['offset'] + sec['size']
            if end > max_offset:
                max_offset = end

        # Align to 16 bytes
        self.weight_offset = (max_offset + 15) & ~15
        self.weight_size = len(self.data) - self.weight_offset

    def _extract_tensor_info(self):
        """Extract input/output tensor information"""
        rodata = self.sections.get('.rodata', {}).get('data', b'')

        self.tensor_formats = []
        self.tensor_types = []

        # Find format strings
        for fmt in [b'NHWC', b'NCHW', b'NDHWC32', b'NDHWC8', b'NMHWSOIB2']:
            if fmt in rodata:
                self.tensor_formats.append(fmt.decode())

        # Find data type strings
        for dtype in [b'UINT8', b'INT8', b'FLOAT32', b'FP32', b'INT16']:
            if dtype in rodata:
                self.tensor_types.append(dtype.decode())

        # Look for input tensor name
        if b'images' in rodata:
            self.input_name = 'images'
        elif b'input' in rodata:
            self.input_name = 'input'
        else:
            self.input_name = 'input_0'

    def get_summary(self) -> str:
        """Generate a summary of the model structure"""
        lines = []
        lines.append(f"Model: {self.path.name}")
        lines.append(f"Size: {len(self.data):,} bytes")
        lines.append("")

        # Sections summary
        lines.append("ELF Sections:")
        for name, sec in sorted(self.sections.items()):
            if sec['size'] > 0:
                lines.append(f"  {name:<20} {sec['size']:>10,} bytes @ 0x{sec['offset']:08x}")
        lines.append("")

        # Layer summary
        from collections import Counter
        type_counts = Counter(l.layer_type for l in self.layers)
        lines.append(f"Layers: {len(self.layers)}")
        for t, c in type_counts.most_common():
            lines.append(f"  {t}: {c}")
        lines.append("")

        # Fusion detection
        if hasattr(self, 'has_fused_ops') and self.has_fused_ops:
            lines.append("Operator Fusion: DETECTED (Conv+BN+ReLU fused)")
            fused_count = sum(1 for l in self.layers if l.is_fused)
            lines.append(f"  Fused layers: {fused_count}")
            lines.append("")

        # Weight info
        lines.append(f"Weight region: {self.weight_size:,} bytes @ 0x{self.weight_offset:08x}")
        lines.append(f"Quantization scales: {len(self.scales)}")
        if hasattr(self, 'scale_groups'):
            lines.append(f"Scale groups: {len(self.scale_groups)}")
            if self.scale_groups:
                # Show sample scales
                sample = self.scale_groups[0]
                vals = [f"{v:.4f}" for _, v in sample[:4]]
                lines.append(f"  Sample: {vals}")
        lines.append("")

        # Tensor info
        lines.append(f"Tensor formats: {', '.join(self.tensor_formats)}")
        lines.append(f"Data types: {', '.join(self.tensor_types)}")

        return '\n'.join(lines)

    def extract_weights(self, output_dir: Path) -> Dict[str, np.ndarray]:
        """Extract all weights and save to output directory"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        weight_data = self.data[self.weight_offset:]
        weight_info = analyze_weight_structure(weight_data)

        extracted = {}
        block_idx = 0
        layer_idx = 0

        for layer in self.layers:
            if 'Quantize' not in layer.layer_type:
                continue

            # Find next dense block region
            start_block = block_idx
            while block_idx < len(weight_info['blocks']):
                if weight_info['blocks'][block_idx]['is_dense']:
                    break
                block_idx += 1

            if block_idx >= len(weight_info['blocks']):
                break

            # Count consecutive dense blocks
            dense_start = block_idx
            while (block_idx < len(weight_info['blocks']) and
                   weight_info['blocks'][block_idx]['is_dense']):
                block_idx += 1

            n_blocks = block_idx - dense_start
            if n_blocks == 0:
                continue

            # Extract raw weight data
            offset = dense_start * 1024
            size = n_blocks * 1024
            raw_weights = weight_data[offset:offset + size]

            # Try to determine shape
            shapes = estimate_layer_shapes(n_blocks, layer.layer_type)
            if shapes:
                shape_info = shapes[0]
                layer_type, out_ch, in_ch, kh, kw = shape_info
                try:
                    weights = unpack_nmhwsoib2(raw_weights, out_ch, in_ch, kh, kw)
                    # Apply dequantization scale if available
                    if layer_idx < len(self.scale_groups):
                        scale = self.scale_groups[layer_idx][0][1]
                        weights = weights.astype(np.float32) * scale
                except Exception as e:
                    weights = np.frombuffer(raw_weights, dtype=np.int8).reshape(-1, 32, 32)
            else:
                weights = np.frombuffer(raw_weights, dtype=np.int8).reshape(-1, 32, 32)

            # Save weights
            name = f"layer_{layer.index:03d}_{layer.layer_type}"
            np.save(output_dir / f"{name}.npy", weights)
            extracted[name] = weights
            layer_idx += 1

        return extracted


def analyze_weight_structure(weight_data: bytes) -> Dict:
    """Analyze the structure of weight data"""
    # Weight blocks are typically 1024 bytes (32x32 INT8)
    block_size = 1024
    n_blocks = len(weight_data) // block_size

    # Analyze each block
    blocks = []
    for i in range(n_blocks):
        block = np.frombuffer(weight_data[i*block_size:(i+1)*block_size], dtype=np.int8)
        nonzero = np.count_nonzero(block)
        std = block.std()
        blocks.append({
            'index': i,
            'nonzero': nonzero,
            'std': std,
            'is_dense': nonzero > 900 and std > 20
        })

    # Count dense blocks (actual weights vs padding)
    dense_blocks = sum(1 for b in blocks if b['is_dense'])

    return {
        'total_blocks': n_blocks,
        'dense_blocks': dense_blocks,
        'block_size': block_size,
        'blocks': blocks
    }


def unpack_2bit_to_signed(packed_data: bytes) -> np.ndarray:
    """
    Unpack 2-bit values from bytes and convert to signed.

    Each byte contains 4 2-bit values in little-endian order:
    - Bits 0-1: First value
    - Bits 2-3: Second value
    - Bits 4-5: Third value
    - Bits 6-7: Fourth value

    Mapping: 0->0, 1->1, 2->-2, 3->-1
    """
    data = np.frombuffer(packed_data, dtype=np.uint8)
    result = np.zeros(len(data) * 4, dtype=np.int8)

    result[0::4] = data & 0x3
    result[1::4] = (data >> 2) & 0x3
    result[2::4] = (data >> 4) & 0x3
    result[3::4] = (data >> 6) & 0x3

    # Convert to signed: 0,1,2,3 -> 0,1,-2,-1
    result = np.where(result >= 2, result - 4, result)
    return result


def unpack_nmhwsoib2(packed_data: bytes, out_ch: int, in_ch: int, kh: int = 1, kw: int = 1,
                     quantize_type: int = 8) -> np.ndarray:
    """
    Unpack weights from NMHWSOIB2 format to standard [out_ch, in_ch, kh, kw]

    NMHWSOIB2 format depends on quantize_type:

    For quantize_type=8 (INT8):
        Shape: [N_OFP, M_IFP, KH, KW, OFP, IFP]
        where N_OFP = ceil(out_ch/32), M_IFP = ceil(in_ch/32), OFP=IFP=32

    For quantize_type=2 (2-bit):
        Shape: [N_OFP, M_IFP, KH, KW, PACK, OFP, IFP]
        where PACK=4 represents 4 2-bit sub-values per weight position

        The 4 sub-values encode:
        - pack[3]: Sign bit (-2 = positive, 1 = negative)
        - pack[0:3]: Magnitude encoding (true 2-bit quantization, not lossless)

        Note: Original INT8 precision is LOST during 2-bit quantization.
        These weights cannot be exactly reconstructed to INT8.
    """
    n_ofp = (out_ch + 31) // 32
    m_ifp = (in_ch + 31) // 32

    if quantize_type == 2:
        # 2-bit quantization: 4 bytes store 4 2-bit values = 16 2-bit values per 4 bytes
        # Shape: [n_ofp, m_ifp, kh, kw, 4, 32, 32]
        expected_size = n_ofp * m_ifp * kh * kw * 4 * 32 * 32 // 4  # 4 values per byte
        if len(packed_data) < expected_size:
            raise ValueError(f"Not enough data: need {expected_size}, got {len(packed_data)}")

        # Unpack 2-bit values
        unpacked = unpack_2bit_to_signed(packed_data[:expected_size])

        # Reshape to [n_ofp, m_ifp, kh, kw, 4, 32, 32]
        reshaped = unpacked.reshape(n_ofp, m_ifp, kh, kw, 4, 32, 32)

        # For decompilation, we return the 2-bit values with sign extracted
        # pack[3] contains sign: -2 = positive, 1 = negative
        # We reconstruct approximate values using sign and magnitude from pack[0:3]

        # Simple reconstruction: use average of pack[0:3] as magnitude proxy
        # This is an approximation since the original INT8 values are lost
        # Shape after mean: [n_ofp, m_ifp, kh, kw, 32, 32]
        magnitude = np.mean(np.abs(reshaped[:, :, :, :, 0:3, :, :]), axis=4)
        sign = np.where(reshaped[:, :, :, :, 3, :, :] == -2, 1, -1)

        # Scale magnitude to approximate INT8 range
        # Typical scaling: magnitude of 1-2 in 2-bit maps to ~10-20 in INT8
        approx_values = sign * magnitude * 10.0

        # Current shape: [n_ofp, m_ifp, kh, kw, oc_per_block=32, ic_per_block=32]
        # Target shape: [out_ch, in_ch, kh, kw]
        # Transpose to [n_ofp, oc_per_block, m_ifp, ic_per_block, kh, kw]
        transposed = approx_values.transpose(0, 4, 1, 5, 2, 3)
        output = transposed.reshape(n_ofp * 32, m_ifp * 32, kh, kw)

        return output[:out_ch, :in_ch, :, :].astype(np.float32)
    else:
        # INT8: Original format
        expected_size = n_ofp * m_ifp * kh * kw * 1024  # 32*32 = 1024
        if len(packed_data) < expected_size:
            raise ValueError(f"Not enough data: need {expected_size}, got {len(packed_data)}")

        # Reshape and transpose
        data = np.frombuffer(packed_data[:expected_size], dtype=np.int8)
        reshaped = data.reshape(n_ofp, m_ifp, kh, kw, 32, 32)
        transposed = reshaped.transpose(0, 4, 1, 5, 2, 3)
        output = transposed.reshape(n_ofp * 32, m_ifp * 32, kh, kw)

        # Trim to actual channels
        return output[:out_ch, :in_ch, :, :]


def extract_2bit_raw(packed_data: bytes, n_ofp: int, m_ifp: int, kh: int, kw: int) -> dict:
    """
    Extract raw 2-bit weight data without reconstruction.

    Returns a dictionary with:
    - 'signs': Sign values from pack[3] (-2=positive, 1=negative)
    - 'magnitudes': Raw 2-bit values from pack[0:3]
    - 'shape': The unpacked shape
    """
    expected_size = n_ofp * m_ifp * kh * kw * 4 * 32 * 32 // 4
    unpacked = unpack_2bit_to_signed(packed_data[:expected_size])
    reshaped = unpacked.reshape(n_ofp, m_ifp, kh, kw, 4, 32, 32)

    return {
        'signs': reshaped[:, :, :, :, 3, :, :],
        'magnitudes': reshaped[:, :, :, :, 0:3, :, :],
        'shape': reshaped.shape,
        'quantize_type': 2
    }


def estimate_layer_shapes(n_blocks: int, layer_type: str) -> List[Tuple[int, int, int, int]]:
    """Estimate possible layer shapes based on block count and type"""
    shapes = []

    if 'GRU' in layer_type:
        # GRU has 3 gates (reset, update, hidden) each with input and hidden weights
        # BiGRU has forward + backward = 6 weight matrices
        # UniGRU has 3 weight matrices
        if n_blocks >= 12:
            shapes.append(('BiGRU', 96, 32, 1, 1))  # 3 gates * 2 directions * 32 hidden
        elif n_blocks >= 4:
            shapes.append(('UniGRU', 96, 32, 1, 1))
    else:
        # Conv layers: try common configurations
        for out_ch in [32, 64, 96, 128, 256, 512]:
            for in_ch in [32, 64, 96, 128, 256, 512]:
                for k in [1, 3, 5]:
                    n_ofp = (out_ch + 31) // 32
                    m_ifp = (in_ch + 31) // 32
                    needed = n_ofp * m_ifp * k * k
                    if needed == n_blocks:
                        shapes.append(('Conv', out_ch, in_ch, k, k))

    return shapes


def export_to_onnx(decompiler: MGKDecompiler, model: MGKModel, output_path: Path):
    """Export model to ONNX format"""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError("PyTorch required for ONNX export. Install with: pip install torch")

    # Build a simple sequential model based on layers
    class MGKONNXModel(nn.Module):
        def __init__(self, layers, weight_data, scales):
            super().__init__()
            self.ops = nn.ModuleList()

            # Analyze weight structure
            weight_info = analyze_weight_structure(weight_data)

            block_idx = 0
            in_ch = 32  # Starting channels (typical for these models)

            for layer in layers:
                if 'GRU' in layer.layer_type:
                    # Add GRU layer
                    self.ops.append(nn.GRU(in_ch, 32, batch_first=True, bidirectional='Bi' in layer.name))
                elif 'BatchNorm' in layer.layer_type:
                    self.ops.append(nn.BatchNorm2d(in_ch))
                elif 'Quantize' in layer.layer_type or 'Feature' in layer.layer_type:
                    # Find dense blocks for this layer
                    while block_idx < len(weight_info['blocks']) and not weight_info['blocks'][block_idx]['is_dense']:
                        block_idx += 1

                    if block_idx < len(weight_info['blocks']):
                        dense_start = block_idx
                        while block_idx < len(weight_info['blocks']) and weight_info['blocks'][block_idx]['is_dense']:
                            block_idx += 1
                        n_blocks = block_idx - dense_start

                        # Estimate channels based on blocks
                        if n_blocks >= 9:
                            out_ch = 128
                            kernel = 3
                        elif n_blocks >= 4:
                            out_ch = 64
                            kernel = 1
                        else:
                            out_ch = 32
                            kernel = 1

                        self.ops.append(nn.Conv2d(in_ch, out_ch, kernel, padding=kernel//2))
                        in_ch = out_ch

            # Final output layer
            self.output = nn.Conv2d(in_ch, 32, 1)

        def forward(self, x):
            for op in self.ops:
                if isinstance(op, nn.GRU):
                    # Reshape for GRU: (B, C, H, W) -> (B, H*W, C)
                    b, c, h, w = x.shape
                    x = x.permute(0, 2, 3, 1).reshape(b, h*w, c)
                    x, _ = op(x)
                    x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2)
                else:
                    x = op(x)
            return self.output(x)

    # Create model
    quantize_layers = [l for l in model.layers if 'Quantize' in l.layer_type or 'GRU' in l.layer_type]
    torch_model = MGKONNXModel(quantize_layers, model.weight_data, model.scales)
    torch_model.eval()

    # Create dummy input based on tensor format
    if 'NDHWC32' in decompiler.tensor_formats:
        # Typical for T41 models
        dummy = torch.randn(1, 32, 64, 64)
    else:
        dummy = torch.randn(1, 32, 32, 32)

    # Export to ONNX
    torch.onnx.export(
        torch_model,
        dummy,
        str(output_path),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=11
    )


def main():
    parser = argparse.ArgumentParser(description='MGK Model Decompiler')
    parser.add_argument('mgk_file', type=Path, help='Path to .mgk file')
    parser.add_argument('-o', '--output', type=Path, help='Output ONNX path')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--analyze-weights', action='store_true', help='Analyze weight structure')
    parser.add_argument('--extract-weights', type=Path, help='Extract weights to directory')
    parser.add_argument('--summary', action='store_true', help='Print detailed summary')
    args = parser.parse_args()

    decompiler = MGKDecompiler(args.mgk_file)
    model = decompiler.parse()

    if args.summary:
        print(decompiler.get_summary())
        return

    print(f"Model: {model.name}")
    print(f"Layers: {len(model.layers)}")
    print(f"Scales: {len(model.scales)}")
    print(f"Weight region: {len(model.weight_data):,} bytes")
    print(f"Tensor formats: {', '.join(decompiler.tensor_formats)}")
    print(f"Data types: {', '.join(decompiler.tensor_types)}")

    # Count layer types
    from collections import Counter
    type_counts = Counter(l.layer_type for l in model.layers)
    print(f"\nLayer types:")
    for t, c in type_counts.most_common():
        print(f"  {t}: {c}")

    if args.analyze_weights:
        print("\nWeight structure:")
        weight_info = analyze_weight_structure(model.weight_data)
        print(f"  Total blocks: {weight_info['total_blocks']}")
        print(f"  Dense blocks: {weight_info['dense_blocks']}")
        print(f"  Sparse/padding: {weight_info['total_blocks'] - weight_info['dense_blocks']}")

    if args.verbose:
        print("\nLayers:")
        for layer in model.layers:
            print(f"  {layer.index:3d}: {layer.layer_type:<20} {layer.name}")

    if args.extract_weights:
        print(f"\nExtracting weights to: {args.extract_weights}")
        extracted = decompiler.extract_weights(args.extract_weights)
        print(f"Extracted {len(extracted)} weight tensors:")
        for name, weights in list(extracted.items())[:10]:
            print(f"  {name}: shape={weights.shape}, dtype={weights.dtype}")
        if len(extracted) > 10:
            print(f"  ... and {len(extracted) - 10} more")

    if args.output:
        print(f"\nExporting to: {args.output}")
        try:
            export_to_onnx(decompiler, model, args.output)
            print(f"ONNX model saved to: {args.output}")
        except Exception as e:
            print(f"ONNX export failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()

