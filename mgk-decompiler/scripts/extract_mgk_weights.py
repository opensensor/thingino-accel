#!/usr/bin/env python3
"""
Extract weights and quantization parameters from MGK files.

This script analyzes an MGK file and exports:
- INT8 weights as a binary file
- Quantization scales and metadata as JSON

Usage:
    python extract_mgk_weights.py input.mgk output_dir/
"""

import argparse
import json
import struct
import numpy as np
from pathlib import Path


def find_rodata_section(data: bytes) -> tuple:
    """Find .rodata section in ELF file."""
    # ELF header parsing
    e_shoff = struct.unpack('<I', data[32:36])[0]
    e_shentsize = struct.unpack('<H', data[46:48])[0]
    e_shnum = struct.unpack('<H', data[48:50])[0]
    e_shstrndx = struct.unpack('<H', data[50:52])[0]
    
    # Get section name string table
    shstrtab_offset = e_shoff + e_shstrndx * e_shentsize
    shstrtab_sh_offset = struct.unpack('<I', data[shstrtab_offset + 16:shstrtab_offset + 20])[0]
    shstrtab_sh_size = struct.unpack('<I', data[shstrtab_offset + 20:shstrtab_offset + 24])[0]
    shstrtab = data[shstrtab_sh_offset:shstrtab_sh_offset + shstrtab_sh_size]
    
    # Find .rodata section
    for i in range(e_shnum):
        sh_offset = e_shoff + i * e_shentsize
        sh_name_idx = struct.unpack('<I', data[sh_offset:sh_offset + 4])[0]
        sh_offset_val = struct.unpack('<I', data[sh_offset + 16:sh_offset + 20])[0]
        sh_size = struct.unpack('<I', data[sh_offset + 20:sh_offset + 24])[0]
        
        name_end = shstrtab.find(b'\x00', sh_name_idx)
        name = shstrtab[sh_name_idx:name_end].decode('utf-8', errors='ignore')
        
        if name == '.rodata':
            return sh_offset_val, sh_size
    
    return None, None


def extract_quantization_scales(data: bytes, rodata_offset: int, rodata_size: int) -> list:
    """Extract quantization scale factors from .rodata section."""
    rodata = data[rodata_offset:rodata_offset + rodata_size]
    
    # Find FP32 values that look like quantization scales (0.001 to 0.1 range)
    scales = []
    for i in range(0, len(rodata) - 4, 4):
        val = np.frombuffer(rodata[i:i+4], dtype=np.float32)[0]
        if 0.001 < abs(val) < 0.1 and not np.isnan(val) and not np.isinf(val):
            scales.append({
                'offset': rodata_offset + i,
                'value': float(val)
            })
    
    return scales


def find_weight_data(data: bytes) -> tuple:
    """Find the weight data section in the MGK file."""
    # Weight data is typically at a fixed offset after the ELF sections
    # For AEC_T41_16K_NS_OUT_UC.mgk, it's at 0x79294
    weight_offset = 0x79294
    weight_size = 153644
    
    # Verify this looks like weight data (dense INT8 values)
    if weight_offset + weight_size <= len(data):
        weights = np.frombuffer(data[weight_offset:weight_offset + weight_size], dtype=np.int8)
        std = np.std(weights.astype(np.float32))
        if std > 30:  # Dense weight data typically has high variance
            return weight_offset, weight_size
    
    return None, None


def extract_weights(mgk_path: Path, output_dir: Path):
    """Extract weights and metadata from MGK file."""
    print(f"Extracting from: {mgk_path}")
    
    with open(mgk_path, 'rb') as f:
        data = f.read()
    
    print(f"  File size: {len(data):,} bytes")
    
    # Find .rodata section
    rodata_offset, rodata_size = find_rodata_section(data)
    if rodata_offset is None:
        print("  ERROR: Could not find .rodata section")
        return False
    print(f"  .rodata: offset=0x{rodata_offset:x}, size={rodata_size:,}")
    
    # Extract quantization scales
    scales = extract_quantization_scales(data, rodata_offset, rodata_size)
    print(f"  Found {len(scales)} potential quantization scales")
    
    # Find weight data
    weight_offset, weight_size = find_weight_data(data)
    if weight_offset is None:
        print("  ERROR: Could not find weight data")
        return False
    print(f"  Weights: offset=0x{weight_offset:x}, size={weight_size:,}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save weights as binary file
    weights_path = output_dir / "weights.bin"
    with open(weights_path, 'wb') as f:
        f.write(data[weight_offset:weight_offset + weight_size])
    print(f"  Saved weights: {weights_path}")
    
    # Save metadata as JSON
    metadata = {
        'source_file': str(mgk_path),
        'file_size': len(data),
        'weight_offset': weight_offset,
        'weight_size': weight_size,
        'rodata_offset': rodata_offset,
        'rodata_size': rodata_size,
        'quantization_scales': scales,
    }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Extract weights from MGK files')
    parser.add_argument('input', type=Path, help='Input MGK file')
    parser.add_argument('output', type=Path, help='Output directory')
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    success = extract_weights(args.input, args.output)
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())

