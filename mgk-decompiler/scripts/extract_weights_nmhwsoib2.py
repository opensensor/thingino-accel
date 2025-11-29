#!/usr/bin/env python3
"""
MGK Weight Extractor - NMHWSOIB2 Format

This script extracts INT8 quantized weights from MGK files and converts them
to standard PyTorch format.

MGK Weight Format: NMHWSOIB2
- N_OFP: Number of Output Feature Planes (ceil(out_channels / 32))
- M_IFP: Number of Input Feature Planes (ceil(in_channels / 32))
- KERNEL_H, KERNEL_W: Convolution kernel dimensions
- S_BIT2: 2-bit grouping factor (4 for 8-bit weights)
- OFP: Channels per output feature plane (32)
- IFP: Channels per input feature plane (32)

For GRU weights:
- Stored as 32x32 blocks (1024 bytes each)
- Pattern: 768 bytes (24x32) + 256 bytes (8x32) = 32x32 matrix
- Biases stored after weight matrices
"""

import argparse
import json
import struct
import numpy as np
from pathlib import Path


def load_mgk_file(filepath):
    """Load MGK file and find weight region start."""
    with open(filepath, 'rb') as f:
        data = f.read()

    # Find weight region by scanning for dense data after ELF sections
    e_shoff = struct.unpack('<I', data[0x20:0x24])[0]
    e_shentsize = struct.unpack('<H', data[0x2e:0x30])[0]
    e_shnum = struct.unpack('<H', data[0x30:0x32])[0]

    # Start scanning after section header table
    scan_start = e_shoff + e_shentsize * e_shnum

    # Find first dense block (std > 50, mostly non-zero)
    for offset in range(scan_start, min(len(data), scan_start + 0x2000), 256):
        block = np.frombuffer(data[offset:offset+256], dtype=np.int8)
        if block.std() > 50 and np.sum(block != 0) > 200:
            return data, offset

    # Fallback to calculated position
    return data, scan_start


def unpack_nmhwsoib2(packed_data, out_ch, in_ch, kh, kw):
    """
    Unpack NMHWSOIB2 format to standard OIHW format.

    NMHWSOIB2: [N_OFP, M_IFP, KH, KW, S_BIT2, OFP, IFP]
    For 8-bit: [N_OFP, M_IFP, KH, KW, 4, 32, 32]

    Each 1024-byte block = 32x32 weight matrix for one spatial position.
    """
    n_ofp = (out_ch + 31) // 32
    m_ifp = (in_ch + 31) // 32

    expected_size = n_ofp * m_ifp * kh * kw * 1024
    if len(packed_data) != expected_size:
        return None

    data = np.frombuffer(packed_data, dtype=np.int8)

    # Reshape: [N_OFP, M_IFP, KH, KW, 32, 32]
    reshaped = data.reshape(n_ofp, m_ifp, kh, kw, 32, 32)

    # Transpose to OIHW: [N_OFP, 32, M_IFP, 32, KH, KW]
    transposed = reshaped.transpose(0, 4, 1, 5, 2, 3)

    # Reshape to [N_OFP*32, M_IFP*32, KH, KW]
    output = transposed.reshape(n_ofp * 32, m_ifp * 32, kh, kw)

    # Trim to actual channel sizes
    return output[:out_ch, :in_ch, :, :]


def extract_gru_weights(weight_data, offset, size, hidden_size=32, bidirectional=True):
    """
    Extract GRU weights from packed format.

    GRU has 3 gates (reset, update, new), each with weight_ih and weight_hh.
    For bidirectional: 12 weight matrices (6 per direction).
    For unidirectional: 4 blocks (different layout - possibly combined gates)
    """
    data = np.frombuffer(weight_data[offset:offset+size], dtype=np.int8)

    block_size = 1024
    num_blocks = size // block_size

    # Each block is a 32x32 matrix
    matrices = []
    for i in range(num_blocks):
        block = data[i*block_size:(i+1)*block_size]
        # The 768+256 pattern: first 24 rows, then 8 rows
        part_768 = block[:768].reshape(24, 32)
        part_256 = block[768:].reshape(8, 32)
        full_matrix = np.vstack([part_768, part_256])
        matrices.append(full_matrix)

    # Biases are in remainder
    biases = data[num_blocks*block_size:]

    if bidirectional and num_blocks >= 12:
        # Bidirectional: 12 matrices (6 per direction)
        # Order: weight_ir, weight_iz, weight_in, weight_hr, weight_hz, weight_hn
        result = {
            'forward': {
                'weight_ih': np.vstack([matrices[0], matrices[1], matrices[2]]),
                'weight_hh': np.vstack([matrices[3], matrices[4], matrices[5]]),
            },
            'backward': {
                'weight_ih': np.vstack([matrices[6], matrices[7], matrices[8]]),
                'weight_hh': np.vstack([matrices[9], matrices[10], matrices[11]]),
            },
            'biases': biases,
        }
    elif num_blocks == 4:
        # Unidirectional with 4 blocks: 2 for weight_ih, 2 for weight_hh
        # Or possibly: weight_ih (2 blocks = 64 rows), weight_hh (2 blocks = 64 rows)
        result = {
            'forward': {
                'weight_ih': np.vstack([matrices[0], matrices[1]]),  # (64, 32)
                'weight_hh': np.vstack([matrices[2], matrices[3]]),  # (64, 32)
            },
            'biases': biases,
        }
    else:
        # Generic case: just return all matrices
        result = {
            'forward': {
                'matrices': matrices,
            },
            'biases': biases,
        }

    return result


def nmhwsoib2_size(out_ch, in_ch, kh, kw):
    """Calculate expected size in NMHWSOIB2 format."""
    n_ofp = (out_ch + 31) // 32
    m_ifp = (in_ch + 31) // 32
    return n_ofp * m_ifp * kh * kw * 1024


def main():
    parser = argparse.ArgumentParser(description='Extract weights from MGK file')
    parser.add_argument('mgk_file', help='Path to MGK file')
    parser.add_argument('--output', '-o', default='extracted_weights',
                        help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load MGK file
    data, weight_start = load_mgk_file(args.mgk_file)
    weight_data = data[weight_start:]

    print(f"MGK file: {args.mgk_file}")
    print(f"Weight region: offset=0x{weight_start:x}, size={len(weight_data)} bytes")

    # Extract BiGRU weights (12864 bytes at offset 0)
    print("\n=== Extracting BiGRU Weights ===")
    bigru_weights = extract_gru_weights(weight_data, 0, 12864, bidirectional=True)
    np.save(output_dir / 'bigru_forward_weight_ih.npy', bigru_weights['forward']['weight_ih'])
    np.save(output_dir / 'bigru_forward_weight_hh.npy', bigru_weights['forward']['weight_hh'])
    np.save(output_dir / 'bigru_backward_weight_ih.npy', bigru_weights['backward']['weight_ih'])
    np.save(output_dir / 'bigru_backward_weight_hh.npy', bigru_weights['backward']['weight_hh'])
    np.save(output_dir / 'bigru_biases.npy', bigru_weights['biases'])
    print(f"  forward weight_ih: {bigru_weights['forward']['weight_ih'].shape}")
    print(f"  backward weight_ih: {bigru_weights['backward']['weight_ih'].shape}")

    # Extract Unidirectional GRU weights (4096 bytes after BiGRU)
    print("\n=== Extracting Unidirectional GRU Weights ===")
    unigru_weights = extract_gru_weights(weight_data, 12864, 4096, bidirectional=False)
    if 'weight_ih' in unigru_weights['forward']:
        np.save(output_dir / 'unigru_weight_ih.npy', unigru_weights['forward']['weight_ih'])
        np.save(output_dir / 'unigru_weight_hh.npy', unigru_weights['forward']['weight_hh'])
        print(f"  weight_ih: {unigru_weights['forward']['weight_ih'].shape}")
        print(f"  weight_hh: {unigru_weights['forward']['weight_hh'].shape}")
    else:
        print(f"  matrices: {len(unigru_weights['forward']['matrices'])} blocks")

    # Extract Conv weights (after GRU weights)
    print("\n=== Extracting Conv Weights ===")
    conv_start = 12864 + 4096  # After both GRU layers

    # Known conv configurations from model architecture
    conv_configs = [
        ('conv1', 32, 32, 1, 1),   # 1x1 conv
        ('conv2', 32, 32, 3, 3),   # 3x3 conv
        ('conv3', 32, 32, 3, 3),   # 3x3 conv
        ('conv4', 32, 32, 3, 3),   # 3x3 conv
    ]

    offset = conv_start
    for name, out_ch, in_ch, kh, kw in conv_configs:
        size = nmhwsoib2_size(out_ch, in_ch, kh, kw)
        if offset + size > len(weight_data):
            print(f"  {name}: Not enough data (need {size} bytes at offset 0x{offset:x})")
            break

        packed = weight_data[offset:offset+size]
        unpacked = unpack_nmhwsoib2(packed, out_ch, in_ch, kh, kw)

        if unpacked is not None:
            np.save(output_dir / f'{name}_weight.npy', unpacked)
            print(f"  {name}: {unpacked.shape}, std={unpacked.std():.1f}")
            offset += size
        else:
            print(f"  {name}: Failed to unpack")
            break

    print(f"\nAll weights saved to {output_dir}")


if __name__ == '__main__':
    main()

