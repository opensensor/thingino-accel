#!/usr/bin/env python3
"""
MGK Model Extractor - Complete extraction of Ingenic Magik neural network models.

This script extracts the complete model structure from MGK files including:
- Layer topology from device log
- INT8 weights with proper layer mapping
- Quantization scales from .rodata
- Dense weight region analysis

Based on reverse engineering of the MGK format using Binary Ninja analysis.
"""

import argparse
import json
import struct
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class MGKExtractor:
    """Extract model data from MGK files."""
    
    def __init__(self, mgk_path: Path, log_path: Optional[Path] = None):
        self.mgk_path = mgk_path
        self.log_path = log_path
        
        with open(mgk_path, 'rb') as f:
            self.data = f.read()
        
        self.elf_header = self._parse_elf_header()
        self.sections = self._parse_sections()
        self.tensors = []
        self.layers = {}
        
        if log_path and log_path.exists():
            self._parse_log()
    
    def _parse_elf_header(self) -> dict:
        """Parse ELF32 header."""
        if self.data[:4] != b'\x7fELF':
            raise ValueError("Not a valid ELF file")
        
        e_shoff = struct.unpack('<I', self.data[0x20:0x24])[0]
        e_shentsize = struct.unpack('<H', self.data[0x2e:0x30])[0]
        e_shnum = struct.unpack('<H', self.data[0x30:0x32])[0]
        e_shstrndx = struct.unpack('<H', self.data[0x32:0x34])[0]
        
        return {
            'shoff': e_shoff,
            'shentsize': e_shentsize,
            'shnum': e_shnum,
            'shstrndx': e_shstrndx,
            'elf_end': e_shoff + e_shentsize * e_shnum
        }
    
    def _parse_sections(self) -> List[dict]:
        """Parse ELF section headers."""
        sections = []
        strtab_offset = self.elf_header['shoff'] + self.elf_header['shstrndx'] * self.elf_header['shentsize']
        strtab_sh_offset = struct.unpack('<I', self.data[strtab_offset + 16:strtab_offset + 20])[0]
        strtab_sh_size = struct.unpack('<I', self.data[strtab_offset + 20:strtab_offset + 24])[0]
        strtab = self.data[strtab_sh_offset:strtab_sh_offset + strtab_sh_size]
        
        for i in range(self.elf_header['shnum']):
            offset = self.elf_header['shoff'] + i * self.elf_header['shentsize']
            sh_name = struct.unpack('<I', self.data[offset:offset + 4])[0]
            sh_offset = struct.unpack('<I', self.data[offset + 16:offset + 20])[0]
            sh_size = struct.unpack('<I', self.data[offset + 20:offset + 24])[0]
            
            name_end = strtab.find(b'\x00', sh_name)
            name = strtab[sh_name:name_end].decode('ascii', errors='replace')
            
            sections.append({'name': name, 'offset': sh_offset, 'size': sh_size})
        
        return sections
    
    def _parse_log(self):
        """Parse tensor info from device log."""
        with open(self.log_path, 'r') as f:
            log_content = f.read()
        
        pattern = r"\[VENUS\] TensorInfo\[(\d+)\]: name='([^']+)' is_input=(\d) is_output=(\d)\s+dtype_str='([^']+)' layout='([^']+)' channel=(\d+)\s+shape=\[([^\]]+)\]"
        
        for match in re.finditer(pattern, log_content):
            self.tensors.append({
                'index': int(match.group(1)),
                'name': match.group(2),
                'is_input': match.group(3) == '1',
                'is_output': match.group(4) == '1',
                'dtype': match.group(5),
                'layout': match.group(6),
                'channel': int(match.group(7)),
                'shape': [int(x) for x in match.group(8).split(',')]
            })
        
        # Group tensors by layer
        for t in self.tensors:
            name = t['name']
            if name.startswith('layer_'):
                parts = name.split('_')
                layer_num = int(parts[1])
                layer_type = '_'.join(parts[2:]).replace(':1', '')
                
                if layer_num not in self.layers:
                    self.layers[layer_num] = {'type': layer_type, 'tensors': []}
                self.layers[layer_num]['tensors'].append(t)
    
    def get_rodata(self) -> Optional[bytes]:
        """Get .rodata section data."""
        for s in self.sections:
            if s['name'] == '.rodata':
                return self.data[s['offset']:s['offset'] + s['size']]
        return None
    
    def find_weight_offset(self) -> Tuple[int, int]:
        """Find weight data offset and size."""
        weight_offset = self.elf_header['elf_end']
        
        # Skip sparse header data to find dense weights
        for i in range(weight_offset, len(self.data), 256):
            chunk = self.data[i:i + 256]
            non_zero = sum(1 for b in chunk if b != 0)
            if non_zero > 200:
                weight_offset = i
                break
        
        return weight_offset, len(self.data) - weight_offset
    
    def extract_weights(self) -> np.ndarray:
        """Extract weight data as numpy array."""
        offset, size = self.find_weight_offset()
        return np.frombuffer(self.data[offset:offset + size], dtype=np.int8)
    
    def find_dense_regions(self, weights: np.ndarray, window: int = 64) -> List[dict]:
        """Find dense weight regions."""
        regions = []
        region_start = None
        
        for i in range(0, len(weights), window):
            chunk = weights[i:i + window]
            std = np.std(chunk.astype(np.float32))
            is_dense = std > 30
            
            if is_dense and region_start is None:
                region_start = i
            elif not is_dense and region_start is not None:
                regions.append({'start': region_start, 'end': i, 'size': i - region_start})
                region_start = None
        
        if region_start is not None:
            regions.append({'start': region_start, 'end': len(weights), 'size': len(weights) - region_start})
        
        # Merge nearby regions
        merged = []
        for r in regions:
            if merged and r['start'] - merged[-1]['end'] < 256:
                merged[-1]['end'] = r['end']
                merged[-1]['size'] = merged[-1]['end'] - merged[-1]['start']
            else:
                merged.append(r.copy())
        
        return [r for r in merged if r['size'] >= 256]

    def extract_scales(self) -> List[dict]:
        """Extract quantization scales from .rodata."""
        rodata = self.get_rodata()
        if not rodata:
            return []

        scales = []
        i = 0
        while i < len(rodata) - 4:
            val = struct.unpack('<f', rodata[i:i + 4])[0]
            if 0.001 < abs(val) < 0.5 and not np.isnan(val) and not np.isinf(val):
                # Check for consecutive scales
                group = [val]
                j = i + 4
                while j < len(rodata) - 4:
                    next_val = struct.unpack('<f', rodata[j:j + 4])[0]
                    if 0.001 < abs(next_val) < 0.5 and not np.isnan(next_val) and not np.isinf(next_val):
                        group.append(next_val)
                        j += 4
                    else:
                        break

                if len(group) >= 2:
                    scales.append({'offset': i, 'values': group})
                    i = j
                    continue
            i += 4

        return scales

    def export(self, output_dir: Path):
        """Export extracted model data."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract weights
        weights = self.extract_weights()
        weight_offset, weight_size = self.find_weight_offset()

        # Find dense regions
        dense_regions = self.find_dense_regions(weights)

        # Extract scales
        scales = self.extract_scales()

        # Save weights
        weights_path = output_dir / 'weights.bin'
        with open(weights_path, 'wb') as f:
            f.write(weights.tobytes())
        print(f"Saved weights: {weights_path} ({len(weights):,} bytes)")

        # Analyze and save each dense region
        for i, region in enumerate(dense_regions):
            region_data = weights[region['start']:region['end']]
            region['mean'] = float(np.mean(region_data.astype(np.float32)))
            region['std'] = float(np.std(region_data.astype(np.float32)))
            region['zeros_pct'] = float(np.sum(region_data == 0) / len(region_data) * 100)

        # Build layer weight mapping based on expected sizes
        layer_weights = self._map_weights_to_layers(dense_regions)

        # Save metadata
        metadata = {
            'source_file': str(self.mgk_path),
            'file_size': len(self.data),
            'elf_end': self.elf_header['elf_end'],
            'weight_offset': weight_offset,
            'weight_size': weight_size,
            'sections': [s for s in self.sections if s['size'] > 0],
            'tensors': self.tensors,
            'layers': {str(k): v for k, v in self.layers.items()},
            'dense_regions': dense_regions,
            'quantization_scales': scales,
            'layer_weights': layer_weights
        }

        metadata_path = output_dir / 'model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"Saved metadata: {metadata_path}")

        # Save individual layer weights
        layers_dir = output_dir / 'layers'
        layers_dir.mkdir(exist_ok=True)

        for layer_name, layer_info in layer_weights.items():
            if layer_info.get('region_idx') is not None:
                region = dense_regions[layer_info['region_idx']]
                layer_data = weights[region['start']:region['end']]
                layer_path = layers_dir / f'{layer_name}.bin'
                with open(layer_path, 'wb') as f:
                    f.write(layer_data.tobytes())

        print(f"Saved layer weights to: {layers_dir}")

    def _map_weights_to_layers(self, dense_regions: List[dict]) -> dict:
        """Map dense regions to layers based on expected sizes."""
        # Expected weight sizes based on layer types from log analysis
        expected = {
            'layer_2': {'type': 'Feature', 'size': 8*32 + 32},  # 8->32 channels
            'layer_4': {'type': 'Feature', 'size': 32*32*4 + 32},  # 4x1 kernel
            'layer_8': {'type': 'Feature', 'size': 32*32 + 32},
            'layer_10': {'type': 'Feature', 'size': 32*32*4 + 32},
            'layer_14': {'type': 'Feature', 'size': 32*32 + 32},
            'layer_16': {'type': 'Feature', 'size': 32*32*4 + 32},
            'layer_20': {'type': 'Feature', 'size': 32*32 + 32},
            'layer_22': {'type': 'Feature', 'size': 32*32*4 + 32},
            'layer_26': {'type': 'Feature', 'size': 32*32 + 32},
            'layer_28': {'type': 'Feature', 'size': 32*32*4 + 32},
            'layer_32': {'type': 'Feature', 'size': 32*32 + 32},
            'layer_34': {'type': 'BatchNorm', 'size': 64},
            'layer_35': {'type': 'Feature', 'size': 32*32 + 32},
            'layer_37': {'type': 'GRU', 'size': 3*32*32*2 + 3*32*2},  # ih + hh + bias
            'layer_41': {'type': 'Feature', 'size': 32*32 + 32},
            'layer_43': {'type': 'BatchNorm', 'size': 64},
            'layer_44': {'type': 'Feature', 'size': 32*32 + 32},
            'layer_46': {'type': 'GRU_bidir', 'size': 2*(3*32*32*2 + 3*32*2)},
            'layer_58': {'type': 'Feature', 'size': 32*32 + 32},
            'layer_63': {'type': 'Feature', 'size': 32*32 + 32},
            'layer_68': {'type': 'Feature', 'size': 32*32 + 32},
            'layer_73': {'type': 'Feature', 'size': 32*32 + 32},
            'layer_78': {'type': 'Feature', 'size': 32*32 + 32},
            'layer_80': {'type': 'BatchNorm', 'size': 16},
        }

        # Sort regions by size for matching
        sorted_regions = sorted(enumerate(dense_regions), key=lambda x: x[1]['size'], reverse=True)
        sorted_layers = sorted(expected.items(), key=lambda x: x[1]['size'], reverse=True)

        mapping = {}
        used_regions = set()

        for layer_name, layer_info in sorted_layers:
            best_match = None
            best_diff = float('inf')

            for idx, region in sorted_regions:
                if idx in used_regions:
                    continue
                diff = abs(region['size'] - layer_info['size'])
                if diff < best_diff and diff < layer_info['size'] * 2:  # Within 2x
                    best_diff = diff
                    best_match = idx

            if best_match is not None:
                used_regions.add(best_match)
                mapping[layer_name] = {
                    'type': layer_info['type'],
                    'expected_size': layer_info['size'],
                    'region_idx': best_match,
                    'actual_size': dense_regions[best_match]['size'],
                    'offset': dense_regions[best_match]['start']
                }
            else:
                mapping[layer_name] = {
                    'type': layer_info['type'],
                    'expected_size': layer_info['size'],
                    'region_idx': None,
                    'note': 'No matching region found'
                }

        return mapping


def main():
    parser = argparse.ArgumentParser(description='Extract MGK model data')
    parser.add_argument('mgk_file', type=Path, help='Input MGK file')
    parser.add_argument('output_dir', type=Path, help='Output directory')
    parser.add_argument('--log', type=Path, help='Device log file for tensor info')
    args = parser.parse_args()

    if not args.mgk_file.exists():
        print(f"Error: {args.mgk_file} not found")
        return 1

    extractor = MGKExtractor(args.mgk_file, args.log)
    extractor.export(args.output_dir)

    print(f"\nExtracted {len(extractor.tensors)} tensors, {len(extractor.layers)} layers")
    return 0


if __name__ == '__main__':
    exit(main())

