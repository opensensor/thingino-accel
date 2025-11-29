#!/usr/bin/env python3
"""
Complete MGK Model Extraction

Extracts all weights, scales, and metadata from an MGK file.
Outputs a complete model package that can be loaded for inference.
"""

import argparse
import json
import numpy as np
import struct
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class MGKExtractor:
    """Complete MGK model extractor."""
    
    # Known layer weight sizes based on analysis
    LAYER_SIZES = {
        'layer_46_gru_bidir': 12864,  # Bidirectional GRU
        'layer_37_gru': 4096,          # Unidirectional GRU
        'layer_2_feature': 320,
        'layer_4_feature': 3648,
        'layer_8_feature': 1024,
        'layer_10_feature': 2496,
        'layer_14_feature': 1024,
        'layer_16_feature': 2112,
        'layer_20_feature': 832,
        'layer_22_feature': 1772,
        'layer_26_feature': 832,
        'layer_28_feature': 1408,
        'layer_32_feature': 768,
        'layer_35_feature': 704,
        'layer_41_feature': 704,
        'layer_44_feature': 576,
        'layer_58_feature': 576,
        'layer_63_feature': 448,
        'layer_68_feature': 448,
        'layer_73_feature': 448,
        'layer_78_feature': 320,
    }
    
    # Weight offsets discovered from analysis
    LAYER_OFFSETS = {
        'layer_46_gru_bidir': 0x00000,
        'layer_63_feature': 0x03500,
        'layer_68_feature': 0x03900,
        'layer_35_feature': 0x03d00,
        'layer_73_feature': 0x04100,
        'layer_44_feature': 0x11f00,
        'layer_58_feature': 0x12300,
        'layer_78_feature': 0x12700,
        'layer_4_feature': 0x12a00,
        'layer_16_feature': 0x13b00,
        'layer_2_feature': 0x14b00,
        'layer_20_feature': 0x21180,
        'layer_26_feature': 0x215c0,
        'layer_28_feature': 0x21a40,
        'layer_37_gru': 0x220c0,
        'layer_10_feature': 0x231c0,
        'layer_32_feature': 0x23cc0,
        'layer_41_feature': 0x24100,
        'layer_8_feature': 0x24500,
        'layer_14_feature': 0x24a00,
        'layer_22_feature': 0x25140,
    }
    
    def __init__(self, mgk_path: Path):
        self.mgk_path = mgk_path
        with open(mgk_path, 'rb') as f:
            self.data = f.read()
        
        # Find weight base offset
        self.weight_base = self._find_weight_base()
        
    def _find_weight_base(self) -> int:
        """Find the weight data base offset."""
        # Look for the appended data marker
        # The weight data starts after the ELF sections
        # Based on analysis: 0x79294
        return 0x79294
    
    def extract_layer_weights(self, layer_name: str) -> Optional[np.ndarray]:
        """Extract weights for a specific layer."""
        if layer_name not in self.LAYER_OFFSETS:
            return None
        
        offset = self.weight_base + self.LAYER_OFFSETS[layer_name]
        size = self.LAYER_SIZES.get(layer_name, 0)
        
        if size == 0:
            return None
        
        return np.frombuffer(self.data[offset:offset + size], dtype=np.int8)
    
    def extract_all_weights(self) -> Dict[str, np.ndarray]:
        """Extract all layer weights."""
        weights = {}
        for layer_name in self.LAYER_OFFSETS:
            w = self.extract_layer_weights(layer_name)
            if w is not None:
                weights[layer_name] = w
        return weights
    
    def extract_quantization_scales(self) -> List[Dict]:
        """Extract quantization scales from .rodata section."""
        scales = []
        
        # Scales are in .rodata section, starting around 0x6d410
        rodata_start = 0x6c640
        rodata_end = 0x73640
        
        # Look for scale patterns (small positive floats)
        offset = 0x6d410
        while offset < rodata_end:
            try:
                values = []
                for i in range(4):
                    val = struct.unpack('<f', self.data[offset + i*4:offset + i*4 + 4])[0]
                    values.append(val)
                
                # Check if these look like scales (small positive values)
                if all(0.001 < v < 1.0 for v in values):
                    scales.append({
                        'offset': offset,
                        'values': values
                    })
                    offset += 16
                else:
                    offset += 4
            except:
                offset += 4
        
        return scales
    
    def save_extraction(self, output_dir: Path):
        """Save complete extraction to output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        layers_dir = output_dir / 'layers'
        layers_dir.mkdir(exist_ok=True)
        
        # Extract and save weights
        weights = self.extract_all_weights()
        layer_info = {}
        
        for name, w in weights.items():
            # Save as binary
            w.tofile(layers_dir / f'{name}.bin')
            
            layer_info[name] = {
                'offset': self.LAYER_OFFSETS[name],
                'size': len(w),
                'dtype': 'int8',
                'std': float(np.std(w.astype(np.float32))),
                'zeros_pct': float(np.sum(w == 0) / len(w) * 100)
            }
        
        # Extract scales
        scales = self.extract_quantization_scales()
        
        # Save metadata
        metadata = {
            'source_file': str(self.mgk_path),
            'weight_base_offset': self.weight_base,
            'layers': layer_info,
            'quantization_scales': scales,
            'model_info': {
                'type': 'AEC',
                'sample_rate': 16000,
                'frame_size': 256,
                'input_shape': [1, 1, 256, 8],
                'output_shape': [1, 1, 256, 2],
                'hidden_shape': [64, 1, 1, 32]
            }
        }
        
        with open(output_dir / 'model.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Extracted {len(weights)} layers to {output_dir}")
        return metadata


def main():
    parser = argparse.ArgumentParser(description='Extract MGK model')
    parser.add_argument('mgk_file', type=Path, help='Input MGK file')
    parser.add_argument('output_dir', type=Path, help='Output directory')
    args = parser.parse_args()
    
    extractor = MGKExtractor(args.mgk_file)
    extractor.save_extraction(args.output_dir)


if __name__ == '__main__':
    main()

