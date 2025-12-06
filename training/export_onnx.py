"""
Export TinyDet to ONNX format for Mars compiler.

Usage:
  python export_onnx.py --weights ./checkpoints/tinydet_best.pth --output tinydet.onnx
  
Then convert to Mars format:
  cd ../mars-compiler && cargo run -- -i ../training/tinydet.onnx -o ../models/tinydet.mars
"""

import argparse
import torch
import torch.nn as nn
import onnx
from onnx import numpy_helper
import numpy as np

from tinydet import TinyDet, INPUT_W, INPUT_H


def fuse_conv_bn(conv, bn):
    """Fuse Conv2d and BatchNorm2d into a single Conv2d."""
    # Get conv params
    w_conv = conv.weight.clone()
    b_conv = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels)
    
    # Get bn params  
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    
    # Fuse: w_fused = w_conv * gamma / sqrt(var + eps)
    #       b_fused = (b_conv - mean) * gamma / sqrt(var + eps) + beta
    std = torch.sqrt(var + eps)
    w_fused = w_conv * (gamma / std).view(-1, 1, 1, 1)
    b_fused = (b_conv - mean) * gamma / std + beta
    
    # Create new conv with fused weights
    fused = nn.Conv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size,
        conv.stride, conv.padding, conv.dilation, conv.groups, bias=True
    )
    fused.weight.data = w_fused
    fused.bias.data = b_fused
    return fused


def fuse_model(model):
    """Fuse all Conv+BN pairs in the model."""
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            fuse_model(module)
        elif hasattr(module, 'conv') and hasattr(module, 'bn'):
            # ConvBNReLU block
            fused_conv = fuse_conv_bn(module.conv, module.bn)
            module.conv = fused_conv
            module.bn = nn.Identity()
        else:
            fuse_model(module)
    return model


def export_onnx(model, output_path, input_w=INPUT_W, input_h=INPUT_H, opset=13):
    """Export model to ONNX with static shapes."""
    model.eval()

    # Dummy input: [B, C, H, W] = [1, 3, 192, 320]
    dummy_input = torch.randn(1, 3, input_h, input_w)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None  # Static shapes for T41
    )

    # Verify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    return onnx_model


def print_model_info(onnx_path):
    """Print ONNX model info."""
    model = onnx.load(onnx_path)
    
    print(f"\nONNX Model: {onnx_path}")
    print(f"Opset: {model.opset_import[0].version}")
    
    print("\nInputs:")
    for inp in model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: {shape}")
    
    print("\nOutputs:")
    for out in model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {shape}")
    
    print(f"\nNodes: {len(model.graph.node)}")
    
    # Count params
    total_params = 0
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        total_params += arr.size
    print(f"Parameters: {total_params:,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Path to .pth weights')
    parser.add_argument('--output', type=str, default='tinydet.onnx', help='Output ONNX path')
    parser.add_argument('--input-w', type=int, default=INPUT_W, help='Input width')
    parser.add_argument('--input-h', type=int, default=INPUT_H, help='Input height')
    parser.add_argument('--no-fuse', action='store_true', help='Skip Conv+BN fusion')
    parser.add_argument('--num-classes', type=int, default=4, help='Number of detection classes')
    args = parser.parse_args()

    print(f"Loading weights from {args.weights}")
    model = TinyDet(num_classes=args.num_classes)
    state_dict = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(state_dict)

    if not args.no_fuse:
        print("Fusing Conv+BN layers...")
        model = fuse_model(model)

    print(f"Exporting to {args.output} (input: {args.input_w}x{args.input_h})")
    export_onnx(model, args.output, args.input_w, args.input_h)

    print_model_info(args.output)

    print(f"\n✓ Export complete!")
    print(f"\nNext: Convert to Mars format:")
    print(f"  cd ../mars-compiler")
    print(f"  cargo run -- -i ../training/{args.output} -o ../models/tinydet.mars")


if __name__ == '__main__':
    main()

