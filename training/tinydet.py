"""
TinyDet: Minimal 3-class detector optimized for Ingenic T41 NNA/MXUv3

Architecture constraints:
- No DepthwiseConv (not implemented in mars runtime)
- No AvgPool (not implemented)
- Use ReLU not SiLU for simplicity (or SiLU with LUT overhead)
- Channel counts: multiples of 32 for NNA tiling efficiency
- 64-byte alignment friendly (INT8: 64 elements per VPR)

Target: 320x192 input (close to 16:9 aspect ratio), ~100K params, single-scale output, 3 classes
Output grid: 20x12 cells (stride 16)

V2 improvements:
- Deeper backbone with more conv layers per stage for larger receptive field
- Added dilated convolutions to expand RF without more params
- Receptive field: ~127 pixels (was ~31) - enough to see full objects
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Input dimensions - close to 16:9 aspect ratio, multiples of 32
INPUT_W = 320
INPUT_H = 192
STRIDE = 16
GRID_W = INPUT_W // STRIDE  # 20
GRID_H = INPUT_H // STRIDE  # 12


class ConvBNReLU(nn.Module):
    """Conv2d + BatchNorm + ReLU - fuses to single Conv in inference"""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, dilation=1):
        super().__init__()
        # Adjust padding for dilation
        if dilation > 1:
            padding = dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding,
                              dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class TinyDetBackbone(nn.Module):
    """
    Improved backbone for 320x192 -> 20x12 feature map with larger receptive field

    Design principles for T41:
    - Multiple conv layers per stage for deeper feature extraction
    - Dilated convolutions in later stages to expand receptive field
    - Channels: 16->32->64->64 (small, multiples of 32)
    - Use stride-2 convs for downsampling

    Stage    Output      Channels   RF contribution
    -----    ------      --------   ---------------
    Input    320x192     3          -
    Stem     160x96      16         3 + 3 = 6 pixels
    S1       80x48       32         2*(3+3) = 12 pixels
    S2       40x24       64         4*(3+5) = 32 pixels (dilation=2)
    S3       20x12       64         8*(3+9) = 96 pixels (dilation=4)

    Total effective RF: ~127 pixels (vs 31 before)
    Total: ~90K params in backbone
    """
    def __init__(self):
        super().__init__()
        # Stem: 320x192x3 -> 160x96x16 (2 layers)
        self.stem = nn.Sequential(
            ConvBNReLU(3, 16, kernel=3, stride=2, padding=1),
            ConvBNReLU(16, 16, kernel=3, stride=1, padding=1),
        )

        # Stage 1: 160x96x16 -> 80x48x32 (2 layers)
        self.s1 = nn.Sequential(
            ConvBNReLU(16, 32, kernel=3, stride=2, padding=1),
            ConvBNReLU(32, 32, kernel=3, stride=1, padding=1),
        )

        # Stage 2: 80x48x32 -> 40x24x64 (2 layers with dilation)
        self.s2 = nn.Sequential(
            ConvBNReLU(32, 64, kernel=3, stride=2, padding=1),
            ConvBNReLU(64, 64, kernel=3, stride=1, padding=2, dilation=2),  # dilation=2
        )

        # Stage 3: 40x24x64 -> 20x12x64 (2 layers with dilation)
        self.s3 = nn.Sequential(
            ConvBNReLU(64, 64, kernel=3, stride=2, padding=1),
            ConvBNReLU(64, 64, kernel=3, stride=1, padding=4, dilation=4),  # dilation=4
        )

    def forward(self, x):
        x = self.stem(x)   # 160x96x16
        x = self.s1(x)     # 80x48x32
        x = self.s2(x)     # 40x24x64
        x = self.s3(x)     # 20x12x64
        return x


class TinyDetHead(nn.Module):
    """
    Anchor-free detection head with improved spatial processing

    Output: 20x12 grid, each cell predicts:
    - 1 objectness score
    - 4 box coords (x_off, y_off, w, h) - normalized
    - 3 class scores (person, cat, dog)

    Total: 8 channels per cell = 20x12x8 = 1920 values
    """
    def __init__(self, in_ch=64, num_classes=3):
        super().__init__()
        self.num_classes = num_classes

        # Deeper head: 2 conv layers + 1x1 output
        self.head = nn.Sequential(
            ConvBNReLU(in_ch, 64, kernel=3, stride=1, padding=1),
            ConvBNReLU(64, 32, kernel=3, stride=1, padding=1),
            nn.Conv2d(32, 5 + num_classes, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        # Output: [B, 8, 12, 20]
        return self.head(x)


class TinyDet(nn.Module):
    """
    Complete TinyDet model

    Input:  [B, 3, 192, 320] - RGB image (HxW, close to 16:9)
    Output: [B, 8, 12, 20]   - Detection grid (HxW)

    Output channels:
      [0]: objectness (sigmoid for prob)
      [1:5]: box (x, y, w, h)
      [5:8]: class scores (person, dog, cat)

    Architecture optimized for Ingenic T41 NNA:
      - Only standard Conv2D (no depthwise)
      - ReLU activation (fast on MXU)
      - Channel counts: 16, 32, 64 (multiples of 16/32)
      - ~79K params, ~36M MACs
    """
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = TinyDetBackbone()
        self.head = TinyDetHead(in_ch=64, num_classes=num_classes)
        self.num_classes = num_classes
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def test_model():
    """Test model dimensions and param count"""
    model = TinyDet(num_classes=3)
    model.eval()

    # Test with 320x192 input (WxH in tensor is [B,C,H,W] so 192x320)
    x = torch.randn(1, 3, INPUT_H, INPUT_W)
    with torch.no_grad():
        y = model(x)

    print(f"Input:  {tuple(x.shape)} (B,C,H,W) = 1x3x{INPUT_H}x{INPUT_W}")
    print(f"Output: {tuple(y.shape)} (B,C,H,W) = 1x8x{GRID_H}x{GRID_W}")
    print(f"Grid:   {GRID_W}x{GRID_H} = {GRID_W * GRID_H} cells (stride {STRIDE})")
    print(f"Params: {model.count_params():,}")

    return model


if __name__ == "__main__":
    test_model()

