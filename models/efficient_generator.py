import torch.nn as nn
import math
from models.common import (
    Conv, Gencov, PSA, C2f, C2fCIB, SPPELAN,
    Concat, DilateBlock, CBAM, CA, SpatialAttention
)
from models.Repvit import RepViTBlock


class EfficientGenerator(nn.Module):
    """
    EfficientGenerator - A high-performance image generator optimized for 
    low-light enhancement or similar image-to-image translation tasks.

    Key features:
    - Efficient encoder-decoder architecture with skip connections
    - Hybrid attention mechanisms (spatial and channel)
    - Residual learning for better gradient flow 
    - Progressive feature fusion through multi-scale processing
    - Mixed-precision support for faster training

    Input: [B, 3, H, W]
    Output: [B, 3, H, W]
    """

    def __init__(
            self,
            in_channels=3,
            out_channels=3,
            base_channels=32,
            expansion_factor=2.0,
            depth_factor=1.0,
            attention_type='hybrid'
    ):
        """
        Initialize the EfficientGenerator model.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            base_channels (int): Base channel count for network width
            expansion_factor (float): Channel expansion factor 
            depth_factor (float): Controls the depth of the network
            attention_type (str): Type of attention to use
        """
        super(EfficientGenerator, self).__init__()

        self.depth = depth_factor
        self.base_ch = math.ceil(base_channels * self.depth)
        self.mid_ch = math.ceil(self.base_ch * expansion_factor)

        # Input processing with large receptive field
        self.input_conv = Gencov(in_channels, self.base_ch, k=7)

        # Encoder blocks
        # First block: no downsampling
        self.enc1 = self._make_encoder_block(
            self.base_ch, self.base_ch*2, downsample=False)
        self.enc2 = self._make_encoder_block(
            self.base_ch*2, self.base_ch*4, downsample=True)
        self.enc3 = self._make_encoder_block(
            self.base_ch*4, self.base_ch*8, downsample=True)
        self.enc4 = self._make_encoder_block(
            self.base_ch*8, self.base_ch*16, downsample=True)

        # Feature processing with multi-scale context aggregation
        self.context = nn.Sequential(
            SPPELAN(self.base_ch*16, self.base_ch*16, self.base_ch*8),
            PSA(self.base_ch*16, self.base_ch*16),
            DilateBlock(self.base_ch*16, num_heads=8)
        )

        # Attention module based on specified type
        if attention_type == 'channel':
            self.attention = CA(self.base_ch*16, self.base_ch*16)
        elif attention_type == 'spatial':
            self.attention = SpatialAttention()
        else:  # hybrid
            self.attention = CBAM(self.base_ch*16, self.base_ch*16)

        # Decoder blocks with skip connections
        self.dec4 = self._make_decoder_block(self.base_ch*16, self.base_ch*8)
        # doubled input for skip connection
        self.dec3 = self._make_decoder_block(self.base_ch*16, self.base_ch*4)
        # doubled input for skip connection
        self.dec2 = self._make_decoder_block(self.base_ch*8, self.base_ch*2)
        # doubled input for skip connection
        self.dec1 = self._make_decoder_block(self.base_ch*4, self.base_ch)

        # Final refinement
        self.output_conv = nn.Sequential(
            RepViTBlock(self.base_ch*2, self.base_ch*2, 3, 1, 1, 1),
            RepViTBlock(self.base_ch*2, self.base_ch, 1, 1, 0, 0),
            Gencov(self.base_ch, out_channels, k=3, bn=False, act=False),
            nn.Sigmoid()  # Output range [0, 1]
        )

        # For feature fusion
        self.concat = Concat()

        # Initialize weights
        self._initialize_weights()

    def _make_encoder_block(self, in_channels, out_channels, downsample=True):
        """Create an encoder block with optional downsampling."""
        stride = 2 if downsample else 1

        return nn.Sequential(
            RepViTBlock(in_channels, out_channels, 3, stride),
            RepViTBlock(out_channels, out_channels, 3, 1),
            # Efficient feature extraction and fusion
            C2f(out_channels, out_channels, n=1)
        )

    def _make_decoder_block(self, in_channels, out_channels):
        """Create a decoder block with upsampling."""
        return nn.Sequential(
            # Upsample features
            nn.Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=True
            ),
            # Process upsampled features
            C2fCIB(in_channels, out_channels, n=1, shortcut=True, lk=True),
            # Additional processing
            RepViTBlock(out_channels, out_channels, 3, 1, 0, 0)
        )

    def _initialize_weights(self):
        """Initialize weights for better training convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """Forward pass through the generator."""
        # Input size validation for automatic size handling
        _, _, h, w = x.shape

        # Encoder path - progressive feature extraction
        x0 = self.input_conv(x)
        x1 = self.enc1(x0)         # No downsample
        x2 = self.enc2(x1)         # 1/2 resolution
        x3 = self.enc3(x2)         # 1/4 resolution
        x4 = self.enc4(x3)         # 1/8 resolution

        # Feature processing at bottleneck
        features = self.context(x4)
        features = self.attention(features)

        # Decoder path with skip connections
        d4 = self.dec4(features)                      # 1/4 resolution
        d3 = self.dec3(self.concat([d4, x3]))         # 1/2 resolution
        d2 = self.dec2(self.concat([d3, x2]))         # 1/1 resolution
        d1 = self.dec1(self.concat([d2, x1]))         # 1/1 resolution

        # Final prediction
        out = self.output_conv(self.concat([d1, x0]))

        return out


class LightweightGenerator(nn.Module):
    """
    LightweightGenerator - A resource-efficient generator with good performance 
    and reduced computational requirements.

    Designed for scenarios with limited computational resources or real-time 
    applications where efficiency is crucial.

    Input: [B, 3, H, W]
    Output: [B, 3, H, W]
    """

    def __init__(
            self,
            in_channels=3,
            out_channels=3,
            base_channels=16,
            depth_factor=1.0
    ):
        """
        Initialize the LightweightGenerator model.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            base_channels (int): Base channel count for scaling network width
            depth_factor (float): Controls the depth of the network
        """
        super(LightweightGenerator, self).__init__()

        self.depth = depth_factor
        self.base_ch = math.ceil(base_channels * self.depth)

        # Initial feature extraction
        self.input_conv = Gencov(in_channels, self.base_ch, k=3)

        # Encoder blocks with reduced complexity
        self.enc1 = nn.Sequential(
            Gencov(self.base_ch, self.base_ch*2, k=3, s=2),
            Gencov(self.base_ch*2, self.base_ch*2, k=3, s=1)
        )

        self.enc2 = nn.Sequential(
            Gencov(self.base_ch*2, self.base_ch*4, k=3, s=2),
            Gencov(self.base_ch*4, self.base_ch*4, k=3, s=1)
        )

        self.enc3 = nn.Sequential(
            Gencov(self.base_ch*4, self.base_ch*8, k=3, s=2),
            Gencov(self.base_ch*8, self.base_ch*8, k=3, s=1)
        )

        # Lightweight attention at bottleneck
        self.bottleneck = nn.Sequential(
            Gencov(self.base_ch*8, self.base_ch*8, k=3, s=1),
            # Channel attention is computationally efficient
            CA(self.base_ch*8, self.base_ch*8),
            Gencov(self.base_ch*8, self.base_ch*8, k=3, s=1)
        )

        # Decoder blocks with efficient upsampling
        self.dec3 = nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=True
            ),
            Gencov(self.base_ch*16, self.base_ch*4, k=3, s=1)
        )

        self.dec2 = nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=True
            ),
            Gencov(self.base_ch*8, self.base_ch*2, k=3, s=1)
        )

        self.dec1 = nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=True
            ),
            Gencov(self.base_ch*4, self.base_ch, k=3, s=1)
        )

        # Final layer
        self.output_conv = nn.Sequential(
            Gencov(self.base_ch*2, self.base_ch, k=3, s=1),
            Gencov(self.base_ch, out_channels, k=3, bn=False, act=False),
            nn.Sigmoid()  # Output range [0, 1]
        )

        # For feature fusion
        self.concat = Concat()

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for better training convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """Forward pass through the generator."""
        # Input processing
        x0 = self.input_conv(x)

        # Encoder path
        x1 = self.enc1(x0)  # 1/2 resolution
        x2 = self.enc2(x1)  # 1/4 resolution
        x3 = self.enc3(x2)  # 1/8 resolution

        # Process bottleneck features
        features = self.bottleneck(x3)

        # Decoder path with skip connections
        d3 = self.dec3(self.concat([features, x3]))
        d2 = self.dec2(self.concat([d3, x2]))
        d1 = self.dec1(self.concat([d2, x1]))

        # Final output
        out = self.output_conv(self.concat([d1, x0]))

        return out


class OptimizedGenerator(nn.Module):
    """
    OptimizedGenerator - A balanced generator network that offers excellent performance
    with moderate computational requirements.

    Features:
    - Efficient encoder-decoder architecture with skip connections
    - Strategic use of attention mechanisms
    - Balanced depth and width for optimal performance/efficiency trade-off
    - Specialized for image-to-image translation tasks

    Input: [B, in_channels, H, W]
    Output: [B, out_channels, H, W]
    """

    def __init__(
            self,
            in_channels=3,
            out_channels=3,
            base_channels=24,
            attention_type='channel',
            depth_factor=1.0
    ):
        """
        Initialize the OptimizedGenerator model.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            base_channels (int): Base channel count for network width
            attention_type (str): Type of attention ('channel', 'spatial', or 'hybrid')
            depth_factor (float): Controls the depth of the network
        """
        super(OptimizedGenerator, self).__init__()

        self.depth = depth_factor
        self.base_ch = math.ceil(base_channels * self.depth)

        # Initial feature extraction
        self.input_conv = Gencov(in_channels, self.base_ch, k=5)

        # Encoder blocks
        self.enc1 = nn.Sequential(
            Gencov(self.base_ch, self.base_ch*2, k=3, s=2),
            Gencov(self.base_ch*2, self.base_ch*2, k=3, s=1)
        )

        self.enc2 = nn.Sequential(
            Gencov(self.base_ch*2, self.base_ch*4, k=3, s=2),
            Gencov(self.base_ch*4, self.base_ch*4, k=3, s=1)
        )

        self.enc3 = nn.Sequential(
            Gencov(self.base_ch*4, self.base_ch*8, k=3, s=2),
            Gencov(self.base_ch*8, self.base_ch*8, k=3, s=1)
        )

        # Bottleneck with attention
        if attention_type == 'channel':
            attention = CA(self.base_ch*8, self.base_ch*8)
        elif attention_type == 'spatial':
            attention = SpatialAttention()
        else:  # hybrid
            attention = CBAM(self.base_ch*8, self.base_ch*8)

        self.bottleneck = nn.Sequential(
            Gencov(self.base_ch*8, self.base_ch*8, k=3, s=1),
            SPPELAN(self.base_ch*8, self.base_ch*8, self.base_ch*4),
            attention,
            Gencov(self.base_ch*8, self.base_ch*8, k=3, s=1)
        )

        # Decoder blocks with skip connections
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            Gencov(self.base_ch*16, self.base_ch*4, k=3, s=1)
        )

        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            Gencov(self.base_ch*8, self.base_ch*2, k=3, s=1)
        )

        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            Gencov(self.base_ch*4, self.base_ch, k=3, s=1)
        )

        # Final output layers with refinement
        self.output_conv = nn.Sequential(
            Gencov(self.base_ch*2, self.base_ch, k=3, s=1),
            Gencov(self.base_ch, self.base_ch, k=3, s=1),
            Gencov(self.base_ch, out_channels, k=3, bn=False, act=False),
            nn.Sigmoid()  # Output range [0, 1]
        )

        # For feature fusion
        self.concat = Concat()

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through the generator.

        Args:
            x (torch.Tensor): Input tensor [B, in_channels, H, W]

        Returns:
            torch.Tensor: Output tensor [B, out_channels, H, W]
        """
        # Input processing
        x0 = self.input_conv(x)

        # Encoder pathway with skip connections
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        # Bottleneck processing
        x_bottleneck = self.bottleneck(x3)

        # Decoder pathway with skip connections
        d3 = self.dec3(self.concat([x_bottleneck, x2]))
        d2 = self.dec2(self.concat([d3, x1]))
        d1 = self.dec1(self.concat([d2, x0]))

        # Final output with skip connection to input features
        output = self.output_conv(self.concat([d1, x0]))

        return output
