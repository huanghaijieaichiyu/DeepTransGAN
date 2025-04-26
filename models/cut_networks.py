import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_mode import BaseNetwork
from models.common import Gencov, SPPELAN, Concat, PSA
from models.Repvit import RepViTBlock
import math


class CUTGenerator(BaseNetwork):
    """
    CUT Generator based on ResNet-style architecture.
    This generator is designed for the Contrastive Unpaired Translation (CUT) 
    approach.
    
    Input: [B, 3, 256, 256]
    Output: [B, 3, 256, 256]
    """

    def __init__(self, depth=1, weight=1):
        super().__init__()
        self.depth = depth
        self.weight = weight
        base_ch = math.ceil(8 * self.depth)

        # Initial feature extraction
        self.conv1 = Gencov(3, base_ch)

        # Downsampling blocks
        self.down1 = self._make_downblock(base_ch, base_ch * 2)
        self.down2 = self._make_downblock(base_ch * 2, base_ch * 4)
        self.down3 = self._make_downblock(base_ch * 4, base_ch * 8)

        # Bottleneck with attention
        self.bottleneck = nn.Sequential(SPPELAN(base_ch * 8, base_ch * 8, base_ch * 4), PSA(base_ch * 8, base_ch * 8),
                                        RepViTBlock(base_ch * 8, base_ch * 8, 1, 1, 0, 0))

        # Upsampling blocks
        self.up3 = self._make_upblock(base_ch * 8, base_ch * 4)
        # Doubled due to skip connection
        self.up2 = self._make_upblock(base_ch * 8, base_ch * 2)
        # Doubled due to skip connection
        self.up1 = self._make_upblock(base_ch * 4, base_ch)

        # Final output layer with doubled channels due to skip connection
        self.final = Gencov(base_ch * 2, 3, act=False, bn=False)
        self.activation = nn.Sigmoid()
        self.concat = Concat()

    def _make_downblock(self, in_ch, out_ch):
        return nn.Sequential(RepViTBlock(in_ch, out_ch, math.ceil(self.weight), 2),
                             RepViTBlock(out_ch, out_ch, 1, 1, 0, 0))

    def _make_upblock(self, in_ch, out_ch):
        return nn.Sequential(RepViTBlock(in_ch, in_ch, math.ceil(self.weight), 1),
                             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), Gencov(in_ch, out_ch))

    def forward(self, x):
        # Initial feature extraction
        x1 = self.conv1(x)  # [B, base_ch, 256, 256]

        # Encoder
        x2 = self.down1(x1)  # [B, base_ch*2, 128, 128]
        x3 = self.down2(x2)  # [B, base_ch*4, 64, 64]
        x4 = self.down3(x3)  # [B, base_ch*8, 32, 32]

        # Bottleneck
        x4 = self.bottleneck(x4)  # [B, base_ch*8, 32, 32]

        # Decoder with skip connections
        x = self.up3(x4)  # [B, base_ch*4, 64, 64]
        x = self.up2(self.concat([x, x3]))  # [B, base_ch*2, 128, 128]
        x = self.up1(self.concat([x, x2]))  # [B, base_ch, 256, 256]
        x = self.final(self.concat([x, x1]))  # [B, 3, 256, 256]

        return self.activation(x)


class PatchSampleF(BaseNetwork):
    """
    PatchSample Network: Creates feature patches for the PatchNCE loss.
    This network samples patches from different layers of the network and 
    projects them to a common space.
    """

    def __init__(self, use_mlp=True, init_type='normal', nc=256, patch_size=16):
        super().__init__()
        self.use_mlp = use_mlp
        self.nc = nc  # number of output channels
        self.patch_size = patch_size  # size of patches to sample

        # Initialize with default MLP layers that will be used if no features are provided
        self.mlp_init = True

        # Create default MLP layers with common input dimensions
        # These will be replaced when actual features are passed through forward()
        input_dims = [64, 128, 256, 512]  # Common feature dimensions from generators
        self.mlp_layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)) for dim in input_dims])

        # Create a dummy parameter to ensure the optimizer always has something to optimize
        self.dummy_param = nn.Parameter(torch.zeros(1, requires_grad=True))

        # Store device information
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def create_mlp_layers(self, feats):
        """Create or update the MLP layers based on the feature dimensions"""
        device = feats[0].device

        # Ensure we're on the same device as the features
        if next(self.parameters()).device != device:
            self.to(device)

        if self.mlp_layers is None or len(self.mlp_layers) != len(feats):
            self.mlp_layers = nn.ModuleList()
            for feat_id, feat in enumerate(feats):
                input_nc = feat.shape[1]
                mlp = nn.Sequential(nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)).to(device)
                self.mlp_layers.append(mlp)
        else:
            # Update existing layers if dimensions don't match
            for feat_id, feat in enumerate(feats):
                input_nc = feat.shape[1]
                if feat_id < len(self.mlp_layers):
                    # Check if the first linear layer has the right input dimension
                    if self.mlp_layers[feat_id][0].in_features != input_nc:
                        # Replace the MLP with the right dimensions
                        self.mlp_layers[feat_id] = nn.Sequential(nn.Linear(input_nc, self.nc), nn.ReLU(),
                                                                 nn.Linear(self.nc, self.nc)).to(device)

        self.mlp_init = True

        # Make sure mlp_layers are on the same device as feats
        for layer in self.mlp_layers:
            if next(layer.parameters()).device != device:
                layer.to(device)

    def forward(self, feats, num_patches=64, patch_ids=None):
        """
        Extract and project patches from each feature map.
        
        Args:
            feats: List of feature maps from different layers of the generator
            num_patches: Number of patches to sample per layer
            patch_ids: Optional predefined patch locations. If None, patches are
                       sampled randomly.
        
        Returns:
            patch_feats: List of tensors containing the projected patches
        """
        # Ensure we're on the same device as the input
        device = feats[0].device
        if next(self.parameters()).device != device:
            self.to(device)

        # Always update MLP layers to match the current feature dimensions
        self.create_mlp_layers(feats)

        return_ids = []
        return_feats = []

        for feat_id, feat in enumerate(feats):
            B, C, H, W = feat.shape

            # Reshape the feature map to [B, C, H*W]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)

            # Sample patches
            if patch_ids is not None:
                # Use provided patch IDs
                patch_id = patch_ids[feat_id]
            else:
                # Sample random patches
                patch_id = torch.randperm(feat_reshape.shape[1], device=device)
                patch_id = patch_id[:min(num_patches, feat_reshape.shape[1])]

            # Sample patches
            # reshape to (B*num_patches, C)
            x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)

            # Project patches
            if self.use_mlp:
                # Ensure the layer is on the same device
                if next(self.mlp_layers[feat_id].parameters()).device != device:
                    self.mlp_layers[feat_id].to(device)
                x_sample = self.mlp_layers[feat_id](x_sample)

            # Normalize the feature
            x_sample = F.normalize(x_sample, dim=1)

            # Reshape back to [B, num_patches, C]
            x_sample = x_sample.view(B, -1, self.nc)

            return_ids.append(patch_id)
            return_feats.append(x_sample)

        return return_feats, return_ids


class PatchNCELoss(nn.Module):
    """
    Contrastive Learning (NCE) loss over image patches.
    This is the core component of the CUT approach.
    """

    def __init__(self, batch_size, nce_T=0.07):
        super().__init__()
        self.batch_size = batch_size
        self.nce_T = nce_T
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.bool

    def forward(self, query, key, mask=None):
        """
        Query: Patches from the output image [B, num_patches, C]
        Key: Patches from the input image [B, num_patches, C]
        """
        batch_size, num_patches, dim = query.shape

        # Compute similarity matrix
        # [B, num_patches, C] x [B, C, num_patches] -> [B, num_patches, num_patches]
        l_pos = torch.bmm(query, key.transpose(1, 2))

        # Scale by temperature
        l_pos = l_pos / self.nce_T

        # Create labels: diagonal elements are positives
        diagonal = torch.eye(num_patches, device=query.device)
        diagonal = diagonal.repeat(batch_size, 1, 1)

        # Use diagonal elements as positive pairs
        labels = torch.arange(num_patches, device=query.device).repeat(batch_size)

        # Compute cross entropy loss
        loss = self.cross_entropy_loss(l_pos.view(-1, num_patches), labels)

        # Apply mask if provided
        if mask is not None:
            mask = mask.repeat(1, num_patches)
            loss = loss * mask.float()
            loss = loss.sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()

        return loss


class FastCUTGenerator(CUTGenerator):
    """
    FastCUT Generator - a simplified version of CUT with fewer parameters
    and faster training time. Uses the same architecture as CUT but with
    smaller feature dimensions.
    """

    def __init__(self, depth=0.75, weight=0.75):
        # Reduce depth and weight for faster training
        super().__init__(depth=depth, weight=weight)
