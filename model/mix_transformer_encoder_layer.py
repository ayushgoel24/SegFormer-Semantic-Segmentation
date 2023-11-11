from .efficient_self_attention import EfficientSelfAttention
from .mix_ffn import MixFFN
from .overlap_patch_merging import OverlapPatchMerging

import torch.nn as nn

class MixTransformerEncoderLayer(nn.Module):
    """
    MixTransformer Encoder Layer for vision transformer architectures.

    This layer combines overlapping patch merging, efficient self-attention, and mixed feedforward networks (MixFFN) to process input images. It's designed to capture both local and global dependencies in image data.

    Attributes:
        patchMerge (OverlapPatchMerging): Module to merge image patches.
        _attn (nn.ModuleList): List of EfficientSelfAttention modules.
        _ffn (nn.ModuleList): List of MixFFN modules.
        _lNorm (nn.ModuleList): List of LayerNorm modules.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        patch_size (int): Size of each patch.
        stride (int): Stride for patch merging.
        padding (int): Padding for patch merging.
        n_layers (int): Number of layers for attention and feedforward networks.
        reduction_ratio (int): Reduction ratio for efficient self-attention.
        num_heads (int): Number of heads for self-attention.
        expansion_factor (int): Expansion factor for MixFFN.
    """

    def __init__(self, in_channels, out_channels, patch_size, stride, padding, 
                 n_layers, reduction_ratio, num_heads, expansion_factor):
        super().__init__()
        self.patchMerge = OverlapPatchMerging(in_channels, out_channels, patch_size, stride, padding)
        self._attn = nn.ModuleList([EfficientSelfAttention(out_channels, reduction_ratio, num_heads) for _ in range(n_layers)])
        self._ffn = nn.ModuleList([MixFFN(out_channels, expansion_factor) for _ in range(n_layers)])
        self._lNorm = nn.ModuleList([nn.LayerNorm(out_channels) for _ in range(n_layers)])

    def forward(self, x):
        """
        Forward pass of the MixTransformer Encoder Layer.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W), where B is the batch size, C is the number of channels, and H and W are the height and width.

        Returns:
            Tensor: Output tensor of shape (B, C, H, W) after processing through the encoder layer.
        """
        B, C, H, W = x.shape
        x, H, W = self.patchMerge(x)  # Shape: B, N, EmbedDim (C)

        for i in range(len(self._attn)):
            # Apply self-attention and feedforward networks in sequence
            x = x + self._attn[i](x, H, W)  # Shape: B, N, C
            x = x + self._ffn[i](x, H, W)   # Shape: B, N, C
            x = self._lNorm[i](x)           # Shape: B, N, C

        # Reshape and permute to match original input dimensions
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # Shape: B, C, H, W
        return x
