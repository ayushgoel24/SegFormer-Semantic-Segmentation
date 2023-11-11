import torch.nn as nn

class OverlapPatchMerging(nn.Module):
    """
    OverlapPatchMerging module for merging image patches.

    This module applies a convolution operation to merge overlapping patches from an image, reducing its dimensionality. It is typically used in vision transformer architectures to process image data into a suitable form for the transformer encoder.

    Attributes:
        conv_layer (nn.Conv2d): Convolutional layer to merge patches.
        norm_layer (nn.LayerNorm): Layer normalization.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        patch_size (int): Size of the patch.
        stride (int): Stride of the convolution.
        padding (int): Padding added to patches.
    """

    def __init__(self, in_channels, out_channels, patch_size, stride, padding):
        super().__init__()
        # Convolutional layer to merge patches
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=stride, padding=padding)
        # Layer normalization
        self.norm_layer = nn.LayerNorm(out_channels)

    def forward(self, patches):
        """
        Forward pass for merging patches.

        Applies a convolution operation to the input patches and then normalizes the result.

        Args:
            patches (Tensor): Input tensor of shape (B, C, H, W), where B is the batch size, C is the number of channels, and H, W are the height and width of the patches.

        Returns:
            Tuple[Tensor, int, int]: A tuple containing the processed tensor of shape (B, N, EmbedDim), and the height (H) and width (W) of the processed tensor. Here, N is the product of H and W.
        """
        # Apply convolution to the patches
        x = self.conv_layer(patches)
        _, _, H, W = x.shape
        # Flatten and transpose the result for layer normalization
        x = x.flatten(2).transpose(1, 2)
        x = self.norm_layer(x)

        return x, H, W
