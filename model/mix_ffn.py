import torch.nn as nn

class MixFFN(nn.Module):
    """
    Mixed Feedforward Network (MixFFN) module.

    This module is a part of neural network architectures, combining multi-layer perceptron (MLP) layers and a depthwise convolutional layer. It's designed to process inputs with a mix of linear transformations and localized convolutions.

    Attributes:
        mlp1 (nn.Linear): First MLP layer for channel expansion.
        depthwise (nn.Conv2d): Depthwise convolutional layer for spatial feature extraction.
        gelu (nn.GELU): GELU activation function.
        mlp2 (nn.Linear): Second MLP layer to project back to original channel size.

    Args:
        channels (int): Number of channels in the input.
        expansion_factor (int): Factor to expand the channels in the first MLP layer.
    """

    def __init__(self, channels, expansion_factor):
        super().__init__()
        expanded_channels = channels * expansion_factor

        # MLP Layer for channel expansion
        self.mlp1 = nn.Linear(channels, expanded_channels)

        # Depthwise convolutional layer for spatial feature extraction
        self.depthwise = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, padding='same', groups=channels)

        # GELU activation function
        self.gelu = nn.GELU()

        # MLP layer to project back to the original number of channels
        self.mlp2 = nn.Linear(expanded_channels, channels)

    def forward(self, x, H, W):
        """
        Forward pass of the MixFFN module.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W), where B is the batch size, C is the number of channels, and H and W are the height and width.
            H (int): Height of the input.
            W (int): Width of the input.

        Returns:
            Tensor: Output tensor of shape (B, C, H, W) after processing through MixFFN.
        """
        # Apply the first MLP layer
        x = self.mlp1(x)
        B, N, C = x.shape

        # Reshape for depthwise convolution
        x = x.transpose(1, 2).view(B, C, H, W)

        # Apply depthwise convolution followed by GELU activation
        x = self.gelu(self.depthwise(x).flatten(2).transpose(1, 2))

        # Apply the second MLP layer to project back to original channel size
        x = self.mlp2(x)  # Shape: B, N, C
        return x
