import torch.nn as nn

class EfficientSelfAttention(nn.Module):
    """
    Efficient Self-Attention mechanism for neural networks.

    This module implements a self-attention mechanism with a reduced sequence length, aiming to improve efficiency while maintaining the effectiveness of the attention process. It's particularly useful in scenarios where the standard self-attention mechanism might be computationally expensive.

    Attributes:
        heads (int): Number of attention heads.
        cn1 (nn.Conv2d): Convolution layer for dimensionality reduction.
        ln1 (nn.LayerNorm): Layer normalization.
        keyValueExtractor (nn.Linear): Linear layer to extract key and value.
        query (nn.Linear): Linear layer to extract query.
        smax (nn.Softmax): Softmax layer for attention calculation.
        finalLayer (nn.Linear): Final linear projection layer.

    Args:
        channels (int): Number of channels in the input.
        reduction_ratio (int): Ratio for reducing dimensionality.
        num_heads (int): Number of attention heads.
    """

    def __init__(self, channels, reduction_ratio, num_heads):
        super().__init__()
        assert channels % num_heads == 0, f"channels {channels} should be divided by num_heads {num_heads}."

        self.heads = num_heads
        
        # Reduction Parameters
        self.cn1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=reduction_ratio, stride=reduction_ratio)
        self.ln1 = nn.LayerNorm(channels)

        # Attention Parameters
        self.keyValueExtractor = nn.Linear(channels, channels * 2)
        self.query = nn.Linear(channels, channels)
        self.smax = nn.Softmax(dim=-1)
        self.finalLayer = nn.Linear(channels, channels)  # Projection Layer

    def forward(self, x, H, W):
        """
        Forward pass of the Efficient Self-Attention mechanism.

        Args:
            x (Tensor): Input tensor of shape (B, N, C), where B is the batch size, N is the number of queries (equal to H * W), and C is the number of channels.
            H (int): Height of the input when reshaped into a 2D format.
            W (int): Width of the input when reshaped into a 2D format.

        Returns:
            Tensor: Output tensor of shape (B, N, C) after applying self-attention.
        """
        B, N, C = x.shape

        # Dimensionality reduction
        x1 = x.clone().permute(0, 2, 1).reshape(B, C, H, W)
        x1 = self.cn1(x1)
        x1 = x1.reshape(B, C, -1).permute(0, 2, 1)
        x1 = self.ln1(x1)

        # Extracting key and value
        keyVal = self.keyValueExtractor(x1).reshape(B, -1, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        k, v = keyVal[0], keyVal[1]

        # Extracting query and calculating attention
        q = self.query(x).reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)
        dimHead = (C / self.heads) ** 0.5
        attention = self.smax(q @ k.transpose(-2, -1) / dimHead)
        attention = (attention @ v).transpose(1, 2).reshape(B, N, C)

        # Final linear projection
        x = self.finalLayer(attention)
        return x
