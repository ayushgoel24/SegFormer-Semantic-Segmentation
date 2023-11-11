import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPDecoder(nn.Module):
    """
    MLP Decoder for processing multiple feature maps.

    This decoder takes multiple feature maps with varying dimensions, processes them through individual MLP layers, and combines them into a single output feature map. It's commonly used in tasks like image segmentation or object detection.

    Attributes:
        mlp1 (nn.ModuleList): List of 1x1 convolutional layers to bring feature maps to a common channel dimension.
        mlp2 (nn.Conv2d): 1x1 convolutional layer to combine all feature maps.
        bn (nn.BatchNorm2d): Batch normalization layer.
        relu (nn.ReLU): ReLU activation function.
        mlp3 (nn.Conv2d): Final 1x1 convolutional layer to produce the output with the desired number of classes.

    Args:
        in_channels (list of int): Number of input channels for each layer.
        embed_channels (int): Number of embedding channels for standardization.
        out_dims (tuple): Dimensions of the output feature map.
        num_classes (int): Number of output classes.
    """

    def __init__(self, in_channels, embed_channels, out_dims, num_classes):
        super().__init__()
        self.outDim = out_dims
        self.mlp1 = nn.ModuleList([nn.Conv2d(in_channels[i], embed_channels, kernel_size=1) for i in range(len(in_channels))])
        self.mlp2 = nn.Conv2d(len(in_channels) * embed_channels, embed_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(embed_channels)
        self.relu = nn.ReLU(inplace=True)
        self.mlp3 = nn.Conv2d(embed_channels, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the MLP Decoder.

        Args:
            x (list of Tensors): List of input tensors with varying dimensions.

        Returns:
            Tensor: Output tensor with standardized dimensions and channels.
        """
        # Process each feature map to common channel dimensions and interpolate to the same size
        for i in range(len(x)):
            x[i] = self.mlp1[i](x[i])
            x[i] = F.interpolate(x[i], size=self.outDim, mode='bilinear', align_corners=False)

        # Concatenate all feature maps along the channel dimension
        x = torch.cat(x, dim=1)

        # Further process the combined feature map
        x = self.mlp2(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.mlp3(x)

        return x
