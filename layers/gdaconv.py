import torch
import torch.nn as nn
import torch.nn.functional as F


class GDAConv(nn.Module):
    """
    Graph Dual Activate Convolution (GDAConv)

    A novel graph convolution module with dual activation functions (ReLU and GELU)
    that enhances spatial modeling by capturing both linear and complex patterns.
    """

    def __init__(self, in_channels, out_channels, dropout=0.1, type=True):
        super(GDAConv, self).__init__()
        self.type = type

        self.only_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * 2, kernel_size=(1, 1)),
                    nn.BatchNorm2d(out_channels * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Conv2d(out_channels * 2, out_channels, kernel_size=(1, 1)),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * 2, kernel_size=(1, 1)),
                    nn.BatchNorm2d(out_channels * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv2d(out_channels * 2, out_channels, kernel_size=(1, 1)),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ),
            ]
        )

        self.branch_weights = nn.Sequential(
            nn.Conv2d(in_channels, 2, kernel_size=(1, 1)), nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        Forward pass with adaptive dual-branch processing.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Processed tensor with same spatial dimensions
        """
        if not self.type:
            return self.only_relu(x)
        weights = self.branch_weights(x)
        relu_out = self.branches[0](x)
        gelu_out = self.branches[1](x)
        out = weights[:, 0:1] * relu_out + weights[:, 1:2] * gelu_out

        return out


class GDAConvExtended(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1, type=True):
        super(GDAConvExtended, self).__init__()
        self.type = type

        self.only_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 2, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels * 2, out_channels * 4, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_channels * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels * 4, out_channels * 2, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * 2, kernel_size=(1, 1)),
                    nn.BatchNorm2d(out_channels * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Conv2d(out_channels * 2, out_channels * 4, kernel_size=(1, 1)),
                    nn.BatchNorm2d(out_channels * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Conv2d(out_channels * 4, out_channels * 2, kernel_size=(1, 1)),
                    nn.BatchNorm2d(out_channels * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Conv2d(out_channels * 2, out_channels, kernel_size=(1, 1)),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * 2, kernel_size=(1, 1)),
                    nn.BatchNorm2d(out_channels * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv2d(out_channels * 2, out_channels * 4, kernel_size=(1, 1)),
                    nn.BatchNorm2d(out_channels * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv2d(out_channels * 4, out_channels * 2, kernel_size=(1, 1)),
                    nn.BatchNorm2d(out_channels * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv2d(out_channels * 2, out_channels, kernel_size=(1, 1)),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ),
            ]
        )

        self.branch_weights = nn.Sequential(
            nn.Conv2d(in_channels, 2, kernel_size=(1, 1)), nn.Softmax(dim=1)
        )

    def forward(self, x):
        if not self.type:
            return self.only_relu(x)
        weights = self.branch_weights(x)
        relu_out = self.branches[0](x)
        gelu_out = self.branches[1](x)
        out = weights[:, 0:1] * relu_out + weights[:, 1:2] * gelu_out

        return out


class FeatureProjectionLayer(nn.Module):
    """
    Feature Projection Layer

    Linear transformation layer with optional dual-branch architecture.
    Used for projecting features between different dimensional spaces.
    """

    def __init__(self, in_channels, out_channels, dropout=0.1, type=True):
        super(FeatureProjectionLayer, self).__init__()
        self.type = type

        self.only_relu = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_channels, out_channels),
                    nn.LayerNorm(out_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ),
                nn.Sequential(
                    nn.Linear(in_channels, out_channels),
                    nn.LayerNorm(out_channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ),
            ]
        )

        self.branch_weights = nn.Sequential(
            nn.Linear(in_channels, 2), nn.Softmax(dim=-1)
        )

    def forward(self, x):
        if not self.type:
            return self.only_relu(x)
        weights = self.branch_weights(x)

        relu_out = self.branches[0](x)
        gelu_out = self.branches[1](x)
        out = weights[:, :, 0:1] * relu_out + weights[:, :, 1:2] * gelu_out

        return out


class NodeTransformLayer(nn.Module):
    """
    Node Transformation Layer for Transfer Learning

    Maps features between source and target domains with different numbers of nodes.
    Essential for cross-city transfer where cities have different spatial resolutions.
    """

    def __init__(self, source_nodes, target_nodes, channel_size, dropout=0.1):
        super(NodeTransformLayer, self).__init__()
        self.source_nodes = source_nodes
        self.target_nodes = target_nodes
        self.channel_size = channel_size

        self.node_mapping = nn.Parameter(
            torch.empty(channel_size, source_nodes, target_nodes)
        )
        nn.init.xavier_uniform_(self.node_mapping)

        self.norm = nn.LayerNorm([channel_size, target_nodes])
        self.dropout = nn.Dropout(dropout)

        self.activation_transform = FeatureProjectionLayer(
            channel_size, channel_size, dropout=dropout, type=True
        )

    def forward(self, x):
        """
        Transform features from source domain nodes to target domain nodes.

        Args:
            x: Input tensor from source domain
               Shape: (batch_size, channel_size, source_nodes, 1)

        Returns:
            Transformed tensor for target domain
            Shape: (batch_size, channel_size, target_nodes, 1)
        """
        batch_size = x.size(0)
        x = x.squeeze(-1)

        output = torch.bmm(
            x.permute(1, 0, 2),
            self.node_mapping,
        )

        output = output.permute(1, 0, 2)

        output = output.permute(0, 2, 1)
        output = self.activation_transform(output)  # Dual activation gating
        output = output.permute(0, 2, 1)

        output = self.dropout(output)

        output = output.unsqueeze(-1)

        return output
