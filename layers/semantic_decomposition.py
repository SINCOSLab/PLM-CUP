import torch
import torch.nn as nn
import numpy as np


class SemanticDecomposition(nn.Module):
    """
    Semantic Coefficient Decomposition (SCD) module

    Base module for trend and seasonal decomposition using dual-branch architecture.
    Processes urban flow data to extract semantic patterns.
    """

    def __init__(
        self,
        input_size,
        theta_size,
        basis_function,
        share_weights=False,
        activation="relu",
        num_nodes=None,
        small_value_threshold=1.0,
    ):
        super().__init__()

        self.input_size = input_size
        self.theta_size = theta_size
        self.basis_function = basis_function
        self.share_weights = share_weights
        self.small_value_threshold = small_value_threshold

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "selu":
            self.activation = nn.SELU()
        elif activation == "gelu":
            self.activation = nn.GELU()

        self.main_branch = nn.Sequential(
            nn.Linear(input_size, input_size * 2),
            self.activation,
            nn.Linear(input_size * 2, input_size * 4),
            self.activation,
            nn.Linear(input_size * 4, input_size * 2),
            self.activation,
            nn.Linear(input_size * 2, input_size),
            self.activation,
            nn.Linear(input_size, theta_size),
        )

        self.small_value_branch = nn.Sequential(
            nn.Linear(input_size, input_size * 2),
            nn.GELU(),
            nn.Linear(input_size * 2, input_size * 4),
            nn.GELU(),
            nn.Linear(input_size * 4, input_size * 2),
            nn.GELU(),
            nn.Linear(input_size * 2, input_size),
            nn.GELU(),
            nn.Linear(input_size, theta_size),
        )

        self.branch_weights = nn.Sequential(
            nn.Linear(input_size, 2), nn.Softmax(dim=-1)
        )

    def forward(self, x):
        input_x = x

        x = x.squeeze(-1).permute(0, 2, 1)

        weights = self.branch_weights(x)

        theta_main = self.main_branch(x)

        theta_small = self.small_value_branch(x)

        theta = weights[..., 0:1] * theta_main + weights[..., 1:2] * theta_small

        output = self.basis_function(theta)

        output = output.permute(0, 2, 1).unsqueeze(-1)

        return output, input_x - output


class TrendModule(nn.Module):
    """
    Trend Decomposition Module

    Extracts multi-scale trend patterns using polynomial basis functions.
    Part of the Semantic Bridge Encoder in PLM-CUP.
    """

    def __init__(self, input_size, degree, num_nodes, small_value_threshold=1.0):
        super().__init__()
        self.degree = degree
        self.theta_size = degree + 1
        self.small_value_threshold = small_value_threshold
        self.input_size = input_size
        self.block1 = SemanticDecomposition(
            input_size=input_size,
            theta_size=self.theta_size,
            basis_function=self.polynomial_basis,
            activation="relu",
            num_nodes=num_nodes,
            small_value_threshold=small_value_threshold,
        )
        self.block2 = SemanticDecomposition(
            input_size=input_size,
            theta_size=self.theta_size,
            basis_function=self.polynomial_basis,
            activation="relu",
            num_nodes=num_nodes,
            small_value_threshold=small_value_threshold,
        )
        self.block3 = SemanticDecomposition(
            input_size=input_size,
            theta_size=self.theta_size,
            basis_function=self.polynomial_basis,
            activation="relu",
            num_nodes=num_nodes,
            small_value_threshold=small_value_threshold,
        )

    def polynomial_basis(self, theta):
        batch_size, num_nodes, _ = theta.shape

        t = torch.linspace(
            0, 1, self.input_size, device=theta.device, dtype=theta.dtype
        )

        powers = torch.arange(self.degree + 1, device=theta.device, dtype=theta.dtype)
        t_expanded = t.unsqueeze(0)
        powers_expanded = powers.unsqueeze(-1)

        basis = (t_expanded ** powers_expanded.unsqueeze(1)).transpose(0, 1)

        trend = torch.matmul(theta, basis)

        return trend

    def forward(self, x):
        trend_output1, trend_input1 = self.block1(x)
        trend_output2, trend_input2 = self.block2(trend_input1 + x)
        trend_output3, trend_input3 = self.block3(trend_input2 + x)
        trend = trend_output1 + trend_output2 + trend_output3
        return trend


class SeasonalModule(nn.Module):
    """
    Seasonal Decomposition Module

    Isolates cyclic patterns using Fourier basis functions.
    Captures periodic behaviors in urban flow data.
    """

    def __init__(self, input_size, num_harmonics, num_nodes, small_value_threshold=1.0):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.theta_size = 2 * num_harmonics
        self.small_value_threshold = small_value_threshold
        self.input_size = input_size

        self.block1 = SemanticDecomposition(
            input_size=input_size,
            theta_size=self.theta_size,
            basis_function=self.fourier_basis,
            activation="relu",
            num_nodes=num_nodes,
            small_value_threshold=small_value_threshold,
        )
        self.block2 = SemanticDecomposition(
            input_size=input_size,
            theta_size=self.theta_size,
            basis_function=self.fourier_basis,
            activation="relu",
            num_nodes=num_nodes,
            small_value_threshold=small_value_threshold,
        )
        self.block3 = SemanticDecomposition(
            input_size=input_size,
            theta_size=self.theta_size,
            basis_function=self.fourier_basis,
            activation="relu",
            num_nodes=num_nodes,
            small_value_threshold=small_value_threshold,
        )

    def fourier_basis(self, theta):
        batch_size, num_nodes, _ = theta.shape

        t = torch.linspace(
            0, 2 * np.pi, self.input_size, device=theta.device, dtype=theta.dtype
        )

        harmonics = torch.arange(
            1, self.num_harmonics + 1, device=theta.device, dtype=theta.dtype
        )
        t_expanded = t.unsqueeze(0)
        harmonics_expanded = harmonics.unsqueeze(-1)

        angles = harmonics_expanded * t_expanded
        sin_basis = torch.sin(angles)
        cos_basis = torch.cos(angles)

        basis = torch.cat([sin_basis, cos_basis], dim=0).transpose(0, 1)

        seasonal = torch.matmul(theta, basis.transpose(0, 1))

        return seasonal

    def forward(self, x):
        seasonal_output1, seasonal_input1 = self.block1(x)
        seasonal_output2, seasonal_input2 = self.block2(seasonal_input1 + x)
        seasonal_output3, seasonal_input3 = self.block3(seasonal_input2 + x)
        seasonal = seasonal_output1 + seasonal_output2 + seasonal_output3
        return seasonal
