import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainShiftRegularizer(nn.Module):
    """
    Domain Shift Regularizer for PLM-CUP.

    Implements the KL-divergence regularization term from the PLM-CUP paper
    to minimize domain shift between urban flow data and natural language.
    Supports multiple methods: sliced MMD, learned projection, kernel alignment,
    and cosine similarity.
    """

    def __init__(
        self,
        method="sliced_mmd",
        phi=0.01,
        plm_channel=768,
        projection_dim=32,
        n_projections=50,
        device="cuda",
    ):
        super().__init__()
        self.method = method
        self.phi = phi
        self.plm_channel = plm_channel
        self.projection_dim = projection_dim
        self.n_projections = n_projections
        self.device = device

        # Learned projection layer (optional)
        if method == "learned_projection":
            self.plm_projector = nn.Linear(plm_channel, projection_dim)
            self.y_projector = nn.Linear(1, projection_dim)
            self._init_projectors()

        # Add projection layer for simplified cosine similarity method
        if method == "cosine_similarity":
            self.feature_projector = nn.Sequential(
                nn.Linear(plm_channel, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1),
            )
            with torch.no_grad():
                self.feature_projector[-1].weight.data *= 0.01
                self.feature_projector[-1].bias.data.zero_()

    def _init_projectors(self):
        """Initialize projection layers"""
        nn.init.xavier_uniform_(self.plm_projector.weight, gain=0.1)
        nn.init.zeros_(self.plm_projector.bias)
        nn.init.xavier_uniform_(self.y_projector.weight)
        nn.init.zeros_(self.y_projector.bias)

    def _prepare_batch_y(self, batch_y):
        """
        Unified processing of batch_y format.
        Supports multiple input formats with automatic format detection.
        """
        if batch_y.dim() == 2:
            batch_y = batch_y.unsqueeze(1)
        elif batch_y.dim() == 3:
            if batch_y.shape[-1] < batch_y.shape[1] and batch_y.shape[-1] <= 12:
                batch_y = batch_y.transpose(1, 2)
        elif batch_y.dim() == 4:
            batch_y = batch_y.squeeze(-1)

        return batch_y

    def compute_sliced_mmd(self, plm_features, batch_y):
        """
        Sliced MMD: Compute MMD through multiple random projections.
        Handles both single-step and multi-step predictions uniformly.

        Args:
            plm_features: (batch_size, num_nodes, plm_channel)
            batch_y: Any format, will be unified
        """
        # Unified processing of batch_y format
        batch_y = self._prepare_batch_y(batch_y)
        batch_size, steps, num_nodes = batch_y.shape

        total_mmd_across_steps = 0.0

        for step in range(steps):
            y_step = batch_y[:, step, :]

            plm_flat = plm_features.reshape(-1, self.plm_channel)
            y_flat = y_step.reshape(-1, 1)

            projections = torch.randn(
                self.n_projections, self.plm_channel, device=plm_flat.device
            )
            projections = F.normalize(projections, p=2, dim=1)

            step_mmd = 0.0

            for i in range(self.n_projections):
                proj_direction = projections[i]
                plm_projected = plm_flat @ proj_direction

                plm_projected_expanded = plm_projected.unsqueeze(-1)
                plm_norm = plm_projected_expanded / (
                    torch.norm(plm_projected_expanded) + 1e-8
                )
                y_norm = y_flat / (torch.norm(y_flat) + 1e-8)

                mmd = self._compute_1d_mmd(plm_norm, y_norm)
                step_mmd += mmd

            total_mmd_across_steps += step_mmd / self.n_projections

        return total_mmd_across_steps / steps

    def _compute_1d_mmd(self, x, y):
        """Compute MMD for 1D data"""
        joint = torch.cat([x, y], dim=-1)

        perm = torch.randperm(x.shape[0])
        independent = torch.cat([x, y[perm]], dim=-1)

        return self._mmd_rbf(joint, independent)

    def compute_learned_projection_mmd(self, plm_features, batch_y):
        """
        Use learned projection layers to map features to common space.
        Handles both single-step and multi-step predictions uniformly.
        """
        # Unified processing of batch_y format
        batch_y = self._prepare_batch_y(batch_y)
        batch_size, steps, num_nodes = batch_y.shape

        # Calculate MMD separately for each time step, then average
        total_mmd = 0.0

        for step in range(steps):
            y_step = batch_y[:, step, :]

            plm_flat = plm_features.view(-1, self.plm_channel)
            y_flat = y_step.view(-1, 1)

            plm_proj = self.plm_projector(plm_flat)
            y_proj = self.y_projector(y_flat)

            mmd = self._mmd_rbf(plm_proj, y_proj)
            total_mmd += mmd

        # Return average MMD across all steps
        return total_mmd / steps

    def compute_kernel_alignment(self, plm_features, batch_y):
        """
        Use Kernel Alignment method.
        Handles both single-step and multi-step predictions uniformly.
        """
        # Unified processing of batch_y format
        batch_y = self._prepare_batch_y(batch_y)
        batch_size, steps, num_nodes = batch_y.shape

        # Calculate kernel alignment separately for each time step, then average
        total_distance = 0.0

        for step in range(steps):
            y_step = batch_y[:, step, :]

            plm_flat = plm_features.view(-1, self.plm_channel)
            y_flat = y_step.view(-1, 1)

            K_plm = self._compute_kernel_matrix(plm_flat)
            K_y = self._compute_kernel_matrix(y_flat)

            n = K_plm.shape[0]
            H = (
                torch.eye(n, device=K_plm.device)
                - torch.ones(n, n, device=K_plm.device) / n
            )
            K_plm_centered = H @ K_plm @ H
            K_y_centered = H @ K_y @ H

            hsic = torch.trace(K_plm_centered @ K_y_centered) / (n**2)

            norm_plm = torch.sqrt(torch.trace(K_plm_centered @ K_plm_centered) / (n**2))
            norm_y = torch.sqrt(torch.trace(K_y_centered @ K_y_centered) / (n**2))

            alignment = hsic / (norm_plm * norm_y + 1e-8)

            total_distance += 1.0 - alignment

        return total_distance / steps

    def compute_cosine_similarity(self, plm_features, batch_y):
        """
        Use cosine similarity to compute correlation between features and targets.
        Handles both single-step and multi-step predictions uniformly.
        """
        # Unified processing of batch_y format
        batch_y = self._prepare_batch_y(batch_y)
        batch_size, steps, num_nodes = batch_y.shape

        # Calculate cosine similarity separately for each time step, then average
        total_distance = 0.0

        for step in range(steps):
            y_step = batch_y[:, step, :]

            plm_flat = plm_features.reshape(-1, self.plm_channel)
            y_flat = y_step.reshape(-1, 1)

            plm_projected = self.feature_projector(plm_flat)

            plm_norm = torch.norm(plm_projected)
            y_norm = torch.norm(y_flat)

            if plm_norm == 0 or y_norm == 0:
                cosine_sim = 0.0
            else:
                dot_product = (plm_projected * y_flat).sum()
                cosine_sim = dot_product / (plm_norm * y_norm)

            # Accumulate distance
            total_distance += 1.0 - cosine_sim

        return total_distance / steps

    def compute_simple_cosine(self, plm_features, batch_y):
        """
        Simple cosine similarity: use average pooling for dimensionality reduction.
        Handles both single-step and multi-step predictions uniformly.
        """
        # Unified processing of batch_y format
        batch_y = self._prepare_batch_y(batch_y)
        batch_size, steps, num_nodes = batch_y.shape

        # Calculate cosine similarity separately for each time step, then average
        total_distance = 0.0

        for step in range(steps):
            y_step = batch_y[:, step, :]

            plm_flat = plm_features.reshape(-1, self.plm_channel)
            y_flat = y_step.reshape(-1, 1)

            plm_reduced = plm_flat.mean(dim=-1, keepdim=True)

            plm_norm = torch.norm(plm_reduced)
            y_norm = torch.norm(y_flat)

            if plm_norm == 0 or y_norm == 0:
                cosine_sim = 0.0
            else:
                dot_product = (plm_reduced * y_flat).sum()
                cosine_sim = dot_product / (plm_norm * y_norm)

            # Accumulate distance
            total_distance += 1.0 - cosine_sim

        return total_distance / steps

    def _compute_kernel_matrix(self, x, gamma=None):
        """Compute RBF kernel matrix"""
        if gamma is None:
            distances = torch.cdist(x, x)
            gamma = 1.0 / torch.median(distances[distances > 0])

        distances_sq = torch.cdist(x, x) ** 2
        return torch.exp(-gamma * distances_sq)

    def _mmd_rbf(self, x, y, gamma=None):
        """Standard MMD computation"""
        if gamma is None:
            combined = torch.cat([x, y], dim=0)
            distances = torch.cdist(combined, combined)
            mask = distances > 0
            if mask.any():
                gamma = 1.0 / torch.median(distances[mask])
            else:
                gamma = 1.0

        K_xx = torch.exp(-gamma * torch.cdist(x, x) ** 2)
        K_yy = torch.exp(-gamma * torch.cdist(y, y) ** 2)
        K_xy = torch.exp(-gamma * torch.cdist(x, y) ** 2)

        mmd2 = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
        return torch.sqrt(torch.clamp(mmd2, min=0))

    def forward(self, plm_features, batch_y, base_loss):
        """
        Compute total loss.
        All methods handle both single-step and multi-step predictions uniformly.
        """
        if self.method == "sliced_mmd":
            reg_value = self.compute_sliced_mmd(plm_features, batch_y)
        elif self.method == "learned_projection":
            reg_value = self.compute_learned_projection_mmd(plm_features, batch_y)
        elif self.method == "kernel_alignment":
            reg_value = self.compute_kernel_alignment(plm_features, batch_y)
        elif self.method == "cosine_similarity":
            reg_value = self.compute_cosine_similarity(plm_features, batch_y)
        elif self.method == "simple_cosine":
            reg_value = self.compute_simple_cosine(plm_features, batch_y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        total_loss = base_loss + self.phi * reg_value

        return total_loss, reg_value.item()


class MultiScaleMMD(nn.Module):
    """
    Multi-scale MMD: Compute MMD at different dimensions.
    """

    def __init__(self, phi=0.01, scales=[1, 8, 32, 128]):
        super().__init__()
        self.phi = phi
        self.scales = scales

    def forward(self, plm_features, batch_y, base_loss):
        """
        Compute MMD at multiple scales.
        Handles both single-step and multi-step predictions uniformly.
        """
        # Unified processing of batch_y format
        if batch_y.dim() == 2:
            batch_y = batch_y.unsqueeze(1)
        elif batch_y.dim() == 4:
            batch_y = batch_y.squeeze(-1)

        batch_size, steps, num_nodes = batch_y.shape
        total_mmd_across_steps = 0.0

        for step in range(steps):
            y_step = batch_y[:, step, :]
            step_mmd = 0.0

            for scale in self.scales:
                if scale == 1:
                    plm_reduced = plm_features.mean(dim=-1, keepdim=True)
                    y_expanded = y_step.unsqueeze(-1)
                else:
                    proj = torch.randn(
                        plm_features.shape[-1], scale, device=plm_features.device
                    )
                    proj = F.normalize(proj, p=2, dim=0)

                    plm_reduced = plm_features @ proj
                    y_expanded = y_step.unsqueeze(-1).expand(-1, scale)

                plm_flat = plm_reduced.view(-1, scale)
                y_flat = y_expanded.view(-1, scale)

                mmd = self._compute_mmd(plm_flat, y_flat)
                step_mmd += mmd

            total_mmd_across_steps += step_mmd / len(self.scales)

        avg_mmd = total_mmd_across_steps / steps
        total_loss = base_loss + self.phi * avg_mmd

        return total_loss, avg_mmd.item()

    def _compute_mmd(self, x, y):
        """Compute MMD"""
        gamma = 1.0 / (2 * x.shape[-1])

        K_xx = torch.exp(-gamma * torch.cdist(x, x) ** 2)
        K_yy = torch.exp(-gamma * torch.cdist(y, y) ** 2)
        K_xy = torch.exp(-gamma * torch.cdist(x, y) ** 2)

        mmd2 = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
        return torch.sqrt(torch.clamp(mmd2, min=0))
