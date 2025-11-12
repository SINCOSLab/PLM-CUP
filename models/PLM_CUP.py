import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from layers.spatiotemporal_embedding import SpatioTemporalEmbedding
from layers.plm_adapter import PLMAdapter
from layers.semantic_decomposition import TrendModule, SeasonalModule
from layers.gdaconv import (
    GDAConv,
    GDAConvExtended,
    FeatureProjectionLayer,
    NodeTransformLayer,
)
import os


class PLM_CUP(nn.Module):
    """
    PLM-CUP: Pre-trained Language Model for Cross-city Urban flow Prediction

    This model leverages pre-trained language models (PLMs) as an additional source domain
    for cross-city urban flow prediction. The architecture consists of:
    1. Semantic Bridge Encoder: Transforms spatiotemporal flow patterns into language-aligned representations
    2. Fine-tuned PLM: Pre-trained language model for knowledge transfer
    3. Task Adapter: Spatiotemporal self-attention for multi-step prediction

    Reference: "Exploiting Pre-trained Language Model for Cross-city Urban Flow Prediction
               Guided by Information-theoretic Analysis"
    """

    def __init__(
        self,
        device,
        input_dim=3,
        channels=64,
        num_nodes=None,
        input_len=12,
        output_len=12,
        dropout=0.1,
        gpt_path="",
        gpt_layers=6,
        U=1,
        small_value_threshold=1.0,
        use_lora=True,
        lora_r=8,
        lora_alpha=32,
        is_transfer=False,
        source_nodes=None,
        pretrained_model_path=None,
        pretrain_model="",
    ):
        super().__init__()

        self.device = device
        self.num_nodes = num_nodes
        self.node_dim = channels
        self.input_len = input_len
        self.input_dim = input_dim
        self.output_len = output_len
        self.dropout = dropout
        self.gpt_path = gpt_path
        self.gpt_layers = gpt_layers
        self.U = U
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.gpt_channel = 256
        self.is_transfer = is_transfer
        self.source_nodes = source_nodes

        if num_nodes is None:
            raise ValueError("num_nodes must be provided")

        if self.is_transfer and self.source_nodes is None:
            raise ValueError("source_nodes must be provided when is_transfer is True")

        if (
            num_nodes == 358 or num_nodes == 307 or num_nodes == 883 or num_nodes == 170
        ):  # pems datasets
            time = 288
        elif num_nodes == 250 or num_nodes == 266:  # bike and taxi
            time = 48
        elif num_nodes == 225:
            time = 24  # hk, nj and sh

        gpt_channel = self.gpt_channel

        self.Temb = SpatioTemporalEmbedding(time, gpt_channel, dropout=dropout)

        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, gpt_channel))
        nn.init.xavier_uniform_(self.node_emb)

        self.start_conv = GDAConv(
            self.input_dim * self.input_len, gpt_channel, type=True
        )

        self.fusion_transform = GDAConv(gpt_channel * 3, gpt_channel, type=True)

        self.trend_module = TrendModule(
            input_size=gpt_channel,
            degree=7,
            num_nodes=num_nodes,
            small_value_threshold=small_value_threshold,
        )
        self.seasonal_module = SeasonalModule(
            input_size=gpt_channel,
            num_harmonics=2,
            num_nodes=num_nodes,
            small_value_threshold=small_value_threshold,
        )

        self.trend_preproc = GDAConvExtended(gpt_channel, gpt_channel, type=True)
        self.seasonal_preproc = GDAConvExtended(gpt_channel, gpt_channel, type=True)

        self.feature_transform = GDAConv(gpt_channel * 2, gpt_channel, type=True)

        self.plm = PLMAdapter(
            device=self.device,
            gpt_layers=self.gpt_layers,
            U=self.U,
            gpt_path=self.gpt_path,
            pretrain_model=pretrain_model,
            use_lora=self.use_lora,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
        )

        plm_channel = self.plm.hidden_size

        self.to_plm = FeatureProjectionLayer(gpt_channel, plm_channel, type=False)

        self.from_plm = FeatureProjectionLayer(plm_channel, gpt_channel, type=False)

        self.temporal_attention = nn.ModuleDict(
            {
                "query_proj": GDAConv(gpt_channel, gpt_channel, type=False),
                "key_proj": GDAConv(gpt_channel, gpt_channel, type=True),
                "value_proj": GDAConv(gpt_channel, gpt_channel, type=True),
                "output_proj": GDAConv(gpt_channel, gpt_channel, type=False),
            }
        )

        self.step_embeddings = nn.Parameter(torch.empty(self.output_len, gpt_channel))
        nn.init.xavier_uniform_(self.step_embeddings)

        self.regression_layer = nn.Sequential(
            GDAConv(gpt_channel, gpt_channel, type=True),
            nn.Conv2d(gpt_channel, self.output_len, kernel_size=(1, 1)),
        )

        if self.is_transfer:
            self.forward_mapper = NodeTransformLayer(
                source_nodes=num_nodes,  # Target domain -> Source domain
                target_nodes=source_nodes,
                channel_size=gpt_channel,
                dropout=dropout,
            )

            self.backward_mapper = NodeTransformLayer(
                source_nodes=source_nodes,  # Source domain -> Target domain
                target_nodes=num_nodes,
                channel_size=gpt_channel,
                dropout=dropout,
            )

            if pretrained_model_path is not None:
                self.load_pretrained_components(pretrained_model_path)
            else:
                pass  # No pretrained path provided

        self._check_and_set_dtype()

    def _check_and_set_dtype(self):
        """Check PLM model dtype and convert entire model if needed."""
        plm_dtype = next(self.plm.parameters()).dtype

        if plm_dtype == torch.float16:
            other_dtype = next(
                (p.dtype for name, p in self.named_parameters() if "plm" not in name),
                torch.float32,
            )

            if other_dtype != torch.float16:
                for name, module in self.named_modules():
                    if "plm" not in name and hasattr(module, "half"):
                        module.half()

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def load_pretrained_components(self, pretrained_model_path):
        """Optimized pretrained component loading - directly load all parameters of Component A"""
        component_A_path = f"{pretrained_model_path}_A.pth"
        component_B_path = f"{pretrained_model_path}_B.pth"

        # Load Component A - directly load all parameters (since parameters are node-independent)
        if os.path.exists(component_A_path):
            component_A = torch.load(component_A_path, map_location=self.device)

            try:
                # Directly load all Component A parameters - no need to check shapes
                self.trend_module.load_state_dict(component_A["trend_module"])
                self.seasonal_module.load_state_dict(component_A["seasonal_module"])
                self.trend_preproc.load_state_dict(component_A["trend_preproc"])
                self.seasonal_preproc.load_state_dict(component_A["seasonal_preproc"])
                self.feature_transform.load_state_dict(component_A["feature_transform"])

                # Component A loaded successfully
            except Exception as e:
                # Error loading Component A, using random initialization
                raise e
        else:
            # Component A not found
            raise ValueError(f"Component A not found at {component_A_path}")

        # Load Component B (PLM) - support flexible loading with different layer numbers
        if os.path.exists(component_B_path):
            component_B = torch.load(component_B_path, map_location=self.device)

            try:
                # Try direct loading (when layer numbers match)
                self.plm.load_state_dict(component_B["plm"])
                # Component B loaded
            except RuntimeError as e:
                if "Unexpected key(s)" in str(e) or "Missing key(s)" in str(e):
                    # PLM layer mismatch, attempting partial load

                    # Get state dict of current model and pretrained model
                    current_state = self.plm.state_dict()
                    pretrained_state = component_B["plm"]

                    # Load only matching parameters
                    matched_params = {}
                    for key, value in pretrained_state.items():
                        if (
                            key in current_state
                            and current_state[key].shape == value.shape
                        ):
                            matched_params[key] = value

                    # Load matching parameters
                    self.plm.load_state_dict(matched_params, strict=False)

                    # Statistics
                    total_params = len(pretrained_state)
                    loaded_params = len(matched_params)
                    # Partial PLM load successful
                else:
                    # Error loading Component B, using random initialization
                    raise e
        else:
            # Component B not found
            raise ValueError(f"Component B not found at {component_B_path}")

    def forward(self, history_data):
        """
        Forward pass of PLM-CUP model.

        Args:
            history_data: Historical urban flow data
                         Shape: (batch_size, input_dim, num_nodes, input_len)

        Returns:
            Predicted urban flow for future time steps
            Shape: (batch_size, output_len, num_nodes, 1)
        """
        # Ensure input data dtype matches model parameters
        model_dtype = next(self.parameters()).dtype
        if history_data.dtype != model_dtype:
            history_data = history_data.to(model_dtype)

        input_data = history_data
        batch_size, _, num_nodes, _ = input_data.shape

        # Prepare embeddings and features
        history_data = history_data.permute(0, 3, 2, 1)
        tem_emb = self.Temb(history_data)

        node_emb = []
        node_emb.append(
            self.node_emb.unsqueeze(0)
            .expand(batch_size, -1, -1)
            .transpose(1, 2)
            .unsqueeze(-1)
        )

        # Transform input data
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = (
            input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        )
        input_data = self.start_conv(input_data)

        # Fuse features
        data_st = torch.cat([input_data] + [tem_emb] + node_emb, dim=1)
        data_st = self.fusion_transform(data_st)

        # Semantic decomposition
        if self.is_transfer:
            # Transfer learning mode
            data_in_source_space = self.forward_mapper(data_st)
            trend_output = self.trend_module(data_in_source_space)
            seasonal_output = self.seasonal_module(data_in_source_space)
            trend_output = self.trend_preproc(trend_output)
            seasonal_output = self.seasonal_preproc(seasonal_output)
            combined_source = torch.cat([trend_output, seasonal_output], dim=1)
            combined_source = self.feature_transform(combined_source)
            combined_output = self.backward_mapper(combined_source)
        else:
            # Standard mode
            trend_output = self.trend_module(data_st)
            seasonal_output = self.seasonal_module(data_st)
            trend_output = self.trend_preproc(trend_output)
            seasonal_output = self.seasonal_preproc(seasonal_output)
            combined_output = torch.cat([trend_output, seasonal_output], dim=1)
            combined_output = self.feature_transform(combined_output)

        # PLM processing
        features = combined_output.squeeze(-1).permute(0, 2, 1)
        features = self.to_plm(features)
        features = self.plm(features)
        features = self.from_plm(features)
        features = features.permute(0, 2, 1).unsqueeze(-1)
        features = features + data_st

        # Multi-step temporal attention
        query = self.temporal_attention["query_proj"](features)
        key = self.temporal_attention["key_proj"](features)
        value = self.temporal_attention["value_proj"](features)

        step_emb = self.step_embeddings.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        step_emb = step_emb.expand(batch_size, -1, num_nodes, 1, -1)

        query = query.unsqueeze(1).expand(-1, self.output_len, -1, -1, -1)
        query = query.permute(0, 1, 3, 4, 2) + step_emb

        key = (
            key.unsqueeze(1)
            .expand(-1, self.output_len, -1, -1, -1)
            .permute(0, 1, 3, 4, 2)
        )

        value = (
            value.unsqueeze(1)
            .expand(-1, self.output_len, -1, -1, -1)
            .permute(0, 1, 3, 4, 2)
        )

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (
            self.gpt_channel**0.5
        )
        attention_weights = F.softmax(attention_scores, dim=-1)

        attended_features = torch.matmul(attention_weights, value)
        attended_features = attended_features.permute(0, 4, 2, 3, 1)

        features = attended_features.permute(0, 1, 2, 4, 3).contiguous()
        features = features.view(batch_size, self.gpt_channel, num_nodes, -1)

        features = self.temporal_attention["output_proj"](features)
        features = features.mean(dim=-1, keepdim=True)

        output = self.regression_layer(features)

        return output

    def save_component_A(self, path):
        """Save Component A (semantic decomposition modules) for transfer learning."""
        component_A = {
            "trend_module": self.trend_module.state_dict(),
            "seasonal_module": self.seasonal_module.state_dict(),
            "trend_preproc": self.trend_preproc.state_dict(),
            "seasonal_preproc": self.seasonal_preproc.state_dict(),
            "feature_transform": self.feature_transform.state_dict(),
        }
        torch.save(component_A, path)

    def save_component_B(self, path):
        """Save Component B (PLM adapter) for transfer learning."""
        component_B = {
            "plm": self.plm.state_dict(),
        }
        torch.save(component_B, path)
