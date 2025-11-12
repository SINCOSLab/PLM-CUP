import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, AutoModel
import math
import os
import time
from pathlib import Path
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils.model_cache import get_cached_model
except ImportError:
    get_cached_model = None


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, lora_alpha=32):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.weight.requires_grad = False

        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)

        # Improved initialization for better numerical stability with FP16
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # Ensure input dtype matches linear layer dtype
        if x.dtype != self.linear.weight.dtype:
            x = x.to(self.linear.weight.dtype)

        result = self.linear(x)
        lora_output = self.lora_A(x)
        lora_output = self.lora_B(lora_output)

        # Add numerical stability check
        output = result + lora_output * self.scaling
        if torch.isnan(output).any() or torch.isinf(output).any():
            # Fallback to base model output if LoRA causes instability
            return result
        return output


class LoRAConv1D(nn.Module):
    """Wrapper for GPT2 Conv1D with LoRA adaptation"""

    def __init__(self, original_conv1d, r=8, lora_alpha=32):
        super().__init__()
        self.original_conv1d = original_conv1d
        # Conv1D: weight shape is [nx, nf] where nx=input_dim, nf=output_dim
        # For c_attn: nx=768 (hidden_size), nf=2304 (3*hidden_size for Q,K,V)
        self.nx = original_conv1d.weight.shape[0]  # input dimension
        self.nf = original_conv1d.weight.shape[1]  # output dimension

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # Freeze original weights
        self.original_conv1d.weight.requires_grad = False
        self.original_conv1d.bias.requires_grad = False

        # LoRA layers: input_dim -> r -> output_dim
        self.lora_A = nn.Linear(self.nx, r, bias=False)
        self.lora_B = nn.Linear(r, self.nf, bias=False)

        # Initialize LoRA weights with smaller std for FP16 stability
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # Original Conv1D forward
        base_output = self.original_conv1d(x)

        # LoRA computation
        lora_output = self.lora_A(x)
        lora_output = self.lora_B(lora_output)

        return base_output + lora_output * self.scaling


class PLMAdapter(nn.Module):
    def __init__(
        self,
        device="cuda:0",
        gpt_layers=6,
        U=1,
        gpt_path=None,
        pretrain_model="gpt2",
        use_lora=False,
        lora_r=8,
        lora_alpha=32,
    ):
        super(PLMAdapter, self).__init__()

        self.device = device
        self.gpt_layers = gpt_layers
        self.U = U
        self.pretrain_model = pretrain_model
        self.use_lora = str(use_lora).lower() == "true"
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha

        if gpt_path is None:
            raise ValueError(f"gpt_path is required for model {pretrain_model}")

        self._load_model(gpt_path)

        self.hidden_size = self.model.config.hidden_size
        self.num_heads = self.model.config.num_attention_heads

        self._limit_layers()

        if self.pretrain_model == "qwen3-0.6b":
            if next(self.model.parameters()).dtype != torch.float16:
                self.model = self.model.half()

        if self.use_lora:
            self._apply_lora()
        else:
            self._apply_traditional_finetuning()

    def _load_model(self, model_path):
        start_time = time.time()

        # Use the simple caching system
        if get_cached_model:
            try:
                self.model, from_cache = get_cached_model(
                    self.pretrain_model, model_path, None
                )

                if (
                    self.device != "cpu"
                    and hasattr(self.model, "device")
                    and self.model.device != self.device
                ):
                    self.model = self.model.to(self.device)

                load_time = time.time() - start_time
                return

            except Exception as e:
                raise e

        # Common args for optimization
        common_args = {
            "output_attentions": True,
            "output_hidden_states": True,
            "low_cpu_mem_usage": True,
        }

        # Check if local model
        is_local_model = os.path.isdir(model_path) and os.path.exists(
            os.path.join(model_path, "config.json")
        )

        if is_local_model:
            common_args["local_files_only"] = True

        try:
            if self.pretrain_model == "gpt2":
                self.model = GPT2Model.from_pretrained(model_path, **common_args)
            elif self.pretrain_model == "qwen3-0.6b":
                common_args.update(
                    {
                        "trust_remote_code": True,
                        "torch_dtype": torch.float16,
                        "attn_implementation": "eager",
                    }
                )
                self.model = AutoModel.from_pretrained(model_path, **common_args)
            else:
                raise ValueError(f"Unsupported model type: {self.pretrain_model}")

            load_time = time.time() - start_time

        except Exception as e:
            raise

    def _limit_layers(self):
        # If gpt_layers == -1, use all layers
        if self.gpt_layers == -1:
            if hasattr(self.model, "h"):
                actual_layers = len(self.model.h)
                self.gpt_layers = actual_layers  # Update to actual layer count
            elif hasattr(self.model, "layers"):
                actual_layers = len(self.model.layers)
                self.gpt_layers = actual_layers  # Update to actual layer count
        else:
            # Limit layer count
            if hasattr(self.model, "h"):
                self.model.h = self.model.h[: self.gpt_layers]
            elif hasattr(self.model, "layers"):
                self.model.layers = self.model.layers[: self.gpt_layers]

    def _apply_lora(self):

        if self.pretrain_model == "gpt2":
            for layer in self.model.h:
                attn = layer.attn
                old_c_attn = attn.c_attn

                # Skip if already a LoRAConv1D
                if isinstance(old_c_attn, LoRAConv1D):
                    continue

                # Wrap original Conv1D with LoRA
                lora_c_attn = LoRAConv1D(
                    old_c_attn,
                    r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                )
                attn.c_attn = lora_c_attn

        elif self.pretrain_model == "qwen3-0.6b":
            layers = (
                self.model.layers if hasattr(self.model, "layers") else self.model.h
            )

            for layer in layers:
                attn = layer.self_attn if hasattr(layer, "self_attn") else layer.attn

                for proj_name in ["q_proj", "k_proj", "v_proj"]:
                    if hasattr(attn, proj_name):
                        old_proj = getattr(attn, proj_name)

                        # Skip if already a LoRALinear
                        if isinstance(old_proj, LoRALinear):
                            continue

                        lora_proj = LoRALinear(
                            old_proj.in_features,
                            old_proj.out_features,
                            r=self.lora_r,
                            lora_alpha=self.lora_alpha,
                        )
                        lora_proj.linear.weight.data = old_proj.weight.data

                        # Ensure LoRA weights match the original model's dtype
                        target_dtype = old_proj.weight.dtype
                        lora_proj.lora_A.weight.data = lora_proj.lora_A.weight.data.to(
                            target_dtype
                        )
                        lora_proj.lora_B.weight.data = lora_proj.lora_B.weight.data.to(
                            target_dtype
                        )

                        # Also ensure the linear layer itself has the correct dtype
                        lora_proj.linear = lora_proj.linear.to(target_dtype)

                        setattr(attn, proj_name, lora_proj)

        # First, freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Then selectively enable training for specific parameters
        for name, param in self.model.named_parameters():
            if "lora" in name:
                # All LoRA parameters are trainable
                param.requires_grad = True
            else:
                param.requires_grad = False

        # LoRA configuration complete: all LoRA parameters trainable, original parameters frozen

    def _apply_traditional_finetuning(self):

        if self.pretrain_model == "gpt2":
            if hasattr(self.model, "wpe"):
                self.model.wpe.weight.requires_grad = True

            for layer_index, layer in enumerate(self.model.h):
                for name, param in layer.named_parameters():
                    if layer_index < self.gpt_layers - self.U:
                        # First L-U layers: only LayerNorm trainable
                        if "ln" in name:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                    else:
                        # Last U layers: LayerNorm and attention trainable, MLP frozen
                        if "mlp" in name:
                            param.requires_grad = False
                        elif "ln" in name or "attn" in name:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
        else:
            layers = (
                self.model.layers if hasattr(self.model, "layers") else self.model.h
            )

            for layer_index, layer in enumerate(layers):
                for name, param in layer.named_parameters():
                    if layer_index < self.gpt_layers - self.U:
                        # First L-U layers: only LayerNorm-related parameters trainable
                        if "norm" in name.lower() or "layernorm" in name.lower():
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                    else:
                        # Last U layers: LayerNorm and attention trainable, MLP frozen
                        if "mlp" in name.lower() or "feed_forward" in name.lower():
                            param.requires_grad = False
                        elif (
                            "norm" in name.lower()
                            or "layernorm" in name.lower()
                            or "attn" in name.lower()
                            or "attention" in name.lower()
                        ):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False

    def forward(self, x):
        # Ensure input matches model dtype
        model_dtype = next(self.model.parameters()).dtype
        if x.dtype != model_dtype:
            x = x.to(model_dtype)

        batch_size, seq_len = x.size(0), x.size(1)
        attention_mask = torch.ones(
            batch_size, seq_len, dtype=torch.long, device=x.device
        )

        outputs = self.model(
            inputs_embeds=x,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
        )

        return outputs.last_hidden_state
