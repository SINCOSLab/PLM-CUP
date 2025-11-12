import torch
from utils.metrics import MAE_torch, MAPE_torch, RMSE_torch, WMAPE_torch
from utils.ranger import Ranger
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from utils.domain_shift_regularizer import DomainShiftRegularizer
import os
import json
from datetime import datetime


class trainer:
    def __init__(
        self,
        scaler,
        learning_rate,
        weight_decay,
        lr_decay,
        lr_decay_patience,
        min_learning_rate,
        model,
        device,
        is_pems=False,
        use_domain_shift_regularization=False,
        domain_shift_config=None,
    ):
        self.model = model
        self.model.to(device)
        self.device = device

        # Domain shift regularization settings
        self.use_domain_shift_regularization = use_domain_shift_regularization
        self.domain_shift_regularizer = None

        # Initialize logging
        self.ds_log = []
        self.current_epoch = 0
        self.batch_count = 0

        if self.use_domain_shift_regularization:
            # Dynamically get plm_channel from model
            plm_channel = 768  # Default value
            if hasattr(model, "plm") and hasattr(model.plm, "hidden_size"):
                plm_channel = model.plm.hidden_size
                pass  # PLM channel detected
            else:
                pass  # Using default PLM channel

            # Default configuration
            default_config = {
                "method": "kernel_alignment",
                "phi": 0.01,
                "plm_channel": plm_channel,
            }

            # Simple configuration for gradient detachment
            self.detach_features = False

            if domain_shift_config:
                # Extract detach_features config (not passed to regularizer)
                self.detach_features = domain_shift_config.get("detach_features", False)

                # Create a copy to avoid modifying original config
                config_copy = domain_shift_config.copy()
                if "detach_features" in config_copy:
                    config_copy.pop("detach_features")

                default_config.update(config_copy)
                # If plm_channel not specified in config, use value from model
                if "plm_channel" not in config_copy:
                    default_config["plm_channel"] = plm_channel

            # Gradient configuration set

            self.domain_shift_regularizer = DomainShiftRegularizer(**default_config).to(
                device
            )

            # Domain shift regularizer initialized

        # Optimizer setup
        self.optimizer = Ranger(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=lr_decay,
            patience=lr_decay_patience,
            min_lr=min_learning_rate,
            verbose=True,
        )

        self.loss = MAE_torch
        self.scaler = scaler
        self.clip = 5
        self.learning_rate = learning_rate
        self.warmup_steps = 1600 if is_pems else 800
        self.current_step = 0

        self.use_fp16 = False
        for param in model.parameters():
            if param.dtype == torch.float16:
                self.use_fp16 = True
                break

        if not self.use_fp16:
            self.grad_scaler = GradScaler(
                init_scale=2.0**16,
                growth_factor=2,
                backoff_factor=0.5,
                growth_interval=2000,
            )
        else:
            self.grad_scaler = None

    def _extract_plm_features(self, model, input):
        """
        Extract features after to_plm for domain shift regularization
        """
        features = {}

        def hook_fn(module, input, output):
            features["plm"] = output

        # Register hook on to_plm layer
        handle = None
        if hasattr(model, "to_plm"):
            handle = model.to_plm.register_forward_hook(hook_fn)

        # Forward pass
        output = model(input)

        # Remove hook
        if handle:
            handle.remove()

        return output, features.get("plm", None)

    def set_epoch(self, epoch):
        """Set current epoch for logging"""
        self.current_epoch = epoch
        self.batch_count = 0

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        self.batch_count += 1

        # Re-check model dtype in case it was changed after initialization
        current_dtype = next(self.model.parameters()).dtype
        actual_use_fp16 = current_dtype == torch.float16

        if actual_use_fp16:
            with autocast(dtype=torch.float16):
                if (
                    self.use_domain_shift_regularization
                    and self.domain_shift_regularizer is not None
                ):
                    # Extract PLM features
                    output, plm_features = self._extract_plm_features(self.model, input)
                else:
                    output = self.model(input)
                    plm_features = None

                output = output.transpose(1, 3)
                # Convert to FP32 for loss calculation
                if output.dtype == torch.float16:
                    output = output.float()
                real = torch.unsqueeze(real_val, dim=1)
                predict = self.scaler.inverse_transform(output)
                base_loss = self.loss(predict, real, 0.0)

                # Add kernel alignment regularization
                if (
                    self.use_domain_shift_regularization
                    and self.domain_shift_regularizer is not None
                    and plm_features is not None
                ):
                    # Decide whether to detach gradient based on config
                    if self.detach_features:
                        plm_features_input = plm_features.detach()
                    else:
                        plm_features_input = plm_features

                    total_loss, ds_value = self.domain_shift_regularizer(
                        plm_features_input, real_val, base_loss
                    )
                    # Record log
                    self.ds_log.append(
                        {
                            "epoch": self.current_epoch,
                            "batch": self.batch_count,
                            "base_loss": base_loss.item(),
                            "ds_value": ds_value,
                            "total_loss": total_loss.item(),
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                    self.ds_value = ds_value
                else:
                    total_loss = base_loss
                    self.ds_value = 0.0

        else:
            # For FP32 models, use standard autocast with gradient scaling
            with autocast():
                if (
                    self.use_domain_shift_regularization
                    and self.domain_shift_regularizer is not None
                ):
                    # Extract PLM features
                    output, plm_features = self._extract_plm_features(self.model, input)
                else:
                    output = self.model(input)
                    plm_features = None

                output = output.transpose(1, 3)
                real = torch.unsqueeze(real_val, dim=1)
                predict = self.scaler.inverse_transform(output)
                base_loss = self.loss(predict, real, 0.0)

                # Add kernel alignment regularization
                if (
                    self.use_domain_shift_regularization
                    and self.domain_shift_regularizer is not None
                    and plm_features is not None
                ):
                    # Decide whether to detach gradient based on config
                    if self.detach_features:
                        plm_features_input = plm_features.detach()
                    else:
                        plm_features_input = plm_features

                    total_loss, ds_value = self.domain_shift_regularizer(
                        plm_features_input, real_val, base_loss
                    )
                    # Record log
                    self.ds_log.append(
                        {
                            "epoch": self.current_epoch,
                            "batch": self.batch_count,
                            "base_loss": base_loss.item(),
                            "ds_value": ds_value,
                            "total_loss": total_loss.item(),
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                    self.ds_value = ds_value
                else:
                    total_loss = base_loss
                    self.ds_value = 0.0

        if self.current_step < self.warmup_steps:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate * 10
        elif self.current_step < self.warmup_steps * 1.5:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate * 5
        elif self.current_step < self.warmup_steps * 2.5:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate * 2
        elif self.current_step == self.warmup_steps * 2.5:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

        if actual_use_fp16:
            total_loss.backward()
            if self.clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
        else:
            # For FP32 models, use gradient scaling
            if self.grad_scaler is None:
                # Create grad scaler if needed
                self.grad_scaler = GradScaler(
                    init_scale=2.0**16,
                    growth_factor=2,
                    backoff_factor=0.5,
                    growth_interval=2000,
                )
            self.grad_scaler.scale(total_loss).backward()
            if self.clip is not None:
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

        with torch.no_grad():
            mape = MAPE_torch(predict, real, 0.0).item()
            rmse = RMSE_torch(predict, real, 0.0).item()
            wmape = WMAPE_torch(predict, real, 0.0).item()
        self.current_step += 1

        # Return metrics including kernel alignment values
        return base_loss.item(), mape, rmse, wmape, self.ds_value

    @torch.no_grad()
    def eval(self, input, real_val):
        self.model.eval()

        output = self.model(input)
        output = output.transpose(1, 3)

        # Convert output back to FP32 for metric calculations
        if output.dtype == torch.float16:
            output = output.float()

        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = MAPE_torch(predict, real, 0.0).item()
        rmse = RMSE_torch(predict, real, 0.0).item()
        wmape = WMAPE_torch(predict, real, 0.0).item()
        return loss.item(), mape, rmse, wmape

    def save_ds_log(self, filepath):
        """Save domain shift regularization log to file"""
        if self.ds_log:
            with open(filepath, "w") as f:
                json.dump(self.ds_log, f, indent=2)
