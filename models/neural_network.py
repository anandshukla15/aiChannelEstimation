"""Configurable DNN-based Massive MIMO channel estimator in PyTorch."""

from typing import Sequence
import torch
import torch.nn as nn

from config.config import DEFAULT_MIMO, DEFAULT_MODEL


class ChannelEstimator(nn.Module):
    """Fully-connected Deep Neural Network (DNN) for Massive MIMO channel estimation.

    Maps flattened real/imaginary received pilot signals:
        x in R^{2 * Nr * Tp} -> H_est in R^{2 * Nr * Nt}

    Architecture:
        - Nonlinear MLP backbone: hidden_dims (default: 512 -> 256 -> 256) with ReLU
        - Optional linear residual skip connection: Linear(input_dim, output_dim)
          which captures the linear pilot-matched projection (akin to LS/LMMSE
          projection) while the ReLU MLP learns SNR-adaptive shrinkage and
          structured channel priors without suffering from a ReLU rank bottleneck.
    """

    def __init__(
        self,
        input_dim: int = DEFAULT_MIMO.input_dim,
        output_dim: int = DEFAULT_MIMO.output_dim,
        hidden_dims: Sequence[int] = DEFAULT_MODEL.hidden_dims,
        dropout: float = DEFAULT_MODEL.dropout,
        use_layer_norm: bool = False,
        residual_skip: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.dropout = float(dropout)
        self.use_layer_norm = bool(use_layer_norm)
        self.residual_skip = bool(residual_skip)

        layers = []
        in_features = self.input_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(in_features, h_dim))
            if self.use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            if self.dropout > 0.0:
                layers.append(nn.Dropout(p=self.dropout))
            in_features = h_dim

        layers.append(nn.Linear(in_features, self.output_dim))
        self.network = nn.Sequential(*layers)
        self.skip = nn.Linear(self.input_dim, self.output_dim) if self.residual_skip else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)
        if self.skip is not None:
            out = out + self.skip(x)
        return out

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> dict:
        """Return architecture hyperparameters for checkpoint serialization."""
        return {
            "model_type": "dnn",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "use_layer_norm": self.use_layer_norm,
            "residual_skip": self.residual_skip,
        }
