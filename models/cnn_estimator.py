"""Optional CNN-based Massive MIMO channel estimator operating on 2D (Nr, Tp) grids."""

import torch
import torch.nn as nn

from config.config import DEFAULT_MIMO


class CNNChannelEstimator(nn.Module):
    """2D Convolutional Neural Network for Massive MIMO channel estimation.

    Reshapes flat input features (B, 2 * Nr * Tp) into (B, 2, Nr, Tp),
    extracts spatial-temporal features across receive antennas and pilot slots,
    and projects to flattened channel coefficients (B, 2 * Nr * Nt).
    """

    def __init__(
        self,
        nr: int = DEFAULT_MIMO.nr,
        nt: int = DEFAULT_MIMO.nt,
        tp: int = DEFAULT_MIMO.tp,
        channels: tuple = (32, 64, 32),
    ) -> None:
        super().__init__()
        self.nr = int(nr)
        self.nt = int(nt)
        self.tp = int(tp)
        self.input_dim = 2 * self.nr * self.tp
        self.output_dim = 2 * self.nr * self.nt
        self.channels = tuple(int(c) for c in channels)

        c1, c2, c3 = self.channels
        self.conv = nn.Sequential(
            nn.Conv2d(2, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * self.nr * self.tp, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        half = self.nr * self.tp
        real_grid = x[:, :half].view(batch_size, 1, self.nr, self.tp)
        imag_grid = x[:, half:].view(batch_size, 1, self.nr, self.tp)
        grid = torch.cat([real_grid, imag_grid], dim=1)  # (B, 2, Nr, Tp)
        feats = self.conv(grid)
        return self.head(feats)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> dict:
        """Return architecture hyperparameters for checkpoint serialization."""
        return {
            "model_type": "cnn",
            "nr": self.nr,
            "nt": self.nt,
            "tp": self.tp,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "channels": self.channels,
        }
