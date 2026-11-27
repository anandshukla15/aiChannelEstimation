"""Centralized configuration for Massive MIMO and 6G XL-MIMO channel estimation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class MIMOConfig:
    """System-level Massive MIMO parameters.

    System model: Y = H @ X + N
    Dimensions:
        H: (Nr, Nt) complex channel matrix
        X: (Nt, Tp) complex/BPSK pilot matrix with X @ X^H = Tp * I_{Nt}
        Y: (Nr, Tp) complex received signal matrix
        N: (Nr, Tp) complex AWGN matrix
    """

    nt: int = 32
    nr: int = 4
    tp: int = 32
    supported_nt: Tuple[int, ...] = (8, 16, 32, 64)
    supported_nr: Tuple[int, ...] = (2, 4, 8)
    snr_db_range: Tuple[float, float] = (0.0, 30.0)
    num_train_val_samples: int = 50000
    num_test_samples: int = 5000
    pilot_type: str = "hadamard"
    train_seed: int = 42
    test_seed: int = 10042

    @property
    def input_dim(self) -> int:
        return 2 * self.nr * self.tp

    @property
    def output_dim(self) -> int:
        return 2 * self.nr * self.nt


@dataclass
class ModelConfig:
    """Neural network architecture and optimization configuration."""

    hidden_dims: Tuple[int, ...] = (512, 256, 256)
    dropout: float = 0.0
    batch_size: int = 256
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    val_split: float = 0.2
    early_stopping_patience: int = 10
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 4
    min_lr: float = 1e-6
    seed: int = 42


@dataclass
class ExperimentConfig:
    """Configuration for experimental sweeps and evaluation."""

    snr_levels: List[float] = field(default_factory=lambda: [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
    antenna_nt_list: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    fixed_nr: int = 4
    pilot_lengths_for_nt32: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    ber_num_data_symbols: int = 50
    ber_samples_per_snr: int = 2000
    sweep_train_samples: int = 25000
    sweep_test_samples: int = 4000
    sweep_epochs: int = 40
    modulation: str = "bpsk"
    detector: str = "mmse"


@dataclass
class XLMIMOConfig:
    """6G-oriented XL-MIMO near-field spherical-wave channel parameters."""

    carrier_frequency_hz: float = 28e9  # 28 GHz mmWave band
    speed_of_light: float = 299792458.0
    nt: int = 64  # Large ULA transmit aperture
    nr: int = 4   # Receive ULA
    tp: int = 64
    antenna_spacing_factor: float = 0.5  # d = 0.5 * lambda
    tx_position_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    near_field_distance_range_m: Tuple[float, float] = (3.0, 20.0)
    far_field_distance_m: float = 200.0
    angle_range_deg: Tuple[float, float] = (-60.0, 60.0)
    non_stationary_visibility: bool = True
    visibility_fraction_range: Tuple[float, float] = (0.4, 1.0)
    rician_k_factor_db: float = 10.0
    num_train_samples: int = 25000
    num_test_samples: int = 4000

    @property
    def wavelength_m(self) -> float:
        return self.speed_of_light / self.carrier_frequency_hz

    @property
    def antenna_spacing_m(self) -> float:
        return self.antenna_spacing_factor * self.wavelength_m

    @property
    def tx_aperture_m(self) -> float:
        return (self.nt - 1) * self.antenna_spacing_m

    @property
    def rayleigh_distance_m(self) -> float:
        """Fraunhofer / Rayleigh distance: d_R = 2 * D^2 / lambda."""
        return 2.0 * (self.tx_aperture_m ** 2) / self.wavelength_m


@dataclass
class PathConfig:
    """Output directories and artifact file paths."""

    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"
    results_dir: Path = PROJECT_ROOT / "results"
    plots_dir: Path = PROJECT_ROOT / "results" / "plots"
    tables_dir: Path = PROJECT_ROOT / "results" / "tables"
    checkpoints_dir: Path = PROJECT_ROOT / "results" / "checkpoints"
    default_dataset_path: Path = PROJECT_ROOT / "data" / "channel_dataset.npz"
    default_test_dataset_path: Path = PROJECT_ROOT / "data" / "channel_test_dataset.npz"

    def ensure_dirs(self) -> None:
        for directory in (
            self.data_dir,
            self.models_dir,
            self.results_dir,
            self.plots_dir,
            self.tables_dir,
            self.checkpoints_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


DEFAULT_MIMO = MIMOConfig()
DEFAULT_MODEL = ModelConfig()
DEFAULT_EXPERIMENT = ExperimentConfig()
DEFAULT_XL_MIMO = XLMIMOConfig()
DEFAULT_PATHS = PathConfig()
