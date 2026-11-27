"""Training pipeline for Massive MIMO AI-based channel estimators."""

import argparse
from pathlib import Path
import random
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config.config import DEFAULT_MIMO, DEFAULT_MODEL, DEFAULT_PATHS
from data.dataset import generate_mimo_dataset, load_mimo_dataset
from evaluation.metrics import count_parameters
from evaluation.visualization import plot_training_curves, save_csv
from models.cnn_estimator import CNNChannelEstimator
from models.neural_network import ChannelEstimator


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(
    model_type: str,
    nr: int,
    nt: int,
    tp: int,
    hidden_dims: Tuple[int, ...] = DEFAULT_MODEL.hidden_dims,
    dropout: float = DEFAULT_MODEL.dropout,
    use_layer_norm: bool = False,
    residual_skip: bool = True,
) -> nn.Module:
    input_dim = 2 * nr * tp
    output_dim = 2 * nr * nt
    if model_type.lower() == "dnn":
        return ChannelEstimator(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            residual_skip=residual_skip,
        )
    if model_type.lower() == "cnn":
        return CNNChannelEstimator(nr=nr, nt=nt, tp=tp)
    raise ValueError(f"Unknown model_type='{model_type}'. Expected 'dnn' or 'cnn'.")


def train_channel_estimator(
    inputs: np.ndarray,
    targets: np.ndarray,
    nr: int,
    nt: int,
    tp: int,
    checkpoint_path: Path,
    loss_plot_path: Optional[Path] = None,
    loss_csv_path: Optional[Path] = None,
    model_type: str = "dnn",
    hidden_dims: Tuple[int, ...] = DEFAULT_MODEL.hidden_dims,
    dropout: float = DEFAULT_MODEL.dropout,
    use_layer_norm: bool = False,
    batch_size: int = DEFAULT_MODEL.batch_size,
    epochs: int = DEFAULT_MODEL.epochs,
    learning_rate: float = DEFAULT_MODEL.learning_rate,
    weight_decay: float = DEFAULT_MODEL.weight_decay,
    val_split: float = DEFAULT_MODEL.val_split,
    early_stopping_patience: int = DEFAULT_MODEL.early_stopping_patience,
    lr_scheduler_factor: float = DEFAULT_MODEL.lr_scheduler_factor,
    lr_scheduler_patience: int = DEFAULT_MODEL.lr_scheduler_patience,
    min_lr: float = DEFAULT_MODEL.min_lr,
    seed: int = DEFAULT_MODEL.seed,
    backup_existing: bool = True,
    verbose: bool = True,
) -> Dict[str, object]:
    """Train a Massive MIMO channel estimator with train-only normalization,
    LR scheduling, early stopping, and best-checkpoint saving.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs_f32 = np.asarray(inputs, dtype=np.float32)
    targets_f32 = np.asarray(targets, dtype=np.float32)

    expected_in = 2 * nr * tp
    expected_out = 2 * nr * nt
    if inputs_f32.ndim != 2 or inputs_f32.shape[1] != expected_in:
        raise ValueError(f"Expected inputs shape (N, {expected_in}), got {inputs_f32.shape}")
    if targets_f32.ndim != 2 or targets_f32.shape[1] != expected_out:
        raise ValueError(f"Expected targets shape (N, {expected_out}), got {targets_f32.shape}")
    if inputs_f32.shape[0] != targets_f32.shape[0]:
        raise ValueError("Inputs and targets must have matching sample count.")

    num_samples = inputs_f32.shape[0]
    val_size = max(1, int(val_split * num_samples))
    train_size = num_samples - val_size

    # Deterministic permutation for clean train/val split without leakage
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_samples)
    train_idx = perm[:train_size]
    val_idx = perm[train_size:]

    x_train_raw = inputs_f32[train_idx]
    x_val_raw = inputs_f32[val_idx]
    y_train = targets_f32[train_idx]
    y_val = targets_f32[val_idx]

    # Compute normalization statistics on TRAINING SET ONLY
    input_mean = x_train_raw.mean(axis=0)
    input_std = np.clip(x_train_raw.std(axis=0), 1e-6, None)

    x_train = (x_train_raw - input_mean) / input_std
    x_val = (x_val_raw - input_mean) / input_std

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=batch_size,
        shuffle=False,
    )

    model = build_model(
        model_type=model_type,
        nr=nr,
        nt=nt,
        tp=tp,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_layer_norm=use_layer_norm,
    ).to(device)
    num_params = count_parameters(model)

    if verbose:
        print(f"Device             : {device}")
        print(f"Antenna config     : Nt={nt}, Nr={nr}, Tp={tp}")
        print(f"Dataset dimensions : X={inputs_f32.shape}, Y={targets_f32.shape} (train={train_size}, val={val_size})")
        print(f"Model ({model_type.upper()})       : hidden={hidden_dims}, params={num_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
        min_lr=min_lr,
    )

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if backup_existing and checkpoint_path.exists():
        backup_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".bak")
        checkpoint_path.replace(backup_path)
        if verbose:
            print(f"Backed up existing checkpoint to {backup_path}")

    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    best_state_dict = None
    history_rows = []
    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_train = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            running_train += loss.item() * batch_x.size(0)

        train_loss = running_train / train_size

        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                running_val += loss.item() * batch_x.size(0)

        val_loss = running_val / val_size
        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
            }
        )

        if verbose:
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}"
            )

        if val_loss < best_val_loss - 1e-7:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch} (best epoch: {best_epoch}).")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    payload = {
        "model_state_dict": best_state_dict if best_state_dict is not None else model.state_dict(),
        "input_mean": torch.from_numpy(input_mean.astype(np.float32)),
        "input_std": torch.from_numpy(input_std.astype(np.float32)),
        "model_config": model.get_config(),
        "mimo_config": {"nt": int(nt), "nr": int(nr), "tp": int(tp)},
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "num_parameters": int(num_params),
        "seed": int(seed),
    }
    torch.save(payload, checkpoint_path)
    if verbose:
        print(f"Final best validation loss: {best_val_loss:.6f} (epoch {best_epoch})")
        print(f"Saved best checkpoint at: {checkpoint_path}")

    if loss_plot_path is not None:
        plot_training_curves(
            train_losses,
            val_losses,
            loss_plot_path,
            title=f"Training vs Validation MSE ({model_type.upper()}, Nt={nt}, Nr={nr}, Tp={tp})",
        )
    if loss_csv_path is not None:
        save_csv(loss_csv_path, history_rows)

    return {
        "model": model,
        "checkpoint_path": str(checkpoint_path),
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "num_parameters": num_params,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "input_mean": input_mean,
        "input_std": input_std,
    }


def load_trained_estimator(
    checkpoint_path: Path,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, torch.Tensor, torch.Tensor, dict]:
    """Load trained PyTorch model and normalization statistics from checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    m_cfg = ckpt["model_config"]
    mimo_cfg = ckpt["mimo_config"]
    model = build_model(
        model_type=m_cfg.get("model_type", "dnn"),
        nr=int(mimo_cfg["nr"]),
        nt=int(mimo_cfg["nt"]),
        tp=int(mimo_cfg["tp"]),
        hidden_dims=tuple(m_cfg.get("hidden_dims", DEFAULT_MODEL.hidden_dims)),
        dropout=float(m_cfg.get("dropout", 0.0)),
        use_layer_norm=bool(m_cfg.get("use_layer_norm", False)),
        residual_skip=bool(m_cfg.get("residual_skip", True)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    input_mean = ckpt["input_mean"].to(device)
    input_std = ckpt["input_std"].to(device)
    return model, input_mean, input_std, ckpt


def predict_mimo_channel(
    model: nn.Module,
    inputs: np.ndarray,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    nr: int,
    nt: int,
    device: Optional[torch.device] = None,
    batch_size: int = 1024,
) -> np.ndarray:
    """Run batched inference and reshape output to complex (B, Nr, Nt)."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    x_np = np.asarray(inputs, dtype=np.float32)
    preds_list = []
    with torch.no_grad():
        for start in range(0, x_np.shape[0], batch_size):
            xb = torch.from_numpy(x_np[start : start + batch_size]).to(device)
            xb = (xb - input_mean) / input_std
            preds_list.append(model(xb).cpu().numpy())
    preds = np.concatenate(preds_list, axis=0)
    half = nr * nt
    h_real = preds[:, :half].reshape(-1, nr, nt)
    h_imag = preds[:, half:].reshape(-1, nr, nt)
    return (h_real + 1j * h_imag).astype(np.complex128)


def main() -> None:
    DEFAULT_PATHS.ensure_dirs()
    parser = argparse.ArgumentParser(description="Train AI Massive MIMO channel estimator.")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_PATHS.default_dataset_path))
    parser.add_argument("--model-type", type=str, default="dnn", choices=["dnn", "cnn"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_MODEL.epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_MODEL.batch_size)
    parser.add_argument("--lr", type=float, default=DEFAULT_MODEL.learning_rate)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=list(DEFAULT_MODEL.hidden_dims))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_MODEL.seed)
    args = parser.parse_args()

    data = load_mimo_dataset(args.dataset)
    nt = int(data["nt"])
    nr = int(data["nr"])
    tp = int(data["tp"])

    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint is not None
        else DEFAULT_PATHS.checkpoints_dir / f"mimo_{args.model_type}_nt{nt}_nr{nr}_tp{tp}.pth"
    )
    loss_plot = DEFAULT_PATHS.plots_dir / f"training_loss_{args.model_type}_nt{nt}_nr{nr}_tp{tp}.png"
    loss_csv = DEFAULT_PATHS.tables_dir / f"training_loss_{args.model_type}_nt{nt}_nr{nr}_tp{tp}.csv"

    train_channel_estimator(
        inputs=data["inputs"],
        targets=data["targets"],
        nr=nr,
        nt=nt,
        tp=tp,
        checkpoint_path=ckpt_path,
        loss_plot_path=loss_plot,
        loss_csv_path=loss_csv,
        model_type=args.model_type,
        hidden_dims=tuple(args.hidden_dims),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
