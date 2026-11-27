"""Publication-quality plotting and CSV export utilities."""

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_csv(
    filepath: Union[str, Path],
    rows: Sequence[Dict[str, object]],
    fieldnames: Sequence[str] = None,
) -> Path:
    """Save list of metric dicts to CSV."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot save empty rows to CSV.")
    cols = list(fieldnames) if fieldnames is not None else list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _configure_plot_style() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "lines.linewidth": 2.0,
        "lines.markersize": 6.5,
        "figure.dpi": 150,
    })


def plot_training_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    output_path: Union[str, Path],
    title: str = "Training vs Validation MSE Loss",
) -> Path:
    _configure_plot_style()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="Train Loss (MSE)")
    ax.plot(epochs, val_losses, linestyle="--", label="Validation Loss (MSE)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_metric_vs_x(
    x_values: Sequence[float],
    series: Dict[str, Sequence[float]],
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Union[str, Path],
    logy: bool = False,
    markers: Sequence[str] = ("o", "s", "^", "D", "v"),
) -> Path:
    _configure_plot_style()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for idx, (label, y_vals) in enumerate(series.items()):
        marker = markers[idx % len(markers)]
        if logy:
            # Guard zero BER entries so semilogy remains clean
            safe_y = [max(float(v), 1e-7) for v in y_vals]
            ax.semilogy(x_values, safe_y, marker=marker, label=label)
        else:
            ax.plot(x_values, y_vals, marker=marker, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_inference_time_bar(
    labels: Sequence[str],
    times_ms: Sequence[float],
    output_path: Union[str, Path],
    title: str = "Inference Time Comparison (Batch Evaluation)",
    ylabel: str = "Inference Time (ms)",
) -> Path:
    _configure_plot_style()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, times_ms, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"][: len(labels)])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    for bar, val in zip(bars, times_ms):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{val:.2f} ms",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path
