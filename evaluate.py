"""Master evaluation runner for Massive MIMO and 6G XL-MIMO experiments."""

import argparse
from pathlib import Path
from typing import Optional

from config.config import DEFAULT_MIMO, DEFAULT_PATHS
from experiments.antenna_experiment import run_antenna_experiment
from experiments.generalization_experiment import (
    run_cross_antenna_generalization_experiment,
    run_snr_generalization_experiment,
)
from experiments.pilot_experiment import run_pilot_experiment
from experiments.snr_experiment import run_snr_experiment
from experiments.xl_mimo_experiment import run_xl_mimo_experiment


def main() -> None:
    DEFAULT_PATHS.ensure_dirs()
    parser = argparse.ArgumentParser(description="Run Massive MIMO & 6G XL-MIMO evaluations.")
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["all", "snr", "antenna", "pilot", "generalization", "xl_mimo"],
        help="Which experiment suite to run.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(
            DEFAULT_PATHS.checkpoints_dir
            / f"mimo_dnn_nt{DEFAULT_MIMO.nt}_nr{DEFAULT_MIMO.nr}_tp{DEFAULT_MIMO.tp}.pth"
        ),
        help="Trained baseline Massive MIMO checkpoint path.",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if args.experiment in ("all", "snr"):
        run_snr_experiment(checkpoint_path=ckpt_path)
    if args.experiment in ("all", "antenna"):
        run_antenna_experiment()
    if args.experiment in ("all", "generalization"):
        run_snr_generalization_experiment()
        run_cross_antenna_generalization_experiment()
    if args.experiment in ("all", "pilot"):
        run_pilot_experiment()
    if args.experiment in ("all", "xl_mimo"):
        run_xl_mimo_experiment()

    print("\nAll selected evaluations completed! Plots and CSV tables saved under results/.")


if __name__ == "__main__":
    main()
