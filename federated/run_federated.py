"""Command-line entry point for the four-client OASIS-1 FedAvg experiment."""

import argparse
import sys
import pandas as pd

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from federated.federated_config import FederatedConfig
from federated.server import start_federated_simulation
from federated.evaluate import evaluate


def consolidate_client_metrics(output_dir):
    paths = sorted(output_dir.glob("client_*_training_metrics.csv"))
    if not paths:
        raise FileNotFoundError("No per-client training metrics were produced.")
    pd.concat([pd.read_csv(path) for path in paths], ignore_index=True).sort_values(
        ["round", "client_id"]
    ).to_csv(output_dir / "client_metrics.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--partition-dir", type=Path, default=None)
    args = parser.parse_args()
    config = FederatedConfig(
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        partition_dir=args.partition_dir or FederatedConfig().partition_dir,
        output_dir=args.output_dir or FederatedConfig().output_dir,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    for path in (config.output_dir / "round_history.csv", config.output_dir / "client_metrics.csv"):
        path.unlink(missing_ok=True)
    for path in config.output_dir.glob("client_*_training_metrics.csv"):
        path.unlink()
    start_federated_simulation(config)
    consolidate_client_metrics(config.output_dir)
    evaluate(config.output_dir)


if __name__ == "__main__":
    main()