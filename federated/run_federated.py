"""Command-line entry point for the four-client OASIS-1 FedAvg experiment."""

import argparse
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from federated.federated_config import FederatedConfig
from federated.server import start_federated_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    config = FederatedConfig(
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=args.output_dir or FederatedConfig().output_dir,
    )
    start_federated_simulation(config)


if __name__ == "__main__":
    main()