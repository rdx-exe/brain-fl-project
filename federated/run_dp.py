"""Opt-in entry point for participant-level DP FedAvg.

This command is intentionally separate from ``run_federated.py``. Review
PRIVACY_METHODOLOGY.md and dp_experiment_config.json before running it.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from federated.dp_config import DPConfig
from federated.dp_server import start_dp_federated_simulation
from federated.federated_config import FederatedConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-multiplier", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--partition-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "federated" / "dp")
    args = parser.parse_args()
    config = FederatedConfig(
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        partition_dir=args.partition_dir or FederatedConfig().partition_dir,
        output_dir=args.output_dir,
    )
    dp_config = DPConfig(
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.max_grad_norm,
        delta=args.delta,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "dp_experiment_config.json").write_text(json.dumps({
        **{key: str(value) if isinstance(value, Path) else value for key, value in config.__dict__.items()},
        **dp_config.__dict__,
        "aggregation": "FedAvg",
        "final_privacy_experiment_reviewed": False,
    }, indent=2) + "\n")
    start_dp_federated_simulation(config, dp_config)


if __name__ == "__main__":
    main()
