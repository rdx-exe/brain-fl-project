"""Configuration and reproducibility helpers for federated experiments."""

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FederatedConfig:
    num_clients: int = 4
    num_rounds: int = 5
    local_epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42
    image_size: int = 224
    num_classes: int = 2
    pretrained: bool = False
    num_workers: int = 0
    partition_dir: Path = PROJECT_ROOT / "data" / "partitions" / "federated"
    participant_manifest: Path = PROJECT_ROOT / "data" / "partitions" / "participant_splits.csv"
    extraction_report: Path = PROJECT_ROOT / "results" / "quality_control" / "extraction_report.csv"
    output_dir: Path = PROJECT_ROOT / "results" / "federated" / "fedavg"

    def validate(self):
        if self.num_clients != 4:
            raise ValueError("This experiment requires exactly four clients.")
        if self.num_rounds <= 0 or self.local_epochs <= 0:
            raise ValueError("Communication rounds and local epochs must be positive.")
        if self.batch_size <= 0 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Batch size and learning rate must be positive; weight decay cannot be negative.")


def set_seed(seed: int):
    """Set deterministic seeds for Python, NumPy, PyTorch, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")