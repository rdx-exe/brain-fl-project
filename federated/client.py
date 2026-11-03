"""Flower client implementation for one participant-level OASIS partition."""

import logging
import sys
from pathlib import Path

import flwr as fl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.resnet18_model import build_resnet18
from preprocessing.pytorch_dataset import OASISSliceDataset, SELECTED_SLICES
from federated.federated_config import FederatedConfig, get_device, set_seed


LOGGER = logging.getLogger(__name__)


class ClientSliceDataset(OASISSliceDataset):
    """Reuse centralized preprocessing after restricting to one client."""

    def __init__(self, client_id: str, config: FederatedConfig):
        client_path = config.partition_dir / f"client_{int(client_id) + 1}.csv"
        if not client_path.exists():
            raise FileNotFoundError(f"Client partition not found: {client_path}")
        client_manifest = pd.read_csv(client_path)
        required = {"participant_id", "label", "split"}
        missing = required - set(client_manifest.columns)
        if missing:
            raise ValueError(f"Client manifest is missing columns: {sorted(missing)}")
        if set(client_manifest["split"]) != {"train"}:
            raise ValueError(f"Client {client_id} contains non-training records.")
        participant_ids = set(client_manifest["participant_id"].astype(str))

        super().__init__(
            report_path=config.extraction_report,
            split="train",
            selected_slices=SELECTED_SLICES,
            image_size=config.image_size,
            cache_size=2,
        )
        self.report = self.report[self.report["participant_id"].isin(participant_ids)].copy()
        if set(self.report["participant_id"]) != participant_ids:
            missing_ids = participant_ids - set(self.report["participant_id"])
            raise ValueError(f"Client {client_id} IDs missing from extraction report: {sorted(missing_ids)}")
        expected_labels = client_manifest.set_index("participant_id")["label"].astype(int)
        actual_labels = self.report.set_index("participant_id")["label"].astype(int)
        if not expected_labels.sort_index().equals(actual_labels.sort_index()):
            raise ValueError(f"Client {client_id} labels do not match the extraction report.")
        self.samples = [
            {
                "participant_id": record["participant_id"],
                "session_id": record["session_id"],
                "label": int(record["label"]),
                "image_path": record["image_path"],
                "slice_index": int(slice_index),
            }
            for _, record in self.report.iterrows()
            for slice_index in self.selected_slices
        ]
        if len(self.samples) != len(participant_ids) * len(self.selected_slices):
            raise ValueError(f"Client {client_id} does not contain seven slices per participant.")
        self.client_id = client_id


def _state_to_parameters(model):
    return [value.detach().cpu().numpy() for value in model.state_dict().values()]


def _parameters_to_state(model, parameters):
    state = model.state_dict()
    if len(parameters) != len(state):
        raise ValueError(f"Expected {len(state)} model tensors, received {len(parameters)}.")
    converted = {
        key: torch.as_tensor(value, dtype=state[key].dtype)
        for key, value in zip(state.keys(), parameters)
    }
    model.load_state_dict(converted, strict=True)


class OASISFlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id: str, config: FederatedConfig):
        self.client_id = client_id
        self.config = config
        set_seed(config.seed + int(client_id))
        self.device = get_device()
        self.model = build_resnet18(num_classes=config.num_classes, pretrained=config.pretrained).to(self.device)
        self.dataset = ClientSliceDataset(client_id, config)
        self.loader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        LOGGER.info("Client %s initialized with %d participants and %d slices on %s", client_id, len(self.dataset.report), len(self.dataset), self.device)

    def get_parameters(self, config):
        return _state_to_parameters(self.model)

    def fit(self, parameters, config):
        _parameters_to_state(self.model, parameters)
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        total_loss = 0.0
        total_samples = 0
        correct = 0
        for _ in range(self.config.local_epochs):
            for images, labels, _ in self.loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss = criterion(logits, labels)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)
        if total_samples == 0:
            raise RuntimeError(f"Client {self.client_id} has an empty training loader.")
        train_loss = total_loss / total_samples
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = self.config.output_dir / f"client_{int(self.client_id) + 1}_training_metrics.csv"
        train_accuracy = correct / total_samples
        metrics_row = pd.DataFrame([{
            "round": int(config.get("server_round", 0)),
            "client_id": f"client_{int(self.client_id) + 1}",
            "train_loss": float(train_loss),
            "train_accuracy": float(train_accuracy),
            "participants": len(self.dataset.report),
            "slices": len(self.dataset),
        }])
        metrics_row.to_csv(metrics_path, mode="a", header=not metrics_path.exists(), index=False)
        LOGGER.info("Client %s completed local training: loss=%.5f", self.client_id, train_loss)
        return _state_to_parameters(self.model), len(self.dataset), {"train_loss": float(train_loss), "train_accuracy": float(train_accuracy), "participants": len(self.dataset.report)}

    def evaluate(self, parameters, config):
        _parameters_to_state(self.model, parameters)
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_samples = 0
        correct = 0
        with torch.no_grad():
            for images, labels, _ in self.loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                logits = self.model(images)
                total_loss += criterion(logits, labels).item() * images.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total_samples += images.size(0)
        if total_samples == 0:
            raise RuntimeError(f"Client {self.client_id} has no samples for evaluation.")
        return float(total_loss / total_samples), len(self.dataset), {"accuracy": float(correct / total_samples)}


def client_fn(client_id: str, config: FederatedConfig):
    """Create a Flower Client for simulation; ``client_id`` is 0-based."""
    return OASISFlowerClient(client_id, config).to_client()