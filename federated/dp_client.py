"""Flower client with participant-level local DP-SGD via Opacus.

Each dataset item is one participant containing all seven slices. The loss is
computed after averaging the seven slice logits, so Opacus clips and noises one
per-participant gradient rather than seven independent slice gradients.
"""

import csv
import logging
import sys
from types import MethodType
from pathlib import Path

import flwr as fl
import pandas as pd
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision.models.resnet import BasicBlock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from federated.client import (
    ClientSliceDataset,
    _parameters_to_state,
    _state_to_parameters,
)
from federated.dp_config import DPConfig
from federated.federated_config import FederatedConfig, get_device, set_seed
from federated.privacy_accountant import ParticipantPrivacyAccountant
from models.resnet18_model import build_resnet18


LOGGER = logging.getLogger(__name__)
EXPECTED_SLICES = 7


def _safe_basic_block_forward(self, x):
    identity = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    if self.downsample is not None:
        identity = self.downsample(x)
    out = out + identity
    out = self.relu(out)
    return out


class ParticipantDataset(Dataset):
    """Group the existing seven preprocessed slices into one participant item."""

    def __init__(self, client_id: str, config: FederatedConfig):
        self.base = ClientSliceDataset(client_id, config)
        self.report = self.base.report.reset_index(drop=True)
        self.selected_slices = list(self.base.selected_slices)
        if len(self.selected_slices) != EXPECTED_SLICES:
            raise ValueError("Participant-level DP requires exactly seven slices per participant.")

    def __len__(self):
        return len(self.report)

    def __getitem__(self, index):
        record = self.report.iloc[index]
        volume = self.base.get_volume(record["image_path"])
        images = torch.stack([
            self.base.prepare_slice(volume[:, :, slice_index])
            for slice_index in self.selected_slices
        ])
        return images, torch.tensor(int(record["label"]), dtype=torch.long), str(record["participant_id"])


class ParticipantResNet18(nn.Module):
    """ResNet-18 with seven-slice participant aggregation in the forward pass."""

    def __init__(self, num_classes: int):
        super().__init__()
        base = build_resnet18(num_classes=num_classes, pretrained=False)
        self.backbone = base.backbone
        self.num_classes = num_classes

    def forward(self, participant_images):
        if participant_images.ndim != 5 or participant_images.shape[1] != EXPECTED_SLICES:
            raise ValueError("Expected participant images with shape (B, 7, C, H, W).")
        logits = torch.stack(
            [self.backbone(participant_images[:, slice_index]) for slice_index in range(EXPECTED_SLICES)],
            dim=1,
        )
        return logits.mean(dim=1)


def build_private_model(num_classes: int):
    model = ParticipantResNet18(num_classes=num_classes)
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        model = ModuleValidator.fix(model)
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
        if isinstance(module, BasicBlock):
            module.forward = MethodType(_safe_basic_block_forward, module)
    remaining_errors = ModuleValidator.validate(model, strict=False)
    if remaining_errors:
        raise ValueError(f"Opacus model validation failed: {remaining_errors}")
    return model


class ParticipantDPFlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id: str, config: FederatedConfig, dp_config: DPConfig):
        dp_config.validate()
        self.client_id = client_id
        self.config = config
        self.dp_config = dp_config
        set_seed(config.seed + int(client_id))
        self.device = get_device()
        self.model = build_private_model(config.num_classes).to(self.device)
        self.dataset = ParticipantDataset(client_id, config)
        self.loader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        self.accountant = ParticipantPrivacyAccountant(
            noise_multiplier=dp_config.noise_multiplier,
            delta=dp_config.delta,
        )
        LOGGER.info("DP client %s initialized with %d participant privacy units", client_id, len(self.dataset))

    def get_parameters(self, config):
        return _state_to_parameters(self.model)

    def fit(self, parameters, config):
        _parameters_to_state(self.model, parameters)
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        privacy_engine = PrivacyEngine(accountant="rdp")
        private_model, private_optimizer, private_loader = privacy_engine.make_private(
            module=self.model,
            optimizer=optimizer,
            data_loader=self.loader,
            noise_multiplier=self.dp_config.noise_multiplier,
            max_grad_norm=self.dp_config.max_grad_norm,
            poisson_sampling=self.dp_config.poisson_sampling,
        )
        sample_rate = min(1.0, self.config.batch_size / max(len(self.dataset), 1))
        total_loss = 0.0
        total_units = 0
        correct = 0
        private_model.train()
        for _ in range(self.config.local_epochs):
            for participant_images, labels, _ in private_loader:
                participant_images = participant_images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                private_optimizer.zero_grad(set_to_none=True)
                logits = private_model(participant_images)
                loss = criterion(logits, labels)
                loss.backward()
                private_optimizer.step()
                self.accountant.step(sample_rate)
                total_loss += loss.item() * labels.size(0)
                total_units += labels.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()
        if total_units == 0:
            raise RuntimeError(f"DP client {self.client_id} has no participant batches.")
        self.model = private_model._module
        privacy = self.accountant.summary()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = self.config.output_dir / f"client_{int(self.client_id) + 1}_dp_metrics.csv"
        row = {
            "round": int(config.get("server_round", 0)),
            "client_id": f"client_{int(self.client_id) + 1}",
            "privacy_unit": "participant",
            "participants": len(self.dataset),
            "slices": len(self.dataset) * EXPECTED_SLICES,
            "train_loss": total_loss / total_units,
            "train_accuracy": correct / total_units,
            **privacy,
        }
        pd.DataFrame([row]).to_csv(metrics_path, mode="a", header=not metrics_path.exists(), index=False)
        LOGGER.info("DP client %s: epsilon=%.4f delta=%g steps=%d", self.client_id, privacy["epsilon"], privacy["delta"], privacy["steps"])
        return _state_to_parameters(self.model), len(self.dataset), {
            "train_loss": float(row["train_loss"]),
            "train_accuracy": float(row["train_accuracy"]),
            "participants": len(self.dataset),
            "epsilon": privacy["epsilon"],
            "delta": privacy["delta"],
        }

    def evaluate(self, parameters, config):
        _parameters_to_state(self.model, parameters)
        return 0.0, len(self.dataset), {"privacy_unit": "participant"}


def client_fn(client_id: str, config: FederatedConfig, dp_config: DPConfig):
    return ParticipantDPFlowerClient(client_id, config, dp_config).to_client()
