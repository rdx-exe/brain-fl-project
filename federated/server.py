"""FedAvg simulation server with centralized participant-level validation."""

import logging
import json
import sys
from pathlib import Path

import flwr as fl
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from federated.client import _parameters_to_state, _state_to_parameters
from federated.federated_config import FederatedConfig, get_device, set_seed
from models.resnet18_model import build_resnet18
from preprocessing.pytorch_dataset import OASISSliceDataset, SELECTED_SLICES
from experiments.train_centralized import validate_model


LOGGER = logging.getLogger(__name__)


class FederatedServer:
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.config.validate()
        set_seed(config.seed)
        self.device = get_device()
        self.model = build_resnet18(num_classes=config.num_classes, pretrained=config.pretrained).to(self.device)
        validation_dataset = OASISSliceDataset(
            report_path=config.extraction_report,
            split="validation",
            selected_slices=SELECTED_SLICES,
            image_size=config.image_size,
            cache_size=2,
        )
        self.validation_loader = DataLoader(
            validation_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        self.output_dir = config.output_dir
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.output_dir / "round_history.csv"
        self.client_metrics_path = self.output_dir / "client_metrics.csv"
        self._validate_client_manifests()
        self._write_config()
        self.best_loss = float("inf")

    def _validate_client_manifests(self):
        client_sets = []
        for client_number in range(1, self.config.num_clients + 1):
            path = self.config.partition_dir / f"client_{client_number}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing client partition: {path}")
            frame = pd.read_csv(path)
            if set(frame["split"].astype(str)) != {"train"}:
                raise ValueError(f"Client {client_number} contains non-training participants.")
            if frame["participant_id"].duplicated().any():
                raise ValueError(f"Client {client_number} contains duplicate participants.")
            client_sets.append(set(frame["participant_id"].astype(str)))
            LOGGER.info("Client %s initialized: %d participants, %d slices", client_number, len(frame), len(frame) * 7)
        overlap = set().union(*(client_sets[i] & client_sets[j] for i in range(4) for j in range(i + 1, 4)))
        participant_manifest = pd.read_csv(self.config.participant_manifest)
        train_ids = set(participant_manifest.loc[participant_manifest["split"] == "train", "participant_id"].astype(str))
        assigned = set().union(*client_sets)
        if overlap or assigned != train_ids:
            raise ValueError(f"Client partition validation failed: overlap={len(overlap)}, missing={len(train_ids - assigned)}, extra={len(assigned - train_ids)}")
        LOGGER.info("Client partition validation passed: 164 unique training participants, no leakage")

    def _write_config(self):
        config = {key: str(value) if isinstance(value, Path) else value for key, value in self.config.__dict__.items()}
        config.update({"aggregation": "FedAvg", "flower_version": fl.__version__, "test_used_during_training": False})
        (self.output_dir / "experiment_config.json").write_text(json.dumps(config, indent=2) + "\n")

    def evaluate(self, server_round: int, parameters, _config):
        _parameters_to_state(self.model, parameters)
        metrics = validate_model(self.model, self.validation_loader, self.device)
        checkpoint = {
            "round": server_round,
            "model_state_dict": self.model.state_dict(),
            "validation_metrics": metrics,
            "model_name": "ResNet18",
            "num_classes": self.config.num_classes,
            "pretrained": self.config.pretrained,
            "random_seed": self.config.seed,
            "federated_config": self.config.__dict__,
        }
        checkpoint_path = self.checkpoint_dir / f"round_{server_round:03d}.pth"
        torch.save(checkpoint, checkpoint_path)
        if metrics["val_loss"] < self.best_loss:
            self.best_loss = metrics["val_loss"]
            torch.save(checkpoint, self.checkpoint_dir / "best_model.pth")
            torch.save(checkpoint, self.output_dir / "best_global_model.pth")
        row = {"round": server_round, **metrics, "checkpoint": str(checkpoint_path)}
        pd.DataFrame([row]).to_csv(
            self.history_path,
            mode="a",
            header=not self.history_path.exists(),
            index=False,
        )
        LOGGER.info("Round %s global validation: loss=%.5f accuracy=%.4f auc=%.4f", server_round, metrics["val_loss"], metrics["val_accuracy"], metrics["val_auc"])
        return metrics["val_loss"], {"val_accuracy": metrics["val_accuracy"], "val_auc": metrics["val_auc"], "participants": metrics["participants"]}


def start_federated_simulation(config: FederatedConfig):
    """Run ordinary synchronous FedAvg with four simulated clients."""
    config.validate()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    server = FederatedServer(config)

    def make_client(client_id: str):
        from federated.client import client_fn
        return client_fn(client_id, config)

    initial_parameters = fl.common.ndarrays_to_parameters(_state_to_parameters(server.model))
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=config.num_clients,
        min_available_clients=config.num_clients,
        evaluate_fn=server.evaluate,
        initial_parameters=initial_parameters,
        on_fit_config_fn=lambda server_round: {"server_round": server_round},
        fit_metrics_aggregation_fn=lambda metrics: {
            "train_loss": sum(count * values["train_loss"] for count, values in metrics) / sum(count for count, _ in metrics),
            "participants": sum(values["participants"] for _, values in metrics),
        },
    )
    return fl.simulation.start_simulation(
        client_fn=make_client,
        num_clients=config.num_clients,
        config=fl.server.ServerConfig(num_rounds=config.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 1 if server.device.type == "cuda" else 0},
        ray_init_args={"include_dashboard": False, "ignore_reinit_error": True},
    )