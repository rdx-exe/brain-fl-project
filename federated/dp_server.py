"""Opt-in Flower server wiring for participant-level DP clients."""

import logging
import sys
from pathlib import Path

import flwr as fl
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from federated.client import _state_to_parameters
from federated.dp_client import build_private_model, client_fn
from federated.dp_config import DPConfig
from federated.federated_config import FederatedConfig
from federated.server import FederatedServer


LOGGER = logging.getLogger(__name__)


class DPSliceEvaluationModel(nn.Module):
    """Use the DP GroupNorm backbone with the existing slice evaluator."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = build_private_model(num_classes).backbone

    def forward(self, images):
        if images.ndim != 4:
            raise ValueError("Expected slice images with shape (B, C, H, W).")
        return self.backbone(images)


def start_dp_federated_simulation(config: FederatedConfig, dp_config: DPConfig):
    """Start DP FedAvg only after the privacy configuration is reviewed."""
    dp_config.validate()
    config.validate()
    server = FederatedServer(config, model=DPSliceEvaluationModel(config.num_classes))
    initial_parameters = fl.common.ndarrays_to_parameters(_state_to_parameters(server.model))
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=config.num_clients,
        min_available_clients=config.num_clients,
        evaluate_fn=server.evaluate,
        initial_parameters=initial_parameters,
        on_fit_config_fn=lambda server_round: {"server_round": server_round},
    )
    LOGGER.warning(
        "Starting participant-level DP FedAvg: epsilon is local-update scoped; "
        "Flower FedAvg does not provide secure aggregation."
    )
    return fl.simulation.start_simulation(
        client_fn=lambda client_id: client_fn(client_id, config, dp_config),
        num_clients=config.num_clients,
        config=fl.server.ServerConfig(num_rounds=config.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 1 if server.device.type == "cuda" else 0},
        ray_init_args={"include_dashboard": False, "ignore_reinit_error": True},
    )
