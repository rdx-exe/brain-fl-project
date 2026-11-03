"""Configuration for participant-level local differential privacy."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DPConfig:
    enabled: bool = True
    privacy_unit: str = "participant"
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    delta: float = 1e-5
    poisson_sampling: bool = True

    def validate(self):
        if not self.enabled:
            raise ValueError("DPConfig must be enabled for the DP client.")
        if self.privacy_unit != "participant":
            raise ValueError("Only participant-level privacy is supported.")
        if self.noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be greater than zero.")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be greater than zero.")
        if not 0 < self.delta < 1:
            raise ValueError("delta must be strictly between zero and one.")
        if not self.poisson_sampling:
            raise ValueError("Participant accounting requires Poisson sampling.")
