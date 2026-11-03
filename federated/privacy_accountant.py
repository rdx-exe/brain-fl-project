"""Privacy accounting for participant-level local training."""

from dataclasses import dataclass

from opacus.accountants import RDPAccountant


@dataclass
class ParticipantPrivacyAccountant:
    noise_multiplier: float
    delta: float

    def __post_init__(self):
        if self.noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be positive.")
        if not 0 < self.delta < 1:
            raise ValueError("delta must be strictly between zero and one.")
        self.accountant = RDPAccountant()
        self.steps = 0
        self.sample_rate = None

    def step(self, sample_rate: float):
        if not 0 < sample_rate <= 1:
            raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate}.")
        self.accountant.step(
            noise_multiplier=self.noise_multiplier,
            sample_rate=sample_rate,
        )
        self.steps += 1
        self.sample_rate = sample_rate

    def epsilon(self) -> float:
        if self.steps == 0:
            return 0.0
        return float(self.accountant.get_epsilon(self.delta))

    def summary(self):
        return {
            "privacy_unit": "participant",
            "epsilon": self.epsilon(),
            "delta": self.delta,
            "noise_multiplier": self.noise_multiplier,
            "steps": self.steps,
            "sample_rate": self.sample_rate,
        }
