from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    name: str = "tinyml_experiment"
    pruning_ratio: float = 0.10
    min_channels: int = 8
    distillation_alpha: float = 0.5
    device: str = "cpu"
    checkpoint_directory: str = "checkpoints"