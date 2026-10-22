from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn


class ImportanceMethod(ABC):
    """
    Interface générique pour calculer une importance par canal
    sur les modules structuraux du modèle.
    """

    @abstractmethod
    def score(self, module: nn.Module) -> torch.Tensor:
        """
        Retourne un score 1D de taille module.out_channels.
        Plus le score est faible, moins le canal est important.
        """
        raise NotImplementedError

    def validate_score(
        self,
        module: nn.Module,
        scores: torch.Tensor,
    ) -> None:
        if not isinstance(scores, torch.Tensor):
            raise TypeError("Importance score must be a torch.Tensor.")

        if scores.ndim != 1:
            raise ValueError(
                f"Importance score must be 1D, got shape {tuple(scores.shape)}."
            )

        if not hasattr(module, "out_channels"):
            raise TypeError(
                f"Module {module.__class__.__name__} has no out_channels."
            )

        if scores.numel() != module.out_channels:
            raise ValueError(
                f"Score size {scores.numel()} does not match "
                f"out_channels={module.out_channels}."
            )

        if not torch.isfinite(scores).all():
            raise ValueError("Importance scores contain NaN or Inf.")

    def __call__(self, module: nn.Module) -> torch.Tensor:
        scores = self.score(module)
        self.validate_score(module, scores)
        return scores.detach().float().cpu()