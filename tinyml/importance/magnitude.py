import torch
import torch.nn as nn

from .base import ImportanceMethod


class MagnitudeImportance(ImportanceMethod):
    """
    Importance L1 par canal de sortie.

    Pour une Conv2d :
        score[c] = somme(|W[c]|)

    Plus le score est faible, moins le canal contribue
    en magnitude absolue.
    """

    def score(self, module: nn.Module) -> torch.Tensor:
        if not isinstance(module, nn.Conv2d):
            raise TypeError(
                "MagnitudeImportance currently supports nn.Conv2d only."
            )

        weight = module.weight.detach().float()

        if weight.ndim != 4:
            raise ValueError(
                f"Expected Conv2d weights with 4 dimensions, "
                f"got {weight.ndim}."
            )

        return weight.abs().flatten(1).sum(dim=1)