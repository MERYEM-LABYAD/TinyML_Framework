import torch
import torch.nn as nn

from .base import ImportanceMethod


class HybridImportance(ImportanceMethod):
    """
    Combinaison pondérée de plusieurs méthodes.

    Chaque score est normalisé par sa moyenne avant combinaison.
    """

    def __init__(
        self,
        methods,
        weights=None,
    ):
        if not methods:
            raise ValueError("At least one importance method is required.")

        self.methods = methods

        if weights is None:
            weights = [1.0] * len(methods)

        if len(weights) != len(methods):
            raise ValueError(
                "weights and methods must have the same length."
            )

        if sum(weights) <= 0:
            raise ValueError("Importance weights must sum to > 0.")

        self.weights = weights

    def score(self, module: nn.Module) -> torch.Tensor:
        combined = None

        for method, weight in zip(self.methods, self.weights):
            scores = method(module).float()

            mean = scores.mean().clamp_min(1e-12)
            normalized = scores / mean

            if combined is None:
                combined = weight * normalized
            else:
                combined = combined + weight * normalized

        return combined / sum(self.weights)