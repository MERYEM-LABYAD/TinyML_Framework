from typing import Dict

import torch
import torch.nn as nn

from .base import ImportanceMethod


class ActivationImportance(ImportanceMethod):
    """
    Importance basée sur l'activation moyenne absolue
    observée sur les canaux de sortie.
    """

    def __init__(self):
        self._sum = None
        self._count = 0
        self._hook = None

    def _hook_fn(self, module, inputs, output):
        if not isinstance(output, torch.Tensor):
            return

        values = output.detach().float()

        if values.ndim < 2:
            return

        reduce_dims = tuple(
            index for index in range(values.ndim)
            if index != 1
        )

        batch_scores = values.abs().mean(dim=reduce_dims)

        if self._sum is None:
            self._sum = torch.zeros_like(batch_scores.cpu())

        self._sum += batch_scores.cpu()
        self._count += 1

    def collect(
        self,
        module: nn.Module,
        forward_fn,
    ) -> torch.Tensor:
        self._sum = None
        self._count = 0

        self._hook = module.register_forward_hook(self._hook_fn)

        try:
            with torch.no_grad():
                forward_fn()
        finally:
            self._hook.remove()
            self._hook = None

        if self._sum is None or self._count == 0:
            raise RuntimeError(
                "No activation was collected for the selected module."
            )

        scores = self._sum / self._count

        self.validate_score(module, scores)

        return scores

    def score(self, module: nn.Module) -> torch.Tensor:
        raise RuntimeError(
            "ActivationImportance requires runtime data. "
            "Use collect(module, forward_fn)."
        )