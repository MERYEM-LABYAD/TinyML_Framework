from typing import Iterable

import torch
import torch.nn as nn


class Recalibrator:
    """
    Recalibration statistique forward-only.

    Aucun optimizer.
    Aucun backward.
    Aucun changement volontaire des poids.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
    ):
        self.model = model
        self.device = device

    def _batch_norm_modules(self):
        return [
            module
            for module in self.model.modules()
            if isinstance(
                module,
                (
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.BatchNorm3d,
                ),
            )
        ]

    def _snapshot_parameters(self):
        return {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
        }

    def recalibrate(
        self,
        data_loader: Iterable,
        max_batches: int = 100,
    ):
        parameter_snapshot = self._snapshot_parameters()

        previous_training = self.model.training

        bn_modules = self._batch_norm_modules()

        for module in bn_modules:
            module.train()

        self.model.train()

        processed = 0

        with torch.no_grad():
            for batch in data_loader:
                self.model(batch)

                processed += 1

                if processed >= max_batches:
                    break

        for module in bn_modules:
            module.eval()

        self.model.train(previous_training)

        for name, parameter in self.model.named_parameters():
            before = parameter_snapshot[name]

            if not torch.equal(before, parameter.detach()):
                raise RuntimeError(
                    f"Parameter changed during recalibration: {name}"
                )

        return {
            "batches": processed,
            "batch_norm_modules": len(bn_modules),
            "parameters_unchanged": True,
        }