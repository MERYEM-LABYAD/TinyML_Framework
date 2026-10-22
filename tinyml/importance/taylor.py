from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn


class TaylorImportance:
    """
    First-order Taylor channel importance.

    For each channel c:

        I_c = sum(|W_c * dL/dW_c|)

    The accumulation is performed over calibration batches.
    No optimizer update is performed.

    Architecture-agnostic:
    this class only requires a PyTorch model and a callable loss function.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device

        self.scores: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _channel_score(
        parameter: torch.Tensor,
        gradient: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reduce |W * grad| to one score per output channel.

        Conv2d:
            [out_channels, in_channels, kH, kW]
            -> [out_channels]

        Linear:
            [out_features, in_features]
            -> [out_features]

        Generic tensors:
            first dimension is treated as the channel/output dimension.
        """

        contribution = (
            parameter.detach() * gradient.detach()
        ).abs()

        if contribution.ndim == 1:
            return contribution

        return contribution.reshape(
            contribution.shape[0],
            -1,
        ).sum(dim=1)

    def reset(self) -> None:
        self.scores.clear()

    def accumulate_batch(self, batch) -> None:
        """
        Compute one task-loss gradient pass.

        No optimizer step is performed.
        """

        self.model.zero_grad(
            set_to_none=True
        )

        if self.device is not None:
            batch = self._move_to_device(
                batch,
                self.device,
            )

        loss = self.loss_fn(
            self.model,
            batch,
        )

        if not torch.is_tensor(loss):
            raise TypeError(
                "loss_fn must return a torch.Tensor."
            )

        if loss.ndim != 0:
            loss = loss.mean()

        loss.backward()

        for name, module in self.model.named_modules():

            if not isinstance(
                module,
                (nn.Conv2d, nn.Linear),
            ):
                continue

            weight = module.weight

            if weight.grad is None:
                continue

            score = self._channel_score(
                weight,
                weight.grad,
            )

            if not torch.isfinite(score).all():
                raise RuntimeError(
                    f"Non-finite Taylor importance "
                    f"detected in module: {name}"
                )

            if name not in self.scores:
                self.scores[name] = (
                    score.detach().cpu()
                )
            else:
                self.scores[name] += (
                    score.detach().cpu()
                )

    def compute(
        self,
        calibration_batches: Iterable,
        max_batches: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Accumulate Taylor importance over calibration data.

        The model weights are never updated.
        """

        self.reset()

        was_training = self.model.training
        self.model.train()

        try:
            for batch_index, batch in enumerate(
                calibration_batches
            ):
                if (
                    max_batches is not None
                    and batch_index >= max_batches
                ):
                    break

                self.accumulate_batch(batch)

        finally:
            self.model.zero_grad(
                set_to_none=True
            )

            self.model.train(
                was_training
            )

        if not self.scores:
            raise RuntimeError(
                "Taylor importance produced no scores. "
                "Check the calibration data and loss function."
            )

        return {
            name: score.clone()
            for name, score in self.scores.items()
        }

    def get(
        self,
        module_name: str,
    ) -> torch.Tensor:
        if module_name not in self.scores:
            raise KeyError(
                f"No Taylor score available for "
                f"module '{module_name}'."
            )

        return self.scores[module_name]

    @staticmethod
    def _move_to_device(
        value,
        device: torch.device,
    ):
        if torch.is_tensor(value):
            return value.to(device)

        if isinstance(value, dict):
            return {
                key: TaylorImportance._move_to_device(
                    item,
                    device,
                )
                for key, item in value.items()
            }

        if isinstance(value, list):
            return [
                TaylorImportance._move_to_device(
                    item,
                    device,
                )
                for item in value
            ]

        if isinstance(value, tuple):
            return tuple(
                TaylorImportance._move_to_device(
                    item,
                    device,
                )
                for item in value
            )

        return value