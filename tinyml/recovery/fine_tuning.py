from typing import Callable, Iterable

import torch
import torch.nn as nn


class FineTuner:
    """
    Fine-tuning générique.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        device: str = "cpu",
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train(
        self,
        data_loader: Iterable,
        epochs: int = 1,
    ):
        if epochs < 1:
            raise ValueError(
                "epochs must be >= 1."
            )

        history = []

        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            batches = 0

            for batch in data_loader:
                self.optimizer.zero_grad(
                    set_to_none=True
                )

                outputs = self.model(batch)

                loss = self.criterion(
                    outputs,
                    batch,
                )

                if not torch.is_tensor(loss):
                    raise TypeError(
                        "criterion must return a tensor."
                    )

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Non-finite training loss detected."
                    )

                loss.backward()
                self.optimizer.step()

                epoch_loss += float(loss.detach())
                batches += 1

            mean_loss = (
                epoch_loss / batches
                if batches
                else float("nan")
            )

            history.append(
                {
                    "epoch": epoch + 1,
                    "loss": mean_loss,
                    "batches": batches,
                }
            )

        return history