from typing import Iterable

import torch


class CalibrationRunner:

    def __init__(
        self,
        device: str = "cpu",
        max_batches: int = 100,
    ):
        self.device = device
        self.max_batches = max_batches

    def run(
        self,
        model,
        data_loader: Iterable,
    ):
        model.eval()

        processed = 0

        with torch.no_grad():
            for batch in data_loader:
                model(batch)

                processed += 1

                if processed >= self.max_batches:
                    break

        if processed == 0:
            raise RuntimeError(
                "Calibration data loader produced no batches."
            )

        return {
            "calibration_batches": processed,
        }