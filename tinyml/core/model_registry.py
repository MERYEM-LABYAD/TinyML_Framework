import os
from typing import Dict

import torch


class ModelRegistry:

    def __init__(self, directory):
        self.directory = directory
        os.makedirs(
            directory,
            exist_ok=True,
        )

    def path(self, stage):
        return os.path.join(
            self.directory,
            f"{stage}.pth",
        )

    def save(
        self,
        stage,
        model,
        metadata=None,
    ):
        path = self.path(stage)

        torch.save(
            {
                "stage": stage,
                "state_dict": model.state_dict(),
                "metadata": metadata or {},
            },
            path,
        )

        return path

    def load(
        self,
        stage,
        model,
        map_location="cpu",
        strict=True,
    ):
        path = self.path(stage)

        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["state_dict"],
            strict=strict,
        )

        return checkpoint