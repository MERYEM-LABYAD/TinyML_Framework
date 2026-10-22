from typing import Dict

import torch
import torch.nn as nn


class PruningValidator:
    """
    Validation structurelle après pruning.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def check_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.model.parameters()
        )

    def check_finite_parameters(self) -> None:
        for name, parameter in self.model.named_parameters():
            if not torch.isfinite(parameter.detach()).all():
                raise ValueError(
                    f"NaN or Inf detected in parameter: {name}"
                )

    def check_shapes(self) -> None:
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                if module.in_channels < 1:
                    raise ValueError(
                        f"Invalid Conv2d input channels: {name}"
                    )

                if module.out_channels < 1:
                    raise ValueError(
                        f"Invalid Conv2d output channels: {name}"
                    )

    def forward_check(self, forward_fn):
        self.model.eval()

        with torch.no_grad():
            output = forward_fn()

        if output is None:
            raise RuntimeError(
                "Forward returned None."
            )

        return output

    def validate(self, forward_fn) -> Dict[str, object]:
        self.check_shapes()
        self.check_finite_parameters()

        parameters = self.check_parameters()

        output = self.forward_check(forward_fn)

        return {
            "parameters": parameters,
            "forward": True,
            "output_type": type(output).__name__,
        }