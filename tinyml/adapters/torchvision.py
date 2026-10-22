from typing import Any, Dict, Iterable

from .base import BaseAdapter


class TorchVisionAdapter(BaseAdapter):

    @property
    def framework_name(self) -> str:
        return "TorchVision"

    @property
    def architecture_name(self) -> str:
        return self.model.__class__.__name__

    def iter_modules(self) -> Iterable:
        return self.model.named_modules()

    def count_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.model.parameters()
        )

    def count_trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()

        statistics = {}

        for _, module in self.model.named_modules():
            name = module.__class__.__name__

            statistics[name] = (
                statistics.get(name, 0) + 1
            )

        info["module_statistics"] = dict(
            sorted(statistics.items())
        )

        return info