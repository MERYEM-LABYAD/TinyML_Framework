from dataclasses import dataclass, asdict
from typing import Any, Dict, List

import torch.nn as nn


@dataclass
class ModuleInfo:
    name: str
    module_type: str
    parameters: int
    trainable_parameters: int
    parent: str
    depth: int


class ModuleInventory:
    def __init__(self, model: nn.Module):
        self.model = model
        self.modules: List[ModuleInfo] = []

    def build(self) -> List[ModuleInfo]:
        self.modules.clear()

        for name, module in self.model.named_modules():

            # Enregistre le nom réel du module directement sur
            # l'objet PyTorch afin que les autres composants du
            # framework puissent retrouver son identité.
            module._tinyml_name = name

            parent = name.rsplit(".", 1)[0] if "." in name else ""
            depth = name.count(".") + 1 if name else 0

            parameters = sum(
                parameter.numel()
                for parameter in module.parameters(recurse=False)
            )

            trainable_parameters = sum(
                parameter.numel()
                for parameter in module.parameters(recurse=False)
                if parameter.requires_grad
            )

            self.modules.append(
                ModuleInfo(
                    name=name,
                    module_type=module.__class__.__name__,
                    parameters=parameters,
                    trainable_parameters=trainable_parameters,
                    parent=parent,
                    depth=depth,
                )
            )

        return self.modules

    def get_by_type(self, module_type: str) -> List[ModuleInfo]:
        return [
            module
            for module in self.modules
            if module.module_type == module_type
        ]

    def summary(self) -> Dict[str, int]:
        result: Dict[str, int] = {}

        for module in self.modules:
            result[module.module_type] = (
                result.get(module.module_type, 0) + 1
            )

        return dict(sorted(result.items()))

    def to_dict(self) -> List[Dict[str, Any]]:
        return [asdict(module) for module in self.modules]