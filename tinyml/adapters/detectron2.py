from typing import Any, Dict, Iterable

from detectron2.modeling import build_model
from detectron2.config import CfgNode

from .base import BaseAdapter


class Detectron2Adapter(BaseAdapter):
    """
    Adapter permettant au cœur TinyML d'interagir
    avec un modèle Detectron2.
    """

    @property
    def framework_name(self) -> str:
        return "Detectron2"

    @property
    def architecture_name(self) -> str:
        """
        Récupère le nom de l'architecture à partir
        de la structure réelle du modèle.
        """
        model_name = self.model.__class__.__name__

        if hasattr(self.model, "backbone"):
            backbone_name = self.model.backbone.__class__.__name__
            return f"{model_name} + {backbone_name}"

        return model_name

    

    def iter_modules(self) -> Iterable:
        """
        Parcourt tous les modules PyTorch du modèle.

        named_modules() fournit :
        - le nom du module
        - le module lui-même
        """
        return self.model.named_modules()

    def get_example_inputs(self) -> list:
        """
        Construit une entrée représentative conforme au contrat
        d'entrée attendu par un modèle Faster R-CNN Detectron2.

        Detectron2 attend une liste de dictionnaires contenant
        notamment une image sous la forme d'un tenseur [C, H, W].
        """
        import torch

        image = torch.zeros(
            (3, 800, 800),
            dtype=torch.float32,
            device=next(self.model.parameters()).device,
        )

        return [
            {
                "image": image,
                "height": 800,
                "width": 800,
            }
        ]

    def count_parameters(self) -> int:
        """Nombre total de paramètres du modèle."""
        return sum(
            parameter.numel()
            for parameter in self.model.parameters()
        )

    def count_trainable_parameters(self) -> int:
        """Nombre de paramètres entraînables."""
        return sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )

    def get_module_statistics(self) -> Dict[str, int]:
        """
        Compte les différents types de modules présents
        dans le modèle.
        """
        statistics: Dict[str, int] = {}

        for _, module in self.model.named_modules():
            module_type = module.__class__.__name__
            statistics[module_type] = statistics.get(module_type, 0) + 1

        return dict(sorted(statistics.items()))

    def get_model_info(self) -> Dict[str, Any]:
        """
        Informations générales + statistiques des modules.
        """
        info = super().get_model_info()

        info["module_statistics"] = self.get_module_statistics()

        return info