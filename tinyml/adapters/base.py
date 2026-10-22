from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable


class BaseAdapter(ABC):
    """
    Interface commune entre le cœur du framework TinyML
    et une architecture/framework de deep learning particulier.

    Le Core ne doit pas dépendre directement de Detectron2,
    TorchVision, etc.
    """

    def __init__(self, model: Any):
        self.model = model

    @property
    @abstractmethod
    def framework_name(self) -> str:
        """Nom du framework utilisé par le modèle."""
        raise NotImplementedError

    @property
    @abstractmethod
    def architecture_name(self) -> str:
        """Nom de l'architecture du modèle."""
        raise NotImplementedError

    @abstractmethod
    def iter_modules(self) -> Iterable:
        """
        Retourne les modules du modèle de manière uniforme
        pour le reste du framework.
        """
        raise NotImplementedError

    @abstractmethod
    def count_parameters(self) -> int:
        """Retourne le nombre total de paramètres."""
        raise NotImplementedError

    @abstractmethod
    def count_trainable_parameters(self) -> int:
        """Retourne le nombre de paramètres entraînables."""
        raise NotImplementedError

    def get_model_info(self) -> Dict[str, Any]:
        """
        Fournit les informations générales utilisées
        par le ModelAnalyzer.
        """
        return {
            "framework": self.framework_name,
            "architecture": self.architecture_name,
            "parameters": self.count_parameters(),
            "trainable_parameters": self.count_trainable_parameters(),
        }