from typing import Any, Dict

from tinyml.adapters.base import BaseAdapter


class ModelAnalyzer:
    """
    Analyse générique d'un modèle à travers son Adapter.

    Le ModelAnalyzer ne dépend pas directement de Detectron2,
    TorchVision ou d'une architecture particulière.
    """

    def __init__(self, adapter: BaseAdapter):
        self.adapter = adapter

    def analyze(self) -> Dict[str, Any]:
        """
        Effectue l'analyse de base du modèle.
        """
        return self.adapter.get_model_info()

    def print_summary(self) -> None:
        """Affiche un résumé lisible de l'analyse."""
        info = self.analyze()

        print("=" * 60)
        print("TINYML MODEL ANALYSIS")
        print("=" * 60)

        print(f"Framework              : {info['framework']}")
        print(f"Architecture            : {info['architecture']}")
        print(f"Parameters              : {info['parameters']:,}")
        print(
            f"Trainable parameters   : "
            f"{info['trainable_parameters']:,}"
        )

        print("\nModule statistics:")
        print("-" * 60)

        for module_type, count in info["module_statistics"].items():
            print(f"{module_type:<35} {count:>6}")

        print("=" * 60)