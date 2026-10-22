import copy

import torch

from .backend import QuantizationBackend


class FXPTQBackend(QuantizationBackend):
    """
    Post-training static quantization via PyTorch FX.

    Le backend est volontairement strict :
    si le modèle Detectron2 ne peut pas être préparé/converti
    correctement, l'erreur est propagée.

    Aucun faux M4 n'est produit.
    """

    def __init__(
        self,
        backend: str = "fbgemm",
    ):
        self.backend = backend

    def prepare(
        self,
        model,
        example_inputs,
    ):
        try:
            from torch.ao.quantization import (
                get_default_qconfig_mapping,
            )
            from torch.ao.quantization.quantize_fx import (
                prepare_fx,
            )
        except ImportError as error:
            raise RuntimeError(
                "PyTorch FX quantization API is unavailable."
            ) from error

        qconfig_mapping = get_default_qconfig_mapping(
            self.backend
        )

        prepared = prepare_fx(
            copy.deepcopy(model).eval(),
            qconfig_mapping,
            example_inputs,
        )

        return prepared

    def convert(self, model):
        from torch.ao.quantization.quantize_fx import (
            convert_fx,
        )

        return convert_fx(model)


class INT8PTQQuantizer:

    def __init__(
        self,
        backend: str = "fbgemm",
    ):
        self.backend = FXPTQBackend(
            backend=backend
        )

    def quantize(
        self,
        model,
        example_inputs,
        calibration_loader,
        max_batches: int = 100,
    ):
        prepared = self.backend.prepare(
            model,
            example_inputs,
        )

        prepared.eval()

        processed = 0

        with torch.no_grad():
            for batch in calibration_loader:
                prepared(batch)

                processed += 1

                if processed >= max_batches:
                    break

        if processed == 0:
            raise RuntimeError(
                "No calibration batch was processed."
            )

        quantized = self.backend.convert(
            prepared
        )

        return quantized

    @staticmethod
    def has_quantized_modules(model) -> bool:
        quantized_types = (
            "QuantizedConv2d",
            "QuantizedLinear",
            "ConvReLU2d",
            "Linear",
        )

        return any(
            module.__class__.__name__ in quantized_types
            for module in model.modules()
        )

    @staticmethod
    def verify_forward(
        model,
        batch,
    ):
        model.eval()

        with torch.no_grad():
            output = model(batch)

        if output is None:
            raise RuntimeError(
                "Quantized model forward returned None."
            )

        return output