from .complexity import (
    count_parameters,
    parameter_reduction,
)

from .model_size import (
    model_size_bytes,
    size_reduction,
)

from .latency import LatencyBenchmark


class ModelBenchmark:

    def __init__(
        self,
        baseline_model,
    ):
        self.baseline_model = baseline_model

        self.baseline_parameters = count_parameters(
            baseline_model
        )

        self.baseline_size = model_size_bytes(
            baseline_model
        )

    def evaluate(
        self,
        model,
        forward_fn,
    ):
        parameters = count_parameters(model)
        size = model_size_bytes(model)

        latency = LatencyBenchmark().measure(
            model,
            forward_fn,
        )

        return {
            "parameters": parameters,
            "parameter_reduction_percent":
                parameter_reduction(
                    self.baseline_parameters,
                    parameters,
                ),
            "model_size_bytes": size,
            "model_size_mb":
                size / (1024 * 1024),
            "model_size_reduction_percent":
                size_reduction(
                    self.baseline_size,
                    size,
                ),
            **latency,
        }