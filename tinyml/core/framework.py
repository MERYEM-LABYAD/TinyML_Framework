import copy

import torch

from tinyml.core.model_registry import ModelRegistry
from tinyml.core.results import ResultsStore

from tinyml.evaluation.complexity import (
    count_parameters,
)

from tinyml.pruning.validator import (
    PruningValidator,
)


class TinyMLFramework:

    def __init__(
        self,
        adapter,
        checkpoint_directory="checkpoints",
    ):
        self.adapter = adapter

        self.model = adapter.model

        self.registry = ModelRegistry(
            checkpoint_directory
        )

        self.results = ResultsStore()

        self.models = {}

    def register_stage(
        self,
        stage,
        model,
        metadata=None,
    ):
        self.models[stage] = model

        self.registry.save(
            stage=stage,
            model=model,
            metadata=metadata,
        )

        self.results.add(
            stage=stage,
            metrics={
                "parameters":
                    count_parameters(model)
            },
            metadata=metadata or {},
        )

    def get_stage(self, stage):
        if stage not in self.models:
            raise KeyError(
                f"Stage {stage} is not registered."
            )

        return self.models[stage]

    def clone_model(self, stage):
        if stage not in self.models:
            raise KeyError(
                f"Stage {stage} is not registered."
            )

        return copy.deepcopy(
            self.models[stage]
        )

    def validate_stage(
        self,
        stage,
        forward_fn,
    ):
        model = self.get_stage(stage)

        validator = PruningValidator(
            model
        )

        return validator.validate(
            forward_fn
        )