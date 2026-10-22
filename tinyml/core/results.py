from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class StageResult:
    stage: str
    metrics: Dict[str, Any] = field(
        default_factory=dict
    )
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self):
        return {
            "stage": self.stage,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }


class ResultsStore:

    def __init__(self):
        self.results = {}

    def add(
        self,
        stage: str,
        metrics=None,
        metadata=None,
    ):
        result = StageResult(
            stage=stage,
            metrics=metrics or {},
            metadata=metadata or {},
        )

        self.results[stage] = result

        return result

    def get(self, stage):
        if stage not in self.results:
            raise KeyError(
                f"No result registered for {stage}."
            )

        return self.results[stage]

    def to_dict(self):
        return {
            stage: result.to_dict()
            for stage, result in self.results.items()
        }