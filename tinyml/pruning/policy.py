from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PruningPolicy:
    """
    Reference pruning policy.

    GLOBAL_PRUNE_RATIO:
        Global target pruning ratio r.

    FILTER_ZERO_THRESH:
        Fraction of zeroed weights required before a filter
        is considered removable.

    MIN_KEEP:
        Minimum number of output channels retained in a group.

    AGGRESSIVENESS_W:
        Per-group weighting:
            > 1  -> protect
            = 1  -> neutral
            < 1  -> expose
    """

    ratio: float = 0.80

    filter_zero_thresh: float = 0.80

    min_channels: int = 8

    protected_modules: List[str] = field(
        default_factory=list
    )

    ignored_modules: List[str] = field(
        default_factory=list
    )

    aggressiveness: Dict[str, float] = field(
        default_factory=dict
    )

    target_parameter_reduction: float = 0.0

    percentile_clip: Optional[
        Tuple[float, float]
    ] = None

    def validate(self) -> None:

        if not 0.0 <= self.ratio < 1.0:
            raise ValueError(
                "GLOBAL_PRUNE_RATIO must satisfy "
                "0 <= ratio < 1."
            )

        if not 0.0 < self.filter_zero_thresh <= 1.0:
            raise ValueError(
                "FILTER_ZERO_THRESH must satisfy "
                "0 < filter_zero_thresh <= 1."
            )

        if self.min_channels < 1:
            raise ValueError(
                "MIN_KEEP must be >= 1."
            )

        if not 0.0 <= self.target_parameter_reduction < 1.0:
            raise ValueError(
                "target_parameter_reduction must satisfy "
                "0 <= value < 1."
            )

        for group_name, weight in (
            self.aggressiveness.items()
        ):
            if weight <= 0:
                raise ValueError(
                    f"Aggressiveness weight for "
                    f"'{group_name}' must be > 0."
                )

        if self.percentile_clip is not None:

            low, high = self.percentile_clip

            if not 0.0 <= low < high <= 100.0:
                raise ValueError(
                    "percentile_clip must satisfy "
                    "0 <= low < high <= 100."
                )

    def is_protected(
        self,
        name: str,
    ) -> bool:

        return any(
            name == pattern
            or name.startswith(pattern + ".")
            for pattern in self.protected_modules
        )

    def is_ignored(
        self,
        name: str,
    ) -> bool:

        return any(
            name == pattern
            or name.startswith(pattern + ".")
            for pattern in self.ignored_modules
        )

    def allows_pruning(
        self,
        name: str,
    ) -> bool:

        return (
            not self.is_protected(name)
            and not self.is_ignored(name)
        )

    def get_aggressiveness(
        self,
        group_name: str,
    ) -> float:

        return self.aggressiveness.get(
            group_name,
            1.0,
        )

    @property
    def global_prune_ratio(self) -> float:
        return self.ratio

    @property
    def filter_zero_threshold(self) -> float:
        return self.filter_zero_thresh

    @property
    def min_keep(self) -> int:
        return self.min_channels