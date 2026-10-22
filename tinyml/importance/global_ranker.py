from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class ChannelCandidate:
    module_name: str
    module: nn.Module
    channel_idx: int
    score: float
    normalized_score: float
    weighted_score: float
    group_name: Optional[str] = None


class GlobalRanker:
    """
    Global channel ranking.

    Scientific protocol:
        1. collect one importance score per output channel
        2. optional global percentile clipping
        3. global min-max normalization
        4. apply aggressiveness weight
        5. globally sort channels by final score

    Lower final scores are pruned first.
    """

    def __init__(
        self,
        importance_scores: Dict[str, torch.Tensor],
        modules: Optional[Dict[str, nn.Module]] = None,
        aggressiveness: Optional[Dict[str, float]] = None,
        percentile_clip: Optional[Tuple[float, float]] = None,
    ):
        self.importance_scores = importance_scores
        self.modules = modules or {}
        self.aggressiveness = aggressiveness or {}

        if percentile_clip is None:
            self.percentile_clip = None
        else:
            low, high = percentile_clip

            if not 0.0 <= low < high <= 100.0:
                raise ValueError(
                    "percentile_clip must satisfy "
                    "0 <= low < high <= 100."
                )

            self.percentile_clip = (
                float(low),
                float(high),
            )

    # ---------------------------------------------------------
    # Group / aggressiveness helpers
    # ---------------------------------------------------------

    def _get_weight(self, module_name: str) -> float:
        if module_name in self.aggressiveness:
            return float(
                self.aggressiveness[module_name]
            )

        best_match = ""
        best_weight = 1.0

        for pattern, weight in self.aggressiveness.items():
            if (
                module_name.startswith(pattern + ".")
                and len(pattern) > len(best_match)
            ):
                best_match = pattern
                best_weight = float(weight)

        return best_weight

    def _get_group_name(
        self,
        module_name: str,
    ) -> Optional[str]:

        best_match = ""
        best_group = None

        for pattern in self.aggressiveness:

            if module_name == pattern:
                return pattern

            if (
                module_name.startswith(pattern + ".")
                and len(pattern) > len(best_match)
            ):
                best_match = pattern
                best_group = pattern

        return best_group

    # ---------------------------------------------------------
    # Global normalization
    # ---------------------------------------------------------

    def _global_normalize(
        self,
        values: torch.Tensor,
    ) -> torch.Tensor:

        values = (
            values.detach()
            .float()
            .cpu()
        )

        if values.numel() == 0:
            return values

        # Optional GLOBAL percentile clipping
        if self.percentile_clip is not None:

            low_percentile, high_percentile = (
                self.percentile_clip
            )

            low_value = torch.quantile(
                values,
                low_percentile / 100.0,
            )

            high_value = torch.quantile(
                values,
                high_percentile / 100.0,
            )

            if not torch.isclose(
                low_value,
                high_value,
            ):
                values = torch.clamp(
                    values,
                    min=float(low_value),
                    max=float(high_value),
                )

        minimum = values.min()
        maximum = values.max()

        if torch.isclose(
            minimum,
            maximum,
        ):
            return torch.ones_like(values)

        return (
            values - minimum
        ) / (
            maximum - minimum
        )

    # ---------------------------------------------------------
    # Candidate construction
    # ---------------------------------------------------------

    def build_candidates(
        self,
        modules=None,
    ) -> List[ChannelCandidate]:

        if modules is not None:
            self.modules = modules

        raw_entries = []

        # -----------------------------------------------------
        # Collect ALL currently valid channels first.
        # -----------------------------------------------------

        for (
            module_name,
            scores,
        ) in self.importance_scores.items():

            module = self.modules.get(
                module_name
            )

            if module is None:
                continue

            if not isinstance(
                module,
                (
                    nn.Conv2d,
                    nn.Linear,
                ),
            ):
                continue

            scores = (
                scores.detach()
                .float()
                .cpu()
                .reshape(-1)
            )

            if scores.numel() == 0:
                continue

            # Current physical output size.
            if isinstance(
                module,
                nn.Conv2d,
            ):
                channels = int(
                    module.out_channels
                )
            else:
                channels = int(
                    module.out_features
                )

            # Never create candidates for stale
            # Taylor scores beyond the current structure.
            usable_channels = min(
                int(scores.numel()),
                channels,
            )

            if usable_channels <= 0:
                continue

            scores = scores[
                :usable_channels
            ]

            group_name = (
                self._get_group_name(
                    module_name
                )
            )

            weight = (
                self._get_weight(
                    module_name
                )
            )

            for channel_idx in range(
                usable_channels
            ):
                raw_entries.append(
                    (
                        module_name,
                        module,
                        channel_idx,
                        float(
                            scores[channel_idx]
                        ),
                        group_name,
                        weight,
                    )
                )

        if not raw_entries:
            return []

        # -----------------------------------------------------
        # GLOBAL normalization across the entire network.
        # -----------------------------------------------------

        raw_scores = torch.tensor(
            [
                entry[3]
                for entry in raw_entries
            ],
            dtype=torch.float32,
        )

        normalized_scores = (
            self._global_normalize(
                raw_scores
            )
        )

        candidates = []

        for idx, entry in enumerate(
            raw_entries
        ):

            (
                module_name,
                module,
                channel_idx,
                raw_score,
                group_name,
                weight,
            ) = entry

            normalized = float(
                normalized_scores[idx]
            )

            weighted = (
                normalized * weight
            )

            candidates.append(
                ChannelCandidate(
                    module_name=module_name,
                    module=module,
                    channel_idx=channel_idx,
                    score=raw_score,
                    normalized_score=normalized,
                    weighted_score=weighted,
                    group_name=group_name,
                )
            )

        # -----------------------------------------------------
        # GLOBAL ranking.
        #
        # Lowest final score = first pruning candidate.
        # -----------------------------------------------------

        candidates.sort(
            key=lambda candidate: (
                candidate.weighted_score,
                candidate.score,
                candidate.module_name,
                candidate.channel_idx,
            )
        )

        return candidates

    def rank(
        self,
        modules=None,
    ) -> List[ChannelCandidate]:

        return self.build_candidates(
            modules
        )