"""
Classement "normalized-global" tel que decrit dans le rapport (Chap. 4, par. 4.4.2 - 4.4.3).

Rapport, 4.4.2 : "the producers of a residual sum remove the same channel indices, so their
per-channel scores are summed before normalization".
Rapport, 4.4.3 : "normalizes those scores per group (percentile-clipping followed by min-max
scaling to [0, 1]), pools the normalized scores across the whole network, and keeps the global
top fraction (1 - r) for a target ratio r, subject to a MIN_KEEP floor ... s_final = w_g * s_norm".

Difference avec tinyml/importance/global_ranker.py (ancien) :
  * l'ancien fait UNE normalisation min-max globale ; ici la normalisation est PAR GROUPE ;
  * l'ancien ne somme pas les scores des couches couplees ; ici oui ;
  * l'ancien est appele de facon iterative ; ici le classement est fait UNE fois (one-shot).

Hypotheses (non precisees par le rapport, a declarer) :
  * percentiles d'ecretage par defaut : (1, 99) ;
  * departage des egalites : score brut croissant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch


@dataclass
class PruningUnit:
    """Un groupe de canaux qui partagent le meme masque de conservation."""

    uid: int
    members: List[str]                 # modules dont les canaux de SORTIE sont coupes ensemble
    consumers: List[str]               # modules dont les canaux d'ENTREE diminuent avec l'unite
    n_channels: int
    label: str = "other"               # stem | internal | residual | rpn | fc | other
    stage: Optional[str] = None        # res2..res5 si applicable
    protected_reason: Optional[str] = None
    scores: Optional[torch.Tensor] = None   # scores bruts SOMMES sur les membres

    @property
    def protected(self) -> bool:
        return self.protected_reason is not None


def normalize_group(
    scores: torch.Tensor,
    clip: Optional[Tuple[float, float]] = (1.0, 99.0),
) -> torch.Tensor:
    """Ecretage par percentiles puis min-max vers [0, 1], sur UN groupe."""
    v = scores.detach().float().cpu().reshape(-1)
    if v.numel() == 0:
        return v
    if isinstance(clip, str) and clip == "rank":
        # HYPOTHESE EXPLORATOIRE (non decrite par le rapport) : rang percentile dans le groupe -> [0, 1]
        # (distribution uniforme par groupe). Sert a tester si la normalisation du rapport donne des
        # coupes par bloc quasi uniformes (40-49 %) comme sur ses figures.
        if v.numel() == 1:
            return torch.ones_like(v)
        order = torch.argsort(v, stable=True)
        ranks = torch.empty_like(v)
        ranks[order] = torch.arange(v.numel(), dtype=v.dtype)
        return ranks / float(v.numel() - 1)
    if clip is not None:
        lo = torch.quantile(v, clip[0] / 100.0)
        hi = torch.quantile(v, clip[1] / 100.0)
        if float(hi) > float(lo):
            v = v.clamp(float(lo), float(hi))
    vmin, vmax = v.min(), v.max()
    span = float(vmax - vmin)
    if span <= 1e-12 * max(1.0, abs(float(vmax))):
        return torch.ones_like(v)
    return (v - vmin) / span


def unit_weight(unit: PruningUnit, weights: Dict[str, float]) -> float:
    """w_g : cle la plus precise d'abord ('internal:res3'), puis le libelle ('internal')."""
    if unit.stage is not None:
        key = f"{unit.label}:{unit.stage}"
        if key in weights:
            return float(weights[key])
    return float(weights.get(unit.label, 1.0))


def plan_removals(
    units: Sequence[PruningUnit],
    weights: Dict[str, float],
    ratio: float,
    min_keep: int = 8,
    clip: Optional[Tuple[float, float]] = (1.0, 99.0),
) -> Tuple[Dict[int, List[int]], Dict[str, float]]:
    """
    Retourne ({uid: [indices a supprimer]}, statistiques).

    ratio r = fraction des canaux ELIGIBLES (non proteges, > MIN_KEEP) supprimes globalement.
    """
    if not 0.0 <= ratio < 1.0:
        raise ValueError("ratio doit verifier 0 <= r < 1")

    eligible = [u for u in units if not u.protected and u.scores is not None and u.n_channels > min_keep]

    finals: List[torch.Tensor] = []
    raws: List[torch.Tensor] = []
    uid_of: List[torch.Tensor] = []
    ch_of: List[torch.Tensor] = []

    for u in eligible:
        s_norm = normalize_group(u.scores, clip)
        w = unit_weight(u, weights)
        finals.append(w * s_norm)
        raws.append(u.scores.detach().float().cpu().reshape(-1))
        uid_of.append(torch.full((u.n_channels,), u.uid, dtype=torch.long))
        ch_of.append(torch.arange(u.n_channels, dtype=torch.long))

    if not finals:
        return {}, {"pool": 0, "target": 0, "removed": 0}

    final = torch.cat(finals)
    raw = torch.cat(raws)
    uid_t = torch.cat(uid_of)
    ch_t = torch.cat(ch_of)

    # tri croissant par score final, egalites departagees par score brut (tri stable en 2 passes)
    order = torch.argsort(raw, stable=True)
    order = order[torch.argsort(final[order], stable=True)]

    pool = int(final.numel())
    target = int(round(ratio * pool))
    remaining = {u.uid: u.n_channels for u in eligible}
    removed: Dict[int, List[int]] = {}
    n_removed = 0
    last_threshold = 0.0

    for pos in order.tolist():
        if n_removed >= target:
            break
        uid = int(uid_t[pos])
        if remaining[uid] <= min_keep:        # plancher MIN_KEEP
            continue
        removed.setdefault(uid, []).append(int(ch_t[pos]))
        remaining[uid] -= 1
        n_removed += 1
        last_threshold = float(final[pos])

    for uid in removed:
        removed[uid].sort()

    return removed, {
        "pool": pool,
        "target": target,
        "removed": n_removed,
        "score_threshold": last_threshold,
    }
