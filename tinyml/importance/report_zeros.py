"""
Mecanisme "poids mis a zero puis filtres retires" decrit par les TABLES d'experiences du rapport
(Tables 26, 28, 30, 32, 34, 36, 38) :

    GLOBAL_PRUNE_RATIO   = fraction de POIDS mis a zero globalement
    FILTER_ZERO_THRESH   = un filtre est retire si plus de cette fraction de SES poids sont a zero
    MIN_KEEP             = plancher de canaux par groupe
    AGGRESSIVENESS_W     = poids w_g par bloc :  s_final = w_g * s_norm

Pipeline (hypothese de travail, a declarer) :
  1. score Taylor PAR POIDS |w * dL/dw| accumule sur N images (fichier taylor_weights_*.pt) ;
  2. normalisation par groupe (ecretage par percentiles puis min-max) ;  groupe = couche | unite |
     libelle | libelle+etage  (non precise par le rapport : a calibrer) ;
  3. ponderation  s_final = w_g * s_norm ;
  4. seuil GLOBAL : les `ratio` fraction de poids de plus faible s_final sont mis a zero ;
  5. un canal (filtre) est retire si la fraction de zeros de son filtre (pour un groupe couple : de
     TOUS les filtres couples du meme canal) depasse FILTER_ZERO_THRESH, plancher MIN_KEEP ;
  6. les zeros des filtres survivants sont conserves (le modele elague a donc aussi de la sparsite).

Pourquoi ce mecanisme : le classement par canaux seul ne reproduit pas la selectivite des figures du
rapport (ex. 'expose residual' : residuels -39..-48 %, internes presque intacts) ; le seuil de zeros
amplifie fortement un ecart de poids 0.7 contre 1.0. A valider contre les 7 reductions du rapport.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from tinyml.importance.report_ranker import PruningUnit, unit_weight

GROUP_MODES = ("layer", "unit", "label", "labelstage")


def _kth_smallest(v: torch.Tensor, k: int) -> float:
    k = max(1, min(int(k), v.numel()))
    return float(v.kthvalue(k).values)


def _percentile(v: torch.Tensor, p: float) -> float:
    k = int(round(p / 100.0 * (v.numel() - 1))) + 1
    return _kth_smallest(v, k)


def _minmax(v: torch.Tensor, clip: Optional[Tuple[float, float]]) -> torch.Tensor:
    if v.numel() == 0:
        return v
    if clip is not None and v.numel() > 2:
        lo, hi = _percentile(v, clip[0]), _percentile(v, clip[1])
        if hi > lo:
            v = v.clamp(lo, hi)
    vmin, vmax = float(v.min()), float(v.max())
    span = vmax - vmin
    if span <= 1e-12 * max(1.0, abs(vmax)):
        return torch.ones_like(v)
    return (v - vmin) / span


def eligible_units(units: Sequence[PruningUnit], wscores: Dict[str, torch.Tensor], min_keep: int):
    return [u for u in units
            if not u.protected and u.n_channels > min_keep and all(m in wscores for m in u.members)]


def normalize_groups(
    units: Sequence[PruningUnit],
    wscores: Dict[str, torch.Tensor],
    group: str = "layer",
    clip: Optional[Tuple[float, float]] = (1.0, 99.0),
    min_keep: int = 8,
) -> Dict[str, torch.Tensor]:
    """Retourne {module: score normalise [0,1] de meme forme que le poids}."""
    if group not in GROUP_MODES:
        raise ValueError(f"group doit etre dans {GROUP_MODES}")
    elig = eligible_units(units, wscores, min_keep)
    buckets: Dict[object, List[str]] = {}
    for u in elig:
        for m in u.members:
            if group == "layer":
                key = ("layer", m)
            elif group == "unit":
                key = ("unit", u.uid)
            elif group == "label":
                key = ("label", u.label)
            else:
                key = ("labelstage", u.label, u.stage)
            buckets.setdefault(key, []).append(m)

    out: Dict[str, torch.Tensor] = {}
    for key, names in buckets.items():
        flats = [wscores[m].detach().float().reshape(-1) for m in names]
        norm = _minmax(torch.cat(flats) if len(flats) > 1 else flats[0], clip)
        pos = 0
        for m, f in zip(names, flats):
            n = f.numel()
            out[m] = norm[pos:pos + n].reshape(wscores[m].shape).clone()
            pos += n
    return out


def plan_zero_removals(
    units: Sequence[PruningUnit],
    norm: Dict[str, torch.Tensor],
    weights: Dict[str, float],
    ratio: float,
    thresh: float,
    min_keep: int = 8,
) -> Tuple[Dict[int, List[int]], Dict[str, float], Dict[str, torch.Tensor]]:
    """
    Retourne (plan {uid: [canaux retires]}, statistiques, masques {module: bool, True = poids mis a zero}).
    """
    if not 0.0 <= ratio < 1.0:
        raise ValueError("ratio doit verifier 0 <= r < 1")
    elig = [u for u in units if not u.protected and u.n_channels > min_keep
            and all(m in norm for m in u.members)]
    if not elig:
        return {}, {"pool": 0}, {}

    finals: Dict[str, torch.Tensor] = {}
    for u in elig:
        w = unit_weight(u, weights)
        for m in u.members:
            finals[m] = norm[m] * w

    names = list(finals.keys())
    flat = torch.cat([finals[m].reshape(-1) for m in names])
    pool = int(flat.numel())
    k = int(round(ratio * pool))
    if k <= 0:
        return {}, {"pool": pool, "zeroed": 0, "weight_threshold": float("-inf")}, {}
    thr = _kth_smallest(flat, k)
    del flat
    masks = {m: finals[m] <= thr for m in names}
    zeroed = int(sum(int(mk.sum()) for mk in masks.values()))

    plan: Dict[int, List[int]] = {}
    for u in elig:
        zeros = torch.zeros(u.n_channels, dtype=torch.float64)
        denom = 0
        for m in u.members:
            mk = masks[m].reshape(u.n_channels, -1)
            zeros += mk.sum(dim=1).double()
            denom += mk.shape[1]
        frac = zeros / float(denom)
        cand = (frac > thresh).nonzero(as_tuple=True)[0]
        room = u.n_channels - min_keep                         # plancher MIN_KEEP
        if cand.numel() > room:
            order = torch.argsort(frac[cand], descending=True, stable=True)
            cand = cand[order[:room]]
        if cand.numel():
            plan[u.uid] = sorted(int(i) for i in cand.tolist())

    stats = {"pool": pool, "zeroed": zeroed, "zeroed_fraction": zeroed / pool,
             "weight_threshold": thr, "units_with_removal": len(plan),
             "channels_removed": int(sum(len(v) for v in plan.values()))}
    return plan, stats, masks


def channel_scores_from_weights(wscores: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Somme par canal de sortie (sert uniquement a la decouverte des unites)."""
    return {n: t.detach().double().reshape(t.shape[0], -1).sum(dim=1).float() for n, t in wscores.items()}
