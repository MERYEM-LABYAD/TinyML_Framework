"""
Les configurations d'elagage du rapport (Tables 26, 28, 30, 32, 34, 36, 38) et leurs resultats.

Tous les poids w_g sont ceux du rapport. Le rapport ne donne PAS les percentiles d'ecretage :
la valeur par defaut (1, 99) est une hypothese a declarer.

'reference' = ce que le rapport a obtenu APRES recuperation (distillation) ; ce ne sont pas des
valeurs a atteindre en M1 seul (M1 seul s'effondre dans le rapport aussi).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class ReportPolicy:
    name: str
    ratio: float                      # GLOBAL_PRUNE_RATIO
    filter_zero_thresh: float         # FILTER_ZERO_THRESH (voir note dans report_pruner)
    min_keep: int                     # MIN_KEEP
    weights: Dict[str, float]         # w_stem, w_internal, w_residual, w_rpn, w_fc
    clip: Optional[Tuple[float, float]] = (1.0, 99.0)
    note: str = ""
    # valeurs du rapport (apres recuperation)
    ref_params_reduced_pct: Optional[float] = None
    ref_ap: Optional[float] = None
    ref_ap50: Optional[float] = None
    ref_text: str = ""
    # fourchettes de canaux retires (%) lues dans le texte du rapport, par libelle d'unite
    ref_ranges: Optional[Dict[str, Tuple[float, float]]] = None


def _w(stem=1.0, internal=1.0, residual=1.0, rpn=1.0, fc=1.0) -> Dict[str, float]:
    return {"stem": stem, "internal": internal, "residual": residual, "rpn": rpn, "fc": fc}


CONFIGS: Dict[str, ReportPolicy] = {
    "uniform_local": ReportPolicy(
        "uniform_local", 0.80, 0.80, 8, _w(),
        note="ATTENTION : le rapport parle d'un elagage 'local' (meme ratio par bloc). Le mecanisme exact "
             "n'est pas decrit ; ici on applique le classement du rapport avec tous les w = 1. "
             "Reference faible du rapport (baseline naive).",
        ref_params_reduced_pct=51.8, ref_ap=0.5127, ref_ap50=0.8301,
        ref_text="fc6 -99 %, fc7 -98 %, RPN -68 %",
        ref_ranges={"fc1": (99, 99), "fc2": (98, 98), "rpn": (68, 68)}),
    "protect_head": ReportPolicy(
        "protect_head", 0.80, 0.80, 8, _w(fc=3.0),
        ref_params_reduced_pct=48.6, ref_ap=0.7585, ref_ap50=0.9284,
        ref_text="fc6 -13 %, fc7 -6 %",
        ref_ranges={"fc1": (13, 13), "fc2": (6, 6)}),
    "expose_residual": ReportPolicy(
        "expose_residual", 0.55, 0.80, 8, _w(residual=0.7, rpn=3.0, fc=3.5),
        ref_params_reduced_pct=18.4, ref_ap=0.7907, ref_ap50=0.9494,
        ref_text="canaux residuels -39 a -48 %, internes presque intacts",
        ref_ranges={"residual": (39, 48), "internal": (0, 10)}),
    "expose_internal": ReportPolicy(
        "expose_internal", 0.55, 0.80, 8, _w(internal=0.7, rpn=3.0, fc=3.5),
        ref_params_reduced_pct=31.5, ref_ap=0.7886, ref_ap50=0.9434,
        ref_text="canaux internes -42 a -49 %, residuels presque intacts",
        ref_ranges={"internal": (42, 49), "residual": (0, 10)}),
    "expose_whole_backbone": ReportPolicy(
        "expose_whole_backbone", 0.55, 0.80, 8, _w(internal=0.7, residual=0.7, rpn=3.0, fc=3.5),
        note="idk2 : meilleur compromis du rapport (96 % de l'AP de l'enseignant).",
        ref_params_reduced_pct=41.8, ref_ap=0.7955, ref_ap50=0.9553,
        ref_text="environ -40 a -49 % partout dans le backbone",
        ref_ranges={"internal": (40, 49), "residual": (40, 49)}),
    "aggressive": ReportPolicy(
        "aggressive", 0.60, 0.80, 8, _w(stem=0.85, internal=0.7, residual=0.7, rpn=2.7, fc=2.9),
        ref_params_reduced_pct=57.6, ref_ap=0.7394, ref_ap50=0.9178,
        ref_text="backbone -72 a -79 %, fc6 -10 %, fc7 -3 %",
        ref_ranges={"internal": (72, 79), "residual": (72, 79), "fc1": (10, 10), "fc2": (3, 3)}),
    "maximal": ReportPolicy(
        "maximal", 0.90, 0.90, 8, _w(stem=1.5, internal=0.7, residual=0.7, rpn=2.7, fc=3.0),
        ref_params_reduced_pct=61.7, ref_ap=0.6754, ref_ap50=0.8719,
        ref_text="backbone -77 a -91 %",
        ref_ranges={"internal": (77, 91), "residual": (77, 91)}),
}

# Ordre d'execution recommande (le plus utile pour la comparaison finale d'abord)
RECOMMENDED_ORDER = [
    "expose_whole_backbone", "maximal", "aggressive",
    "protect_head", "expose_internal", "expose_residual", "uniform_local",
]
