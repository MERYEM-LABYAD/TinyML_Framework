"""Tests du mecanisme 'poids a zero puis filtres retires' sur un ResNet-18 jouet (sans Detectron2)."""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_report_pruner_toy import _build, _classify
from tinyml.importance.report_zeros import (GROUP_MODES, channel_scores_from_weights, normalize_groups,
                                            plan_zero_removals)
from tinyml.pruning.report_pruner import ReportPruner


def _wscores(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {n: torch.rand(m.weight.shape, generator=g) ** 3 for n, m in model.named_modules()
            if isinstance(m, (nn.Conv2d, nn.Linear))}


def _pruner(model, x, builder, ws):
    p = ReportPruner(model, x, channel_scores_from_weights(ws), classifier=_classify,
                     hard_exclusions=("fc",), graph_builder=builder, verbose=False)
    p.discover_units()
    return p


W = {"stem": 1.0, "internal": 0.7, "residual": 0.7}


def test_zero_ratio_empty_and_monotone():
    model, x, builder = _build()
    ws = _wscores(model)
    p = _pruner(model, x, builder, ws)
    for group in GROUP_MODES:
        norm = normalize_groups(p.units, ws, group, (1.0, 99.0))
        plan, _, masks = plan_zero_removals(p.units, norm, W, 0.0, 0.8)
        assert plan == {} and masks == {}
        reds = []
        for r in (0.3, 0.6, 0.9):
            plan, st, _ = plan_zero_removals(p.units, norm, W, r, 0.5)
            tot, _ = p.predict_parameters(plan)
            reds.append(p.params_before - tot)
        assert reds[0] <= reds[1] <= reds[2], (group, reds)
        assert reds[2] > 0


def test_threshold_selectivity():
    """Un poids w < 1 expose le groupe : il perd beaucoup plus de filtres, via le seuil de zeros."""
    model, x, builder = _build()
    ws = _wscores(model)
    p = _pruner(model, x, builder, ws)
    norm = normalize_groups(p.units, ws, "layer", (1.0, 99.0))
    plan, _, _ = plan_zero_removals(p.units, norm, {"internal": 0.5, "residual": 1.5, "stem": 1.5},
                                    0.5, 0.5)
    pcts = p.unit_pcts(plan)
    assert sum(pcts["internal"]) / len(pcts["internal"]) > sum(pcts["residual"]) / len(pcts["residual"])


def test_surgery_matches_prediction_and_keeps_zeros():
    model, x, builder = _build()
    ws = _wscores(model)
    p = _pruner(model, x, builder, ws)
    norm = normalize_groups(p.units, ws, "layer", (1.0, 99.0))
    plan, st, masks = plan_zero_removals(p.units, norm, W, 0.7, 0.6)
    assert plan, "le test doit couper quelque chose"
    predicted, per_module = p.predict_parameters(plan)
    with torch.no_grad():
        for n, mk in masks.items():
            dict(model.named_modules())[n].weight[mk] = 0.0
    rep = p.apply(plan)
    assert not rep["rejected"], rep["rejected"]
    for n, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            got = m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
            assert got == per_module[n], (n, got, per_module[n])
    out = model(x)
    assert out.shape == (1, 10) and torch.isfinite(out).all()
    zero_frac = sum(int((m.weight == 0).sum()) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))) \
        / sum(m.weight.numel() for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear)))
    assert zero_frac > 0.1, zero_frac          # les zeros des filtres survivants sont conserves


def test_min_keep_floor():
    model, x, builder = _build()
    ws = _wscores(model)
    p = _pruner(model, x, builder, ws)
    norm = normalize_groups(p.units, ws, "unit", None)
    plan, _, _ = plan_zero_removals(p.units, norm, {}, 0.99, 0.0, min_keep=8)
    for u in p.units:
        if u.uid in plan:
            assert u.n_channels - len(plan[u.uid]) >= 8


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
