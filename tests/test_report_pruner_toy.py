"""
Tests sans Detectron2 : le classement du rapport + la chirurgie Torch-Pruning sur un ResNet-18 jouet.
Lancer :  python -m pytest tests/test_report_pruner_toy.py -q      (depuis la racine du projet)
          ou  python tests/test_report_pruner_toy.py
"""
import os
import re
import sys

import torch
import torch.nn as nn
import torch_pruning as tp
import torchvision

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinyml.importance.report_ranker import PruningUnit, normalize_group, plan_removals
from tinyml.pruning.report_policy import ReportPolicy
from tinyml.pruning.report_pruner import ReportPruner


# ------------------------------------------------------------------ classement
def test_normalize_range_and_constant():
    v = torch.tensor([0.0, 1.0, 2.0, 3.0, 100.0])
    n = normalize_group(v, clip=None)
    assert float(n.min()) == 0.0 and float(n.max()) == 1.0
    assert torch.all(normalize_group(torch.ones(5)) == 1.0)


def test_ratio_min_keep_and_weights():
    torch.manual_seed(0)
    a = PruningUnit(0, ["a"], [], 100, "internal", scores=torch.rand(100))
    b = PruningUnit(1, ["b"], [], 100, "fc", scores=torch.rand(100))
    plan, st = plan_removals([a, b], {"internal": 1.0, "fc": 3.0}, ratio=0.5, min_keep=8)
    assert st["removed"] == 100
    assert len(plan.get(0, [])) > len(plan.get(1, []))          # fc (w=3) protege
    plan, _ = plan_removals([a, b], {}, ratio=0.99, min_keep=8)
    assert 100 - len(plan[0]) >= 8 and 100 - len(plan[1]) >= 8   # plancher MIN_KEEP


def test_protected_unit_untouched():
    a = PruningUnit(0, ["a"], [], 50, "internal", scores=torch.rand(50), protected_reason="x")
    b = PruningUnit(1, ["b"], [], 50, "internal", scores=torch.rand(50))
    plan, _ = plan_removals([a, b], {}, ratio=0.5)
    assert 0 not in plan and len(plan[1]) == 25


# ------------------------------------------------------------------ chirurgie sur resnet18
def _classify(members):
    for n in members:
        if n == "conv1":
            return "stem", None
    for n in members:
        m = re.match(r"layer(\d)\.\d\.(conv1|conv2|downsample\.0)$", n)
        if m:
            if m.group(2) in ("conv2", "downsample.0"):
                return "residual", f"res{m.group(1)}"
            return "internal", f"res{m.group(1)}"
    return "other", None


def _scores(model):
    return {n: torch.rand(m.weight.shape[0]) for n, m in model.named_modules()
            if isinstance(m, (nn.Conv2d, nn.Linear))}


def _build(seed=0):
    torch.manual_seed(seed)
    model = torchvision.models.resnet18(num_classes=10).eval()
    x = torch.randn(1, 3, 64, 64)

    def builder(m, ex):
        g = tp.DependencyGraph()
        g.build_dependency(m, example_inputs=ex)
        return g

    return model, x, builder


def test_units_and_exact_parameter_prediction():
    model, x, builder = _build()
    scores = _scores(model)
    p = ReportPruner(model, x, scores, classifier=_classify,
                     hard_exclusions=("fc",), graph_builder=builder, verbose=False)
    units = p.discover_units()
    labels = [u.label for u in units if not u.protected]
    assert "residual" in labels and "internal" in labels and "stem" in labels
    # resnet18 jouet : layer1 a un raccourci identite, donc le stem est couple au flux de layer1 ;
    # il reste 3 flux residuels avec projection (layer2..4) : downsample + conv2 des 2 blocs
    res = [u for u in units if u.label == "residual"]
    assert len(res) == 3
    assert any("layer2.0.downsample.0" in u.members and "layer2.1.conv2" in u.members for u in res)

    pol = ReportPolicy("t", 0.5, 0.8, 8, {"stem": 1.0, "internal": 0.7, "residual": 0.7})
    plan, st = p.plan(pol)
    predicted, per_module = p.predict_parameters(plan)
    rep = p.apply(plan)
    actual = p.count_parameters()
    assert not rep["rejected"], rep["rejected"]
    # le jouet utilise de vrais BatchNorm (parametres coupes par Torch-Pruning, hors Conv/Linear) ;
    # sur Detectron2 les FrozenBN n'ont pas de parametres, donc le total est exact. Ici : Conv/Linear seuls.
    conv_lin = sum(m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
                   for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear)))
    assert conv_lin == predicted - p.other_params, (conv_lin, predicted - p.other_params)
    # chaque module doit aussi correspondre individuellement
    for n, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            got = m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
            assert got == per_module[n], (n, got, per_module[n])
    out = model(x)
    assert out.shape == (1, 10) and torch.isfinite(out).all()
    assert actual < p.params_before


def test_zero_ratio_is_identity():
    model, x, builder = _build()
    ref = model(x).detach()
    p = ReportPruner(model, x, _scores(model), classifier=_classify,
                     hard_exclusions=("fc",), graph_builder=builder, verbose=False)
    p.discover_units()
    pol = ReportPolicy("t", 0.0, 0.8, 8, {})
    plan, _ = p.plan(pol)
    assert plan == {}
    p.apply(plan)
    assert torch.allclose(model(x), ref, atol=0, rtol=0)


def test_solver_hits_target():
    model, x, builder = _build()
    p = ReportPruner(model, x, _scores(model), classifier=_classify,
                     hard_exclusions=("fc",), graph_builder=builder, verbose=False)
    p.discover_units()
    pol = ReportPolicy("t", 0.5, 0.8, 8, {"internal": 0.7, "residual": 0.7})
    r = p.solve_ratio(pol, 40.0, tol=0.3)
    red = p.predicted_reduction_pct(pol, r)
    assert abs(red - 40.0) < 1.0, (r, red)


def test_rank_normalization_is_uniform():
    v = torch.tensor([0.0, 1.0, 2.0, 3.0, 1000.0])
    n = normalize_group(v, clip="rank")
    assert torch.allclose(n, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))
    # avec 'rank', la fraction coupee d'un groupe de poids w sous un seuil t vaut t / w (quel que soit le groupe)
    a = PruningUnit(0, ["a"], [], 1000, "internal", scores=torch.rand(1000) ** 5)
    b = PruningUnit(1, ["b"], [], 1000, "fc", scores=torch.rand(1000) ** 5)
    plan, _ = plan_removals([a, b], {"internal": 0.5, "fc": 2.0}, ratio=0.4, clip="rank")
    assert len(plan[0]) / 1000 > 0.5 and len(plan.get(1, [])) / 1000 < 0.2


def test_unit_pcts_and_deviation_and_solver_with_clip():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import run_m1_report as R
    model, x, builder = _build()
    p = ReportPruner(model, x, _scores(model), classifier=_classify,
                     hard_exclusions=("fc",), graph_builder=builder, verbose=False)
    p.discover_units()
    for clip in (None, (1.0, 99.0), (10.0, 90.0)):
        pol = ReportPolicy("t", 0.5, 0.8, 8, {"internal": 0.7, "residual": 0.7}, clip=clip)
        r = p.solve_ratio(pol, 40.0, tol=0.3)
        plan, _ = p.plan(pol, r)
        pcts = p.unit_pcts(plan)
        assert "internal" in pcts and "residual" in pcts
        dev, parts = R.deviation(pcts, {"internal": (30, 60), "residual": (30, 60)})
        assert dev >= 0.0
    assert R.parse_clip("5,95") == (5.0, 95.0) and R.parse_clip("none") is None and R.parse_clip(None) == "policy"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
