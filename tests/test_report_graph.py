"""Reproduit le bug 'bbox_pred absent du graphe' et verifie que le nouveau constructeur le corrige."""
import os, sys
import torch, torch.nn as nn, torch.nn.functional as F, torch_pruning as tp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tinyml.dependency.report_graph import build_detectron2_graph
from tinyml.pruning.report_policy import ReportPolicy
from tinyml.pruning.report_pruner import ReportPruner


class Inst:
    def __init__(self, **f): self._f = f
    def get_fields(self): return self._f


class Pred(nn.Module):
    def __init__(self):
        super().__init__(); self.cls_score = nn.Linear(16, 4); self.bbox_pred = nn.Linear(16, 12)
    def forward(self, x): return self.cls_score(x), self.bbox_pred(x)


class RoI(nn.Module):
    def __init__(self):
        super().__init__(); self.box_head = nn.Sequential(); self.box_head.add_module("fc1", nn.Linear(32, 16)); self.box_predictor = Pred()


class Det(nn.Module):
    """Sorties finales liees a cls_score seulement (comme une image sans detection : pas de boites)."""
    def __init__(self):
        super().__init__(); self.backbone = nn.Linear(5, 32); self.roi_heads = RoI()
    def forward(self, batch):
        x = F.relu(self.backbone(batch[0]["image"]))
        f = F.relu(self.roi_heads.box_head.fc1(x))
        scores, deltas = self.roi_heads.box_predictor(f)
        return [{"instances": Inst(scores=F.softmax(scores, 1).max(1).values)}]


def _scores(m):
    return {n: torch.rand(mod.weight.shape[0]) for n, mod in m.named_modules() if isinstance(mod, nn.Linear)}


def _classify(members):
    return ("fc", None) if any("fc1" in n or "backbone" in n for n in members) else ("other", None)


def test_old_builder_misses_bbox_pred_new_one_does_not():
    torch.manual_seed(0)
    x = [{"image": torch.randn(1, 5)}]
    m = Det().eval()
    # constructeur "sans hook" : reproduit le defaut
    g0 = tp.DependencyGraph()
    g0.build_dependency(m, example_inputs=x, output_transform=lambda o: tuple(v for it in o for v in it["instances"].get_fields().values()))
    assert m.roi_heads.box_predictor.bbox_pred not in g0.module2node
    # nouveau constructeur
    g1 = build_detectron2_graph(m, x, verbose=False)
    assert m.roi_heads.box_predictor.bbox_pred in g1.module2node


def test_pruning_keeps_forward_valid():
    torch.manual_seed(0)
    x = [{"image": torch.randn(1, 5)}]
    m = Det().eval()
    p = ReportPruner(m, x, _scores(m), classifier=_classify,
                     hard_exclusions=("roi_heads.box_predictor",), verbose=False,
                     graph_builder=lambda mm, ex: build_detectron2_graph(mm, ex, verbose=False))
    p.discover_units()
    plan, _ = p.plan(ReportPolicy("t", 0.5, 0.8, 4, {}))
    pred, _ = p.predict_parameters(plan)
    rep = p.apply(plan)
    assert not rep["rejected"] and not p.rejected, (rep, p.rejected)
    assert p.count_parameters() == pred
    assert m.roi_heads.box_predictor.bbox_pred.in_features == m.roi_heads.box_head.fc1.out_features
    m(x)   # ne leve pas "mat1 and mat2 shapes cannot be multiplied"


if __name__ == "__main__":
    test_old_builder_misses_bbox_pred_new_one_does_not(); print("OK builder")
    test_pruning_keeps_forward_valid(); print("OK pruning")
