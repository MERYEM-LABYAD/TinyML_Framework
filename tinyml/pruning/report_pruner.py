"""
Elagage structure ONE-SHOT conforme au rapport (Chap. 4.3-4.5) :

    score Taylor (deja calcule)  ->  unites couplees (Torch-Pruning)  ->  scores sommes par unite
    -> normalisation par unite -> w_g * s_norm -> classement global -> suppression en UNE fois.

"Borrow the graph, own the policy" : Torch-Pruning fournit les couplages et la chirurgie ;
le classement et la politique sont ceux du rapport (report_ranker.py).

Exclusions DURES (rapport 4.4.2 : bus FPN, sorties des tetes RPN/RoI) :
    FPN (backbone.fpn_*), roi_heads.box_predictor.*, rpn_head.objectness_logits / anchor_deltas
    -> leurs dimensions de sortie ont un sens fixe (256 canaux du bus, nb de classes, nb d'ancres).
Exclusions DOUCES (poids d'agressivite, comme dans les experiences du rapport) :
    stem, convolutions internes, canaux residuels, rpn_head.conv (w_rpn), box_head.fc1/fc2 (w_fc).

Remarque FILTER_ZERO_THRESH : avec un classement par canaux, un canal choisi est supprime en entier
(fraction de zeros = 1.0 >= seuil). Le parametre est donc journalise mais n'a pas d'effet. Le rapport
ne decrit pas de mecanisme de zeros partiels pour le detecteur ; a confirmer avec l'encadrant.

Controles integres :
  * le nombre de parametres apres chirurgie doit egaler EXACTEMENT la prediction analytique ;
  * tout groupe rejete par Torch-Pruning est journalise (raison) et fait echouer la porte finale.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch_pruning as tp

from tinyml.importance.report_ranker import PruningUnit, plan_removals

DEFAULT_HARD_EXCLUSIONS: Tuple[str, ...] = (
    "backbone.fpn_",
    "roi_heads.box_predictor",
    "proposal_generator.rpn_head.objectness_logits",
    "proposal_generator.rpn_head.anchor_deltas",
)

_RES = re.compile(r"(?:^|\.)(res[2-5])\.(\d+)\.(conv1|conv2|conv3|shortcut)$")


def classify_detectron2(members: Sequence[str]) -> Tuple[str, Optional[str]]:
    """(libelle, stage) d'une unite Detectron2 Faster R-CNN R50-FPN."""
    for n in members:
        if n.endswith("stem.conv1"):
            return "stem", None
    label, stage = "other", None
    for n in members:
        m = _RES.search(n)
        if m:
            stage = m.group(1)
            if m.group(3) in ("conv3", "shortcut"):
                return "residual", stage
            label = "internal"
    if label == "internal":
        return "internal", stage
    for n in members:
        if "box_head.fc" in n:
            return "fc", None
        if "rpn_head.conv" in n:
            return "rpn", None
    return "other", None


def _default_graph_builder(model, example_inputs):
    # constructeur avec hooks RPN + box_predictor (sans eux, bbox_pred est absent du graphe)
    from tinyml.dependency.report_graph import build_detectron2_graph
    return build_detectron2_graph(model, example_inputs)


class ReportPruner:
    def __init__(
        self,
        model: nn.Module,
        example_inputs,
        scores: Dict[str, torch.Tensor],
        classifier: Callable = classify_detectron2,
        hard_exclusions: Sequence[str] = DEFAULT_HARD_EXCLUSIONS,
        graph_builder: Optional[Callable] = None,
        verbose: bool = True,
    ):
        self.model = model
        self.example_inputs = example_inputs
        self.scores = scores
        self.classifier = classifier
        self.hard_exclusions = tuple(hard_exclusions)
        self.verbose = verbose

        self.names: Dict[int, str] = {id(m): n for n, m in model.named_modules()}
        self.modules: Dict[str, nn.Module] = dict(model.named_modules())
        self.units: List[PruningUnit] = []
        self.rejected: List[Dict] = []
        self.original_sizes: Dict[str, Tuple[int, int]] = {}   # nom -> (out, in)
        self.params_before = self.count_parameters()

        was_training = model.training
        model.eval()                                  # le graphe doit etre construit en mode eval
        try:
            builder = graph_builder or _default_graph_builder
            self.graph = builder(model, example_inputs)
        finally:
            model.train(was_training)

        for n, m in self.modules.items():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                self.original_sizes[n] = (int(m.weight.shape[0]), int(m.weight.shape[1]))
        conv_lin = sum(
            m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
            for m in self.modules.values() if isinstance(m, (nn.Conv2d, nn.Linear))
        )
        self.other_params = self.params_before - conv_lin     # constant (0 attendu pour R50-FPN)

    # ------------------------------------------------------------------ utilitaires
    def log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def _is_out(self, dep) -> bool:
        try:
            return bool(self.graph.is_out_channel_pruning_fn(dep.handler))
        except Exception:
            tail = str(dep).split(" => ", 1)[-1]
            return "prune_out_channels" in tail and "prune_in_channels" not in tail

    @staticmethod
    def _idx_list(idxs) -> List[int]:
        out = []
        for i in idxs:
            out.append(int(i.idx) if hasattr(i, "idx") else int(i))
        return out

    def _hard_excluded(self, names: Sequence[str]) -> Optional[str]:
        for n in names:
            for pref in self.hard_exclusions:
                if n == pref or n.startswith(pref):
                    return f"exclusion dure ({pref})"
        return None

    # ------------------------------------------------------------------ decouverte des unites
    def discover_units(self) -> List[PruningUnit]:
        self.units = []
        seen_members: Dict[str, int] = {}
        seen_sigs = set()

        candidates = [
            (n, m) for n, m in self.modules.items()
            if isinstance(m, (nn.Conv2d, nn.Linear))
        ]

        for name, module in candidates:
            if name in seen_members:
                continue
            n_out = int(module.weight.shape[0])
            fn = tp.prune_conv_out_channels if isinstance(module, nn.Conv2d) else tp.prune_linear_out_channels
            try:
                group = self.graph.get_pruning_group(module, fn, idxs=list(range(n_out)))
            except Exception as exc:                       # noqa: BLE001
                self.rejected.append({"module": name, "reason": f"get_pruning_group: {exc}"})
                continue

            outs: Dict[str, set] = {}
            ins: Dict[str, set] = {}
            for item in group:
                dep = item.dep
                tm = getattr(dep.target, "module", None)
                if not isinstance(tm, (nn.Conv2d, nn.Linear)):
                    continue
                tname = self.names.get(id(tm))
                if tname is None:
                    continue
                idxs = set(self._idx_list(item.idxs))
                (outs if self._is_out(dep) else ins).setdefault(tname, set()).update(idxs)

            if name not in outs:
                outs[name] = set(range(n_out))

            sig = frozenset(outs.keys())
            if sig in seen_sigs:
                for member in outs:
                    seen_members[member] = 1
                continue
            seen_sigs.add(sig)

            members = sorted(outs.keys())
            # tous les membres doivent partager exactement les memes indices 0..n-1
            full = set(range(n_out))
            coupled_ok = all(outs[m] == full for m in members)

            label, stage = self.classifier(members)
            unit = PruningUnit(
                uid=len(self.units),
                members=members,
                consumers=sorted(set(ins.keys()) - set(members)),
                n_channels=n_out,
                label=label,
                stage=stage,
            )

            reason = self._hard_excluded(members)
            if reason is None and not coupled_ok:
                reason = "indices non identiques entre membres couples (concat ?) : non gere"
            if reason is None and label == "other":
                reason = "bloc non classe (ni stem/internal/residual/rpn/fc)"
            if reason is None:
                missing = [m for m in members if m not in self.scores]
                if missing:
                    reason = f"scores Taylor absents pour {missing[:3]}"
            if reason is None:
                lens = {int(self.scores[m].numel()) for m in members}
                if lens != {n_out}:
                    reason = f"taille des scores {sorted(lens)} != canaux {n_out}"
            unit.protected_reason = reason

            if reason is None:
                total = torch.zeros(n_out, dtype=torch.float64)
                for m in members:
                    total += self.scores[m].detach().double().cpu().reshape(-1)
                unit.scores = total.float()

            self.units.append(unit)
            for member in members:
                seen_members[member] = 1

        return self.units

    # ------------------------------------------------------------------ planification
    def plan(self, policy, ratio: Optional[float] = None, clip="policy"):
        r = policy.ratio if ratio is None else ratio
        c = policy.clip if clip == "policy" else clip
        plan, stats = plan_removals(
            self.units, policy.weights, r, min_keep=policy.min_keep, clip=c
        )
        return plan, stats

    def predict_parameters(self, plan: Dict[int, List[int]]) -> Tuple[int, Dict[str, int]]:
        removed_out: Dict[str, int] = {}
        removed_in: Dict[str, List[Tuple[int, int]]] = defaultdict(list)   # nom -> [(n_unite, n_supprimes)]
        for u in self.units:
            k = len(plan.get(u.uid, []))
            for m in u.members:
                removed_out[m] = k
            for c in u.consumers:
                removed_in[c].append((u.n_channels, k))

        per_module: Dict[str, int] = {}
        counted = 0
        for name, mod in self.modules.items():
            if not isinstance(mod, (nn.Conv2d, nn.Linear)):
                continue
            if isinstance(mod, nn.Conv2d) and mod.groups != 1:
                raise NotImplementedError(f"convolution groupee non geree : {name}")
            out0, in0 = self.original_sizes[name]
            out1 = out0 - removed_out.get(name, 0)
            in1 = in0
            ent = removed_in.get(name, [])
            if len(ent) == 1:
                n_u, k = ent[0]
                if in0 % n_u != 0:
                    raise RuntimeError(f"{name}: in={in0} non multiple de {n_u}")
                in1 = in0 - (in0 // n_u) * k
            elif len(ent) > 1:                         # concatenation
                if in0 != sum(n for n, _ in ent):
                    raise RuntimeError(f"{name}: concat non reconnue")
                in1 = in0 - sum(k for _, k in ent)
            ksz = 1
            if isinstance(mod, nn.Conv2d):
                ksz = mod.kernel_size[0] * mod.kernel_size[1]
            n = out1 * in1 * ksz + (out1 if mod.bias is not None else 0)
            per_module[name] = n
            counted += n
        return counted + self.other_params, per_module

    # ------------------------------------------------------------------ ratio <-> reduction
    def predicted_reduction_pct(self, policy, ratio: float) -> float:
        plan, _ = self.plan(policy, ratio)
        total, _ = self.predict_parameters(plan)
        return 100.0 * (self.params_before - total) / self.params_before

    def solve_ratio(self, policy, target_pct: float, tol: float = 0.05) -> float:
        lo, hi = 0.0, 0.99
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            red = self.predicted_reduction_pct(policy, mid)
            if abs(red - target_pct) <= tol:
                return mid
            if red < target_pct:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # ------------------------------------------------------------------ chirurgie
    @staticmethod
    def _sync_frozen_bn(module: nn.Module, removed: List[int], old_channels: int) -> bool:
        norm = getattr(module, "norm", None)
        if norm is None:
            return False
        for b in ("weight", "bias", "running_mean", "running_var"):
            if not torch.is_tensor(getattr(norm, b, None)):
                return False
        if int(norm.running_mean.numel()) != old_channels:      # deja traite par Torch-Pruning
            return False
        removed_set = set(removed)
        keep = torch.tensor([i for i in range(old_channels) if i not in removed_set],
                            dtype=torch.long, device=norm.running_mean.device)
        with torch.no_grad():
            for b in ("weight", "bias", "running_mean", "running_var"):
                setattr(norm, b, getattr(norm, b).index_select(0, keep).clone())
        return True

    def apply(self, plan: Dict[int, List[int]]) -> Dict:
        by_uid = {u.uid: u for u in self.units}
        n_units = 0
        for uid, idxs in plan.items():
            u = by_uid[uid]
            root = self.modules[u.members[0]]
            fn = tp.prune_conv_out_channels if isinstance(root, nn.Conv2d) else tp.prune_linear_out_channels
            try:
                group = self.graph.get_pruning_group(root, fn, idxs=list(idxs))
                group.prune()
            except Exception as exc:                          # noqa: BLE001
                self.rejected.append({"unit": u.members, "reason": f"prune: {exc}"})
                self.log("REJET", u.members[:2], exc)
                continue
            for m in u.members:
                self._sync_frozen_bn(self.modules[m], idxs, u.n_channels)
            n_units += 1
        self.log(f"chirurgie : {n_units}/{len(plan)} unites coupees")
        return {"units_pruned": n_units, "units_planned": len(plan), "rejected": list(self.rejected)}

    # ------------------------------------------------------------------ rapport
    def channel_map(self, plan: Dict[int, List[int]]) -> List[Dict]:
        rows = []
        for u in self.units:
            k = len(plan.get(u.uid, []))
            rows.append({
                "uid": u.uid,
                "label": u.label,
                "stage": u.stage,
                "members": u.members,
                "channels": u.n_channels,
                "removed": k,
                "removed_pct": 100.0 * k / u.n_channels,
                "protected_reason": u.protected_reason,
            })
        return rows

    def unit_pcts(self, plan: Dict[int, List[int]]) -> Dict[str, List[float]]:
        """% de canaux retires PAR UNITE, regroupes par libelle (stem/internal/residual/rpn/fc/fc1/fc2)."""
        out: Dict[str, List[float]] = defaultdict(list)
        for u in self.units:
            if u.protected:
                continue
            pct = 100.0 * len(plan.get(u.uid, [])) / u.n_channels
            out[u.label].append(pct)
            for m in u.members:
                if "box_head.fc1" in m:
                    out["fc1"].append(pct)
                if "box_head.fc2" in m:
                    out["fc2"].append(pct)
        return out

    def summary_by_label(self, plan: Dict[int, List[int]]) -> Dict[str, Dict]:
        agg: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
        for u in self.units:
            if u.protected:
                continue
            key = u.label if u.stage is None else f"{u.label}:{u.stage}"
            if u.label == "fc":                     # le rapport distingue fc6 (fc1) et fc7 (fc2)
                for m in u.members:
                    if "box_head.fc1" in m:
                        key = "fc1 (fc6)"
                    elif "box_head.fc2" in m:
                        key = "fc2 (fc7)"
            agg[key][0] += len(plan.get(u.uid, []))
            agg[key][1] += u.n_channels
        return {k: {"removed": v[0], "channels": v[1], "removed_pct": 100.0 * v[0] / v[1]}
                for k, v in sorted(agg.items())}
