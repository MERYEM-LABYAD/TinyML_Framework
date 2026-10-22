"""
Construction du graphe de dependances Torch-Pruning pour Detectron2 Faster R-CNN,
avec capture des sorties BRUTES du RPN **et du predicteur de boites (box_predictor)**.

Pourquoi : le graphe est trace depuis les sorties du modele. En mode eval sur une image d'exemple,
la tete de regression des boites (roi_heads.box_predictor.bbox_pred) n'est pas reliee aux sorties
finales (aucune detection survivante) : elle est ABSENTE du graphe. Resultat observe :
    "Module Linear(in_features=1024, out_features=12) is not in the dependency graph"
    "mat1 and mat2 shapes cannot be multiplied (1000x635 and 1024x12)"
car fc2 est coupe mais bbox_pred garde ses 1024 entrees.

Meme principe que tinyml/dependency/graph.py (qui n'est pas modifie), avec des hooks supplementaires.
"""
from __future__ import annotations

from typing import Any, List, Sequence

import torch
import torch.nn as nn
import torch_pruning as tp

DEFAULT_HOOKED_MODULES = (
    "proposal_generator.rpn_head",
    "roi_heads.box_predictor",
)


def _get_submodule(model: nn.Module, dotted: str):
    cur = model
    for part in dotted.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur if isinstance(cur, nn.Module) else None


def _collect_tensors(obj, sink: List[torch.Tensor]) -> None:
    if torch.is_tensor(obj):
        sink.append(obj)
    elif isinstance(obj, (tuple, list)):
        for o in obj:
            _collect_tensors(o, sink)
    elif isinstance(obj, dict):
        for o in obj.values():
            _collect_tensors(o, sink)


def build_detectron2_graph(
    model: nn.Module,
    example_inputs: Any,
    hooked_modules: Sequence[str] = DEFAULT_HOOKED_MODULES,
    verbose: bool = True,
) -> tp.DependencyGraph:
    captured: List[torch.Tensor] = []
    handles = []

    for dotted in hooked_modules:
        m = _get_submodule(model, dotted)
        if m is None:
            if verbose:
                print(f"GRAPH: module '{dotted}' absent, hook ignore", flush=True)
            continue

        def hook(module, inputs, outputs, _dotted=dotted):
            _collect_tensors(outputs, captured)

        handles.append(m.register_forward_hook(hook))
        if verbose:
            print(f"GRAPH: hook actif sur {dotted}", flush=True)

    def output_transform(outputs):
        final: List[torch.Tensor] = []
        if isinstance(outputs, (tuple, list)):
            for item in outputs:
                if not isinstance(item, dict):
                    continue
                inst = item.get("instances")
                if inst is None:
                    continue
                for v in inst.get_fields().values():
                    if torch.is_tensor(v):
                        final.append(v)
        return tuple(final + list(captured))

    graph = tp.DependencyGraph()
    try:
        graph.build_dependency(
            model=model,
            example_inputs=example_inputs,
            output_transform=output_transform,
        )
    finally:
        for h in handles:
            h.remove()
        captured.clear()

    if verbose:
        print("GRAPH: noeuds =", len(graph.module2node), flush=True)
    return graph
