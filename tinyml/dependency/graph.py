from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch_pruning as tp


class DependencyGraphManager:
    """
    Construit et inspecte le Dependency Graph de Torch-Pruning.
    Pour Detectron2 Faster R-CNN, conserve aussi les sorties brutes du RPN
    afin que les couches RPN soient présentes dans le graphe.
    """

    def __init__(self, model: nn.Module, example_inputs: Any):
        self.model = model
        self.example_inputs = example_inputs
        self.graph: Optional[tp.DependencyGraph] = None

    def build(self) -> tp.DependencyGraph:
        print("GRAPH: creation DependencyGraph", flush=True)

        self.graph = tp.DependencyGraph()
        rpn_head = getattr(
            getattr(self.model, "proposal_generator", None),
            "rpn_head",
            None,
        )

        captured_rpn_outputs = []
        hook_handle = None

        if isinstance(rpn_head, nn.Module):
            def capture_rpn_outputs(module, inputs, outputs):
                captured_rpn_outputs.clear()

                if isinstance(outputs, (tuple, list)):
                    for output_group in outputs:
                        if isinstance(output_group, (tuple, list)):
                            captured_rpn_outputs.extend(
                                tensor
                                for tensor in output_group
                                if torch.is_tensor(tensor)
                            )
                        elif torch.is_tensor(output_group):
                            captured_rpn_outputs.append(output_group)

            hook_handle = rpn_head.register_forward_hook(capture_rpn_outputs)
            print("GRAPH: hook RPN active", flush=True)

        def output_transform(outputs):
            final_tensors = []

            if isinstance(outputs, (tuple, list)):
                for item in outputs:
                    if not isinstance(item, dict):
                        continue

                    instances = item.get("instances")
                    if instances is None:
                        continue

                    for value in instances.get_fields().values():
                        if torch.is_tensor(value):
                            final_tensors.append(value)

            # Include RPN logits/deltas in the tracing roots.
            # Keep the detector's final outputs as well.
            return tuple(final_tensors + list(captured_rpn_outputs))

        try:
            self.graph.build_dependency(
                model=self.model,
                example_inputs=self.example_inputs,
                output_transform=output_transform,
            )
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        print(
            "GRAPH: APRES build_dependency; noeuds =",
            len(self.graph.module2node),
            flush=True,
        )
        return self.graph

    def _check_graph(self) -> tp.DependencyGraph:
        if self.graph is None:
            raise RuntimeError(
                "Dependency Graph has not been built yet. Call build() first."
            )
        return self.graph

    def get_pruning_group(
        self,
        module: nn.Module,
        pruning_fn,
        idxs: List[int],
    ):
        graph = self._check_graph()
        return graph.get_pruning_group(module, pruning_fn, idxs=idxs)

    def check_pruning_group(self, group) -> bool:
        self._check_graph()

        if group is None:
            return False

        try:
            return len(group) > 0
        except TypeError:
            return False

    def get_group_summary(self, group) -> List[Dict[str, str]]:
        self._check_graph()
        summary = []

        for item in group:
            dependency = item.dep
            source = dependency.source
            source_module = source.module

            source_name = getattr(
                source_module,
                "_tinyml_name",
                source_module.__class__.__name__,
            )

            summary.append(
                {
                    "module": source_name,
                    "module_type": source_module.__class__.__name__,
                    "dependency": str(dependency),
                }
            )

        return summary

    def count_operations(self) -> int:
        graph = self._check_graph()
        return len(graph.module2node)
