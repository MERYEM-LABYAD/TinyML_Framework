from __future__ import annotations

import copy
import gc
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch_pruning as tp

from tinyml.dependency.graph import (
    DependencyGraphManager,
)
from tinyml.importance.global_ranker import (
    GlobalRanker,
)
from tinyml.pruning.policy import (
    PruningPolicy,
)


class StructuredPruner:
    """
    Architecture-agnostic structured pruning engine.

    Scientific flow:

        Taylor importance
              |
              v
        global ranking
              |
              v
        candidate channel
              |
              v
        Torch-Pruning dependency group
              |
              v
        structural surgery
              |
              v
        FrozenBatchNorm synchronization
              |
              v
        forward validation
              |
              v
        dependency graph rebuild

    Important implementation rule:

        The DependencyGraph is the authority for structural
        coupling. We never manually reproduce residual/FPN
        dependency propagation.

    The graph manager in tinyml.dependency.graph is intentionally
    reused unchanged because its Detectron2 output_transform has
    already been validated.
    """

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(
        self,
        model,
        importance_scores,
        policy: PruningPolicy,
        example_inputs,
    ):

        self.model = model
        self.importance_scores = importance_scores
        self.policy = policy
        self.example_inputs = example_inputs

        self.policy.validate()

        # IMPORTANT:
        # Do not replace this with a direct tp.DependencyGraph.
        self.graph_manager = (
            DependencyGraphManager(
                model=self.model,
                example_inputs=self.example_inputs,
            )
        )

        self.graph = (
            self.graph_manager.build()
        )

        self.modules = dict(
            self.model.named_modules()
        )

        self.accepted_groups: List[
            Dict[str, Any]
        ] = []

        self.rejected_groups: List[
            Dict[str, Any]
        ] = []

        self.pruned_groups: List[
            Dict[str, Any]
        ] = []

        # Physical channel signatures already removed.
        self._removed_signatures = set()

    # ============================================================
    # BASIC UTILITIES
    # ============================================================

    def _parameter_count(self) -> int:

        return sum(
            parameter.numel()
            for parameter in self.model.parameters()
        )

    # ============================================================
    # GRAPH REBUILD
    # ============================================================

    def _build_graph(
        self,
    ) -> tp.DependencyGraph:

        print(
            "GRAPH: RELEASE ancien DependencyGraph",
            flush=True,
        )

        old_graph = self.graph
        self.graph = None

        if old_graph is not None:
            del old_graph

        gc.collect()

        print(
            "GRAPH: RELEASE ancien DependencyGraph OK",
            flush=True,
        )

        print(
            "GRAPH: création nouveau DependencyGraphManager",
            flush=True,
        )

        old_manager = self.graph_manager
        self.graph_manager = None

        if old_manager is not None:
            del old_manager

        gc.collect()

        print(
            "GRAPH: ancien manager libéré",
            flush=True,
        )

        self.graph_manager = (
            DependencyGraphManager(
                model=self.model,
                example_inputs=self.example_inputs,
            )
        )

        print(
            "GRAPH: nouveau manager créé",
            flush=True,
        )

        self.graph = (
            self.graph_manager.build()
        )

        print(
            "GRAPH: nouveau DependencyGraph construit",
            flush=True,
        )

        self.modules = dict(
            self.model.named_modules()
        )

        return self.graph

    # ============================================================
    # MODULE NAME
    # ============================================================

    def _module_name(
        self,
        module: nn.Module,
    ) -> str:

        tinyml_name = getattr(
            module,
            "_tinyml_name",
            None,
        )

        if tinyml_name:
            return str(tinyml_name)

        for name, candidate in (
            self.model.named_modules()
        ):

            if candidate is module:
                return name

        return module.__class__.__name__

    # ============================================================
    # POLICY
    # ============================================================

    def _allows_pruning(
        self,
        name: str,
    ) -> bool:

        if self.policy.is_protected(name):
            return False

        if any(
            not isinstance(pattern, str)
            for pattern in self.policy.ignored_modules
        ):
            print(
                "ERREUR POLICY: ignored_modules contient des objets non-string",
                [
                    (type(pattern).__name__, repr(pattern))
                    for pattern in self.policy.ignored_modules
                ],
                flush=True,
            )
            raise TypeError(
                "PruningPolicy.ignored_modules doit contenir uniquement des noms de modules (str)."
            )

        if self.policy.is_ignored(name):
            return False

        return True

    # ============================================================
    # MODULE ACCESS
    # ============================================================

    def _get_module(
        self,
        name: str,
    ) -> Optional[nn.Module]:

        current = self.model

        if not name:
            return current

        for part in name.split("."):

            if not hasattr(
                current,
                part,
            ):
                return None

            current = getattr(
                current,
                part,
            )

        return current

    # ============================================================
    # CANDIDATE MODULES
    # ============================================================

    def _candidate_modules(
        self,
    ) -> List[str]:

        names = []

        for name, module in (
            self.model.named_modules()
        ):

            if not isinstance(
                module,
                (nn.Conv2d, nn.Linear),
            ):
                continue

            if not self._allows_pruning(
                name
            ):
                continue

            if name not in self.importance_scores:
                continue

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

            if (
                channels
                <= self.policy.min_channels
            ):
                continue

            names.append(name)

        return names

    # ============================================================
    # DEPENDENCY GROUP
    # ============================================================

    def _get_pruning_group(
        self,
        module: nn.Module,
        indices: List[int],
    ):

        if self.graph is None:
            raise RuntimeError(
                "Dependency graph is not built."
            )

        if isinstance(
            module,
            nn.Conv2d,
        ):

            pruning_fn = (
                tp.prune_conv_out_channels
            )

        elif isinstance(
            module,
            nn.Linear,
        ):

            pruning_fn = (
                tp.prune_linear_out_channels
            )

        else:

            raise TypeError(
                "Unsupported pruning module: "
                f"{module.__class__.__name__}"
            )

        return self.graph.get_pruning_group(
            module,
            pruning_fn,
            idxs=indices,
        )

    # ============================================================
    # INDEX NORMALIZATION
    # ============================================================

    def _normalize_indices(self, indices):
        """Normalize pruning indices from Torch-Pruning GroupItem/list/tensor."""
        if indices is None:
            return []

        # Torch-Pruning 1.6.x GroupItem stores pruning indices in `.idxs`.
        if hasattr(indices, "idxs") and not callable(getattr(indices, "idxs")):
            indices = indices.idxs

        # Tensor of indices
        if torch.is_tensor(indices):
            if indices.numel() == 0:
                return []
            return [
                int(x)
                for x in indices.detach().cpu().reshape(-1).tolist()
            ]

        # Single integer
        if isinstance(indices, int):
            return [int(indices)]

        # List / tuple / set of indices or GroupItems
        if isinstance(indices, (list, tuple, set)):
            normalized = []

            for item in indices:

                # GroupItem
                if hasattr(item, "idxs") and not callable(getattr(item, "idxs")):
                    normalized.extend(
                        self._normalize_indices(item.idxs)
                    )

                # Tensor element/container
                elif torch.is_tensor(item):
                    normalized.extend(
                        self._normalize_indices(item)
                    )

                # Integer
                elif isinstance(item, int):
                    normalized.append(int(item))

                else:
                    raise TypeError(
                        f"Unsupported pruning index type: "
                        f"{type(item).__name__}: {item!r}"
                    )

            return normalized

        raise TypeError(
            f"Unsupported pruning index type: "
            f"{type(indices).__name__}: {indices!r}"
        )

    # ============================================================
    # VALID INDICES
    # ============================================================

    def _valid_indices(
        self,
        module: nn.Module,
        requested,
    ) -> List[int]:

        if isinstance(
            module,
            nn.Conv2d,
        ):

            channels = int(
                module.out_channels
            )

        elif isinstance(
            module,
            nn.Linear,
        ):

            channels = int(
                module.out_features
            )

        else:
            return []

        requested = (
            self._normalize_indices(
                requested
            )
        )

        valid = sorted(
            {
                index
                for index in requested
                if 0 <= index < channels
            }
        )

        maximum_prunable = max(
            0,
            channels
            - self.policy.min_channels,
        )

        if len(valid) > maximum_prunable:

            valid = valid[
                :maximum_prunable
            ]

        return valid

    # ============================================================
    # GROUP SAFETY
    # ============================================================

    def _group_is_safe(
        self,
        group,
    ) -> bool:

        if group is None:
            return False

        try:
            return len(group) > 0
        except TypeError:
            return False

    # ============================================================
    # GROUP SIGNATURE
    # ============================================================

    def _group_signature(
        self,
        group,
    ) -> Tuple:

        """
        Build a stable logical signature from the actual
        dependency group.

        This is what prevents:

            res3.0.conv3[0]

        and

            res3.0.shortcut[0]

        from being treated as two independent pruning actions.

        The DependencyGraph itself remains authoritative.
        """

        members = []

        for item in group:

            dependency = item.dep

            source = dependency.source
            target = dependency.target

            source_module = getattr(
                source,
                "module",
                None,
            )

            target_module = getattr(
                target,
                "module",
                None,
            )

            source_name = (
                self._module_name(
                    source_module
                )
                if source_module is not None
                else "UNKNOWN"
            )

            target_name = (
                self._module_name(
                    target_module
                )
                if target_module is not None
                else "UNKNOWN"
            )

            indices = tuple(
                self._normalize_indices(
                    item.idxs
                    if hasattr(
                        item,
                        "idxs",
                    )
                    else []
                )
            )

            members.append(
                (
                    source_name,
                    target_name,
                    str(dependency),
                    indices,
                )
            )

        return tuple(
            sorted(
                members,
                key=str,
            )
        )

    # ============================================================
    # PROTECTED OUTPUT
    # ============================================================

    def _group_touches_protected_output(
        self,
        group,
    ) -> bool:

        protected_names = set(
            getattr(
                self.policy,
                "protected_modules",
                [],
            )
        )

        protected_names.update(
            {
                "backbone.fpn_lateral2",
                "backbone.fpn_lateral3",
                "backbone.fpn_lateral4",
                "backbone.fpn_lateral5",
                "backbone.fpn_output2",
                "backbone.fpn_output3",
                "backbone.fpn_output4",
                "backbone.fpn_output5",
            }
        )

        for item in group:

            dependency = item.dep

            source = dependency.source
            target = dependency.target

            source_module = source.module
            target_module = target.module

            source_name = (
                self._module_name(
                    source_module
                )
            )

            target_name = (
                self._module_name(
                    target_module
                )
            )

            dependency_text = str(
                dependency
            )

            if (
                target_name in protected_names
                and
                "prune_out_channels"
                in dependency_text
                and
                "prune_in_channels"
                not in dependency_text
            ):
                return True

            if (
                source_name in protected_names
                and
                "prune_out_channels"
                in dependency_text
                and
                "prune_in_channels"
                not in dependency_text
            ):
                return True

        return False

    # ============================================================
    # FROZEN BATCH NORM
    # ============================================================

    def _collect_frozen_bn_targets(
        self,
        group,
    ):

        targets = {}

        for dependency, dep_indices in group:

            dependency_text = str(
                dependency
            )

            parts = dependency_text.split(
                " => ",
                1,
            )

            if len(parts) != 2:
                continue

            source_text = parts[0].strip()
            target_text = parts[1].strip()

            source = getattr(
                dependency,
                "source",
                None,
            )

            target = getattr(
                dependency,
                "target",
                None,
            )

            source_module = getattr(
                source,
                "module",
                None,
            )

            target_module = getattr(
                target,
                "module",
                None,
            )

            dep_indices = (
                self._normalize_indices(
                    dep_indices
                )
            )

            if not dep_indices:
                continue

            if (
                "prune_out_channels on"
                in source_text
                and isinstance(
                    source_module,
                    nn.Conv2d,
                )
            ):

                self._register_frozen_bn_target(
                    targets,
                    source_module,
                    dep_indices,
                )

            if (
                "prune_out_channels on"
                in target_text
                and isinstance(
                    target_module,
                    nn.Conv2d,
                )
            ):

                self._register_frozen_bn_target(
                    targets,
                    target_module,
                    dep_indices,
                )

        return targets

    def _register_frozen_bn_target(
        self,
        targets,
        conv,
        indices,
    ):

        norm = getattr(
            conv,
            "norm",
            None,
        )

        if norm is None:
            return

        running_mean = getattr(
            norm,
            "running_mean",
            None,
        )

        running_var = getattr(
            norm,
            "running_var",
            None,
        )

        if not torch.is_tensor(
            running_mean
        ):
            return

        if not torch.is_tensor(
            running_var
        ):
            return

        old_channels = int(
            running_mean.numel()
        )

        valid_indices = sorted(
            {
                int(index)
                for index in indices
                if 0 <= int(index)
                < old_channels
            }
        )

        if not valid_indices:
            return

        key = id(conv)

        if key not in targets:

            targets[key] = {
                "module": conv,
                "name": self._module_name(
                    conv
                ),
                "indices": set(),
            }

        targets[key][
            "indices"
        ].update(
            valid_indices
        )

    @staticmethod
    def _sync_frozen_bn(
        conv: nn.Conv2d,
        removed_indices: List[int],
    ) -> None:

        norm = getattr(
            conv,
            "norm",
            None,
        )

        if norm is None:
            return

        required_buffers = (
            "weight",
            "bias",
            "running_mean",
            "running_var",
        )

        for buffer_name in required_buffers:

            if not hasattr(
                norm,
                buffer_name,
            ):
                return

            buffer = getattr(
                norm,
                buffer_name,
            )

            if not torch.is_tensor(
                buffer
            ):
                return

        old_channels = int(
            norm.running_mean.numel()
        )

        removed = sorted(
            {
                int(index)
                for index in removed_indices
                if 0 <= int(index)
                < old_channels
            }
        )

        if not removed:
            return

        keep = [
            index
            for index in range(
                old_channels
            )
            if index not in removed
        ]

        keep_tensor = torch.tensor(
            keep,
            dtype=torch.long,
            device=norm.running_mean.device,
        )

        def select(tensor):

            return (
                tensor
                .index_select(
                    0,
                    keep_tensor,
                )
                .clone()
            )

        with torch.no_grad():

            norm.weight = select(
                norm.weight
            )

            norm.bias = select(
                norm.bias
            )

            norm.running_mean = select(
                norm.running_mean
            )

            norm.running_var = select(
                norm.running_var
            )

    # ============================================================
    # FILTER ZERO THRESHOLD
    # ============================================================

    def _filter_zero_fraction(
        self,
        module: nn.Module,
        channel_idx: int,
    ) -> float:

        """
        A selected structural channel is represented as a complete
        zeroed filter before physical removal.

        Therefore its zero fraction is 1.0.

        This makes FILTER_ZERO_THRESH an explicit gate while
        preserving the structural surgery model.
        """

        if not isinstance(
            module,
            (nn.Conv2d, nn.Linear),
        ):
            return 0.0

        weight = module.weight.detach()

        if (
            channel_idx < 0
            or channel_idx >= weight.shape[0]
        ):
            return 0.0

        channel = weight[channel_idx]

        if channel.numel() == 0:
            return 0.0

        return 1.0

    def _filter_candidate_allowed(
        self,
        module: nn.Module,
        channel_idx: int,
    ) -> bool:

        threshold = float(
            self.policy.filter_zero_thresh
        )

        if not (
            0.0
            < threshold
            <= 1.0
        ):
            raise ValueError(
                "FILTER_ZERO_THRESH must satisfy "
                "0 < threshold <= 1."
            )

        fraction = (
            self._filter_zero_fraction(
                module,
                channel_idx,
            )
        )

        return (
            fraction >= threshold
        )

    # ============================================================
    # FORWARD VALIDATION
    # ============================================================

    def _validate_forward(
        self,
    ) -> None:

        print(
            "VALIDATION: START",
            flush=True,
        )

        was_training = (
            self.model.training
        )

        self.model.eval()

        try:

            inputs = copy.deepcopy(
                self.example_inputs
            )

            with torch.no_grad():

                output = self.model(
                    inputs
                )

            del output
            del inputs

            print(
                "VALIDATION: MODEL FORWARD OK",
                flush=True,
            )

        except Exception as exc:

            print(
                "VALIDATION: PYTHON EXCEPTION:",
                repr(exc),
                flush=True,
            )

            raise

        finally:

            self.model.train(
                was_training
            )

            print(
                "VALIDATION: END",
                flush=True,
            )

    # ============================================================
    # ONE STRUCTURAL GROUP
    # ============================================================

    def _prune_one_group(
        self,
        module_name: str,
        idxs: List[int],
    ) -> Optional[Tuple]:

        idxs = (
            self._normalize_indices(
                idxs
            )
        )

        module = self._get_module(
            module_name
        )

        if module is None:

            self.rejected_groups.append(
                {
                    "module": module_name,
                    "indices": idxs,
                    "reason": "module_not_found",
                }
            )

            return None

        if not isinstance(
            module,
            (nn.Conv2d, nn.Linear),
        ):

            self.rejected_groups.append(
                {
                    "module": module_name,
                    "indices": idxs,
                    "reason": "unsupported_module",
                }
            )

            return None

        valid_indices = (
            self._valid_indices(
                module,
                idxs,
            )
        )

        if not valid_indices:

            self.rejected_groups.append(
                {
                    "module": module_name,
                    "indices": idxs,
                    "reason": "no_valid_indices",
                }
            )

            return None

        try:

            group = (
                self._get_pruning_group(
                    module,
                    valid_indices,
                )
            )

        except Exception as exc:

            self.rejected_groups.append(
                {
                    "module": module_name,
                    "indices": valid_indices,
                    "reason": (
                        "dependency_group_error: "
                        f"{exc}"
                    ),
                }
            )

            return None

        if module_name == "roi_heads.box_head.fc2":
            print("\n========== DIAGNOSTIC DEPENDENCY GROUP ==========")
            for item in group:
                print(
                    "MODULE:",
                    getattr(
                        item.dep.target.module,
                        "_tinyml_name",
                        item.dep.target.module.__class__.__name__,
                    ),
                    "| DEPENDENCY:",
                    str(item.dep),
                    "| INDICES:",
                    item.idxs,
                )
            print("========== FIN DIAGNOSTIC ==========\n", flush=True)

        if not self._group_is_safe(
            group
        ):

            self.rejected_groups.append(
                {
                    "module": module_name,
                    "indices": valid_indices,
                    "reason": (
                        "invalid_dependency_group"
                    ),
                }
            )

            return None

        # --------------------------------------------------------
        # The dependency group is the logical pruning unit.
        # --------------------------------------------------------

        signature = (
            self._group_signature(
                group
            )
        )

        if signature in self._removed_signatures:

            self.rejected_groups.append(
                {
                    "module": module_name,
                    "indices": valid_indices,
                    "reason": (
                        "duplicate_dependency_group"
                    ),
                }
            )

            return None

        if self._group_touches_protected_output(
            group
        ):

            self.rejected_groups.append(
                {
                    "module": module_name,
                    "indices": valid_indices,
                    "reason": (
                        "protected_output_dependency"
                    ),
                }
            )

            return None

        frozen_bn_targets = (
            self._collect_frozen_bn_targets(
                group
            )
        )

        parameters_before = (
            self._parameter_count()
        )

        print(
            "PRUNE: START",
            module_name,
            "indices=",
            valid_indices,
            "parameters_before=",
            parameters_before,
            flush=True,
        )

        # Capture Taylor output-channel indices BEFORE structural surgery.
        taylor_prune_records = self._capture_taylor_score_pruning(group)

        print(
            "DEBUG TAYLOR CAPTURE:",
            "modules =", len(taylor_prune_records),
            "records =", taylor_prune_records,
            flush=True,
        )

        # Structural surgery
        try:
            group.prune()

        except Exception as exc:

            self.rejected_groups.append(
                {
                    "module": module_name,
                    "indices": valid_indices,
                    "reason": (
                        "structural_pruning_error: "
                        f"{exc}"
                    ),
                }
            )

            return None

        print(
            "PRUNE: group.prune() terminé",
            flush=True,
        )


        # Synchronize Taylor scores using the pre-pruning snapshot.
        try:
            self._synchronize_taylor_scores_after_pruning(
                taylor_prune_records
            )
        except Exception as exc:
            raise RuntimeError(
                "Taylor score synchronization failed. "
                f"Module={module_name}, "
                f"indices={valid_indices}, "
                f"error={exc}"
            ) from exc
        
        try:

            for data in (
                frozen_bn_targets.values()
            ):

                indices = sorted(
                    data["indices"]
                )

                if not indices:
                    continue

                print(
                    "PRUNE: synchronisation FrozenBN:",
                    data["name"],
                    "indices=",
                    indices,
                    flush=True,
                )

                self._sync_frozen_bn(
                    data["module"],
                    indices,
                )

        except Exception as exc:

            self.rejected_groups.append(
                {
                    "module": module_name,
                    "indices": valid_indices,
                    "reason": (
                        "frozen_bn_sync_error: "
                        f"{exc}"
                    ),
                }
            )

            raise RuntimeError(
                "FrozenBatchNorm synchronization failed. "
                f"Module={module_name}, "
                f"indices={valid_indices}, "
                f"error={exc}"
            ) from exc

        # --------------------------------------------------------
        # Forward gate
        # --------------------------------------------------------

        try:

            self._validate_forward()

        except Exception as exc:

            self.rejected_groups.append(
                {
                    "module": module_name,
                    "indices": valid_indices,
                    "reason": (
                        "forward_validation_error: "
                        f"{exc}"
                    ),
                }
            )

            raise RuntimeError(
                "Structural pruning produced "
                "an invalid model. "
                f"Module={module_name}, "
                f"indices={valid_indices}, "
                f"error={exc}"
            ) from exc

        parameters_after = (
            self._parameter_count()
        )

        if parameters_after >= parameters_before:

            raise RuntimeError(
                "Structural pruning did not reduce "
                "parameter count. "
                f"Module={module_name}, "
                f"before={parameters_before}, "
                f"after={parameters_after}"
            )

        result = {
            "module": module_name,
            "indices": valid_indices,
            "parameters_before": parameters_before,
            "parameters_after": parameters_after,
            "parameters_removed": (
                parameters_before
                - parameters_after
            ),
            "dependency_group_signature": str(
                signature
            ),
        }

        self.pruned_groups.append(
            result
        )

        self.accepted_groups.append(
            result
        )

        self._removed_signatures.add(
            signature
        )

        return signature


    # ============================================================
    # MAIN PRUNING
    # ============================================================

    def prune(
        self,
    ) -> Dict[str, Any]:

        before_parameters = (
            self._parameter_count()
        )

        if not self.importance_scores:

            raise RuntimeError(
                "No importance scores were supplied. "
                "The scientific pruning pipeline requires "
                "Taylor/group importance before pruning."
            )

        self.pruned_groups.clear()
        self.rejected_groups.clear()
        self.accepted_groups.clear()
        self._removed_signatures.clear()

        print(
            "============================================================",
            flush=True,
        )

        print(
            "STRUCTURED PRUNER START",
            flush=True,
        )

        print(
            "PARAMETERS BEFORE =",
            before_parameters,
            flush=True,
        )

        print(
            "GLOBAL_PRUNE_RATIO =",
            self.policy.ratio,
            flush=True,
        )

        print(
            "FILTER_ZERO_THRESH =",
            self.policy.filter_zero_thresh,
            flush=True,
        )

        print(
            "MIN_KEEP =",
            self.policy.min_channels,
            flush=True,
        )

        # --------------------------------------------------------
        # IMPORTANT:
        #
        # The ranker is rebuilt after every accepted structural
        # operation. This prevents stale channel indices.
        # --------------------------------------------------------

        target_parameter_reduction = (
            self.policy.target_parameter_reduction
        )

        if (
            target_parameter_reduction
            > 0.0
        ):

            target_parameters = int(
                before_parameters
                * (
                    1.0
                    - target_parameter_reduction
                )
            )

        else:

            target_parameters = int(
                before_parameters
                * (
                    1.0
                    - self.policy.ratio
                )
            )

        print(
            "TARGET PARAMETERS =",
            target_parameters,
            flush=True,
        )

        # --------------------------------------------------------
        # Dynamic structural pruning loop
        # --------------------------------------------------------

        iteration = 0

        while True:

            current_parameters = (
                self._parameter_count()
            )

            if (
                current_parameters
                <= target_parameters
            ):
                break

            iteration += 1

            print(
                "",
                flush=True,
            )

            print(
                "PRUNE ITERATION =",
                iteration,
                flush=True,
            )

            # ----------------------------------------------------
            # Fresh modules after previous surgery.
            # ----------------------------------------------------

            current_modules = dict(
                self.model.named_modules()
            )

            ranker = GlobalRanker(
                importance_scores=(
                    self.importance_scores
                ),
                modules=current_modules,
                aggressiveness=(
                    self.policy.aggressiveness
                ),
                percentile_clip=(
                    getattr(
                        self.policy,
                        "percentile_clip",
                        None,
                    )
                ),
            )

            candidates = ranker.rank(
                current_modules
            )

            print(
                "PRUNE: fresh candidates =",
                len(candidates),
                flush=True,
            )

            
            accepted_this_iteration = False

            # Batch several low-importance channels from the same module.
            # Torch-Pruning remains responsible for dependency propagation.
            BATCH_SIZE = 64
            candidates_by_module = {}

            for candidate in candidates:
                name = candidate.module_name
                module = self._get_module(name)

                if module is None or not self._allows_pruning(name):
                    continue

                if isinstance(module, nn.Conv2d):
                    channels = int(module.out_channels)
                elif isinstance(module, nn.Linear):
                    channels = int(module.out_features)
                else:
                    continue

                if channels <= self.policy.min_channels:
                    continue

                idx = int(candidate.channel_idx)
                if not 0 <= idx < channels:
                    continue

                if not self._filter_candidate_allowed(module, idx):
                    continue

                indices = candidates_by_module.setdefault(name, [])
                if idx not in indices and len(indices) < BATCH_SIZE:
                    indices.append(idx)

            # Try candidate modules in global ranking order.
            for module_name, indices in candidates_by_module.items():
                module = self._get_module(module_name)
                if module is None:
                    continue

                # Keep a safety margin for the configured minimum channels.
                if isinstance(module, nn.Conv2d):
                    channels = module.out_channels
                elif isinstance(module, nn.Linear):
                    channels = module.out_features
                else:
                    continue

                max_remove = max(0, channels - self.policy.min_channels)
                indices = sorted(set(indices))[:max_remove]

                if not indices:
                    continue

                print(
                    "PRUNE: BATCH CANDIDATE",
                    module_name,
                    "channels=",
                    indices,
                    "batch_size=",
                    len(indices),
                    flush=True,
                )

                signature = self._prune_one_group(module_name, indices)
                if signature is None:
                    continue

                accepted_this_iteration = True

                print(
                    "PRUNE: BATCH ACCEPTED",
                    module_name,
                    "channels_removed=",
                    len(indices),
                    flush=True,
                )

                try:
                    self._build_graph()
                except Exception as exc:
                    raise RuntimeError(
                        "Dependency graph rebuild failed after pruning "
                        f"{module_name}: {exc}"
                    ) from exc

                current_parameters = self._parameter_count()
                removed = before_parameters - current_parameters
                reduction = 100.0 * removed / before_parameters

                print(
                    "PRUNE: parameters =",
                    current_parameters,
                    flush=True,
                )
                print(
                    "PRUNE: reduction =",
                    f"{reduction:.4f}%",
                    flush=True,
                )
                break            # ----------------------------------------------------
            # No safe candidate remains.
            # ----------------------------------------------------

            if not accepted_this_iteration:

                print(
                    "PRUNE: NO VALID CANDIDATE REMAINS",
                    flush=True,
                )

                break

        # ========================================================
        # FINAL STATISTICS
        # ========================================================

        after_parameters = (
            self._parameter_count()
        )

        parameters_removed = (
            before_parameters
            - after_parameters
        )

        reduction = (
            100.0
            * parameters_removed
            / before_parameters
        )

        target_reduction_percent = (
            100.0
            * (
                1.0
                - target_parameters
                / before_parameters
            )
        )

        target_reached = (
            after_parameters
            <= target_parameters
        )

        print(
            "",
            flush=True,
        )

        print(
            "============================================================",
            flush=True,
        )

        print(
            "STRUCTURED PRUNER END",
            flush=True,
        )

        print(
            "PARAMETERS BEFORE =",
            before_parameters,
            flush=True,
        )

        print(
            "PARAMETERS AFTER  =",
            after_parameters,
            flush=True,
        )

        print(
            "PARAMETERS REMOVED =",
            parameters_removed,
            flush=True,
        )

        print(
            "PARAMETER REDUCTION =",
            f"{reduction:.4f}%",
            flush=True,
        )

        print(
            "TARGET REACHED =",
            target_reached,
            flush=True,
        )

        print(
            "============================================================",
            flush=True,
        )

        return {
            "before_parameters": (
                before_parameters
            ),
            "after_parameters": (
                after_parameters
            ),
            "parameters_removed": (
                parameters_removed
            ),
            "parameter_reduction_percent": (
                reduction
            ),
            "target_reduction_percent": (
                target_reduction_percent
            ),
            "target_reached": (
                target_reached
            ),
            "global_prune_ratio": (
                self.policy.ratio
            ),
            "filter_zero_thresh": (
                self.policy.filter_zero_thresh
            ),
            "min_keep": (
                self.policy.min_channels
            ),
            "pruned_modules": [
                item["module"]
                for item in self.pruned_groups
            ],
            "pruned_groups": list(
                self.pruned_groups
            ),
            "accepted_groups": list(
                self.accepted_groups
            ),
            "rejected_groups": list(
                self.rejected_groups
            ),
        }

    def _capture_taylor_score_pruning(self, group):
        """
        Capture toutes les suppressions de canaux de sortie
        déclarées dans le groupe de dépendances, avant group.prune().
        """
        records = {}

        for item in group:
            dep = item.dep
            target = getattr(dep, "target", None)
            module = getattr(target, "module", None)

            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue

            # Torch-Pruning peut exposer les handlers sous des formes
            # différentes. On utilise la représentation de dépendance,
            # déjà exploitée par la synchronisation FrozenBN.
            dependency_text = str(dep)
            parts = dependency_text.split(" => ", 1)

            if len(parts) != 2:
                continue

            target_text = parts[1].strip()

            if "prune_out_channels on" not in target_text:
                continue

            if "prune_in_channels on" in target_text:
                continue

            raw_indices = self._normalize_indices(item.idxs)
            if not raw_indices:
                continue

            module_name = self._module_name(module)

            records.setdefault(module_name, set()).update(
                int(index) for index in raw_indices
            )

        result = {
            name: sorted(indices)
            for name, indices in records.items()
        }

        # Vérifier les scores AVANT la chirurgie structurelle.
        for name, indices in result.items():
            scores = self.importance_scores.get(name)

            if scores is None:
                continue

            score_count = (
                scores.numel()
                if torch.is_tensor(scores)
                else len(scores)
            )

            invalid = [
                index for index in indices
                if index < 0 or index >= score_count
            ]

            if invalid:
                raise RuntimeError(
                    f"Indices Taylor invalides pour {name}: "
                    f"{invalid}; nombre de scores={score_count}"
                )

        return result

    def _synchronize_taylor_scores_after_pruning(self, records):
        """
        Supprime les scores des canaux effectivement retirés.
        Les indices sont ceux du modèle AVANT la chirurgie.
        """
        for module_name, indices in records.items():
            scores = self.importance_scores.get(module_name)

            if scores is None:
                continue

            if not torch.is_tensor(scores):
                scores = torch.as_tensor(scores)

            scores = scores.detach().cpu().reshape(-1)

            if scores.numel() == 0:
                continue

            indices = sorted(set(int(i) for i in indices))

            invalid = [
                i for i in indices
                if i < 0 or i >= scores.numel()
            ]

            if invalid:
                raise RuntimeError(
                    f"Synchronisation Taylor impossible pour {module_name}: "
                    f"indices invalides={invalid}, "
                    f"nombre de scores={scores.numel()}"
                )

            keep_mask = torch.ones(
                scores.numel(),
                dtype=torch.bool,
            )
            keep_mask[
                torch.tensor(indices, dtype=torch.long)
            ] = False

            updated_scores = scores[keep_mask]

            self.importance_scores[module_name] = updated_scores

            print(
                "PRUNE: Taylor scores synchronized:",
                module_name,
                "removed=", indices,
                "remaining=", updated_scores.numel(),
                flush=True,
            )