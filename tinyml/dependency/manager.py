from .graph import DependencyGraphManager


class DependencyManager:

    def __init__(
        self,
        model,
        example_inputs,
    ):
        self.graph_manager = (
            DependencyGraphManager(
                model=model,
                example_inputs=example_inputs,
            )
        )

    def build(self):
        return self.graph_manager.build()

    def get_group(
        self,
        module,
        pruning_fn,
        indices,
    ):
        return self.graph_manager.get_pruning_group(
            module,
            pruning_fn,
            indices,
        )

    def validate_group(self, group):
        return self.graph_manager.check_pruning_group(
            group
        )