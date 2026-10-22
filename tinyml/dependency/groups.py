from typing import Dict, List


class DependencyGroupInspector:

    def __init__(self, dependency_manager):
        self.dependency_manager = dependency_manager

    def inspect(self, group) -> List[Dict[str, str]]:
        return self.dependency_manager.get_group_summary(
            group
        )

    def contains_module(
        self,
        group,
        module_name: str,
    ) -> bool:
        for item in group:
            source_module = item.dep.source.module

            name = getattr(
                source_module,
                "_tinyml_name",
                "",
            )

            if name == module_name:
                return True

        return False