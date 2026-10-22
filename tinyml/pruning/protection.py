from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List


@dataclass
class PruningProtection:
    """
    Generic protection policy.

    No architecture-specific code is present here.

    The adapter/user provides the actual module names.
    """

    protected_modules: List[str] = field(
        default_factory=list
    )

    ignored_modules: List[str] = field(
        default_factory=list
    )

    def add_protected(
        self,
        *module_names: str,
    ) -> None:

        for name in module_names:
            if name not in self.protected_modules:
                self.protected_modules.append(
                    name
                )

    def add_ignored(
        self,
        *module_names: str,
    ) -> None:

        for name in module_names:
            if name not in self.ignored_modules:
                self.ignored_modules.append(
                    name
                )

    @staticmethod
    def _matches(
        name: str,
        patterns: Iterable[str],
    ) -> bool:

        for pattern in patterns:

            if (
                name == pattern
                or name.startswith(
                    pattern + "."
                )
            ):
                return True

        return False

    def is_protected(
        self,
        module_name: str,
    ) -> bool:

        return self._matches(
            module_name,
            self.protected_modules,
        )

    def is_ignored(
        self,
        module_name: str,
    ) -> bool:

        return self._matches(
            module_name,
            self.ignored_modules,
        )

    def allows_pruning(
        self,
        module_name: str,
    ) -> bool:

        if self.is_protected(
            module_name
        ):
            return False

        if self.is_ignored(
            module_name
        ):
            return False

        return True

    def filter_modules(
        self,
        module_names: Iterable[str],
    ) -> List[str]:

        return [
            name
            for name in module_names
            if self.allows_pruning(name)
        ]