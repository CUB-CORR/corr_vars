import importlib
import inspect
import os
import pkgutil

from . import cohort_loader, config_loader, var_loader

from collections.abc import Iterable
from types import ModuleType

SOURCES = [
    name
    for _, name, ispkg in pkgutil.iter_modules(__path__)
    if ispkg
    and os.path.isdir(os.path.join(os.path.dirname(__file__), name))
    and os.path.isfile(os.path.join(os.path.dirname(__file__), name, "__init__.py"))
]


def get_src_module(source: str, submodule: str | None = None) -> ModuleType:
    sourced_module = f"corr_vars.sources.{source}"
    return importlib.import_module(
        f"{sourced_module}.{submodule}" if submodule else sourced_module
    )


def get_src_module_mapping(
    include_sources: Iterable[str] | None = None, submodule: str | None = None
) -> dict[str, ModuleType]:
    if include_sources is None:
        include_sources = SOURCES
    return {
        src: get_src_module(source=src, submodule=submodule) for src in include_sources
    }


def guess_variable_source(variable: object) -> str | None:
    """Guess the source of a variable."""
    variable_module = inspect.getmodule(variable)
    if variable_module is None:
        return None

    prefix = f"{__name__}."
    variable_module_name = variable_module.__name__
    if not variable_module_name.startswith(prefix):
        return None

    return variable_module_name.removeprefix(prefix).split(".", maxsplit=1)[0]


__all__ = ["SOURCES", "cohort_loader", "config_loader", "var_loader"]
