"""This source's bundled variable definitions.

``local_datasource`` is routed **local** in ``concepts/routing.toml``: its
definitions are read from ``vars.json`` here rather than fetched from the CORR
Concepts API, so the source works with no network and no API key. Both files
ship essentially empty — the source is a skeleton, and a deployment fills them
in (or routes the source to an endpoint instead).
"""

from . import variables
from .loader import DEFAULT_VARS, VARS

__all__ = [
    "DEFAULT_VARS",
    "VARS",
    "variables",
]
