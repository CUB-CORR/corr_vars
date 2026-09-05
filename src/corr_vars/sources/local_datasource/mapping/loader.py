"""What this source's ``mapping`` package reads from disk.

``local_datasource`` is routed **local** in ``concepts/routing.toml``, so its
variable definitions come from the bundled ``vars.json`` rather than the CORR
Concepts API. It is loaded eagerly: importing the package needs no network, no
project and no API key.
"""

from __future__ import annotations

import copy
import json
from importlib import resources

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


def load_json(
    name: str, object_hook: Callable[[dict[Any, Any]], Any] | None = None
) -> Any:
    # For Python 3.9+
    with (
        resources.files(".".join(__name__.split(".")[:-1]))
        .joinpath(f"{name}.json")
        .open("r") as f
    ):
        return copy.deepcopy(json.load(f, object_hook=object_hook))


#: The full ``vars.json`` document: the ``variables`` mapping plus the
#: ``corr_defaults`` block.
VARS = load_json("vars")

#: The cohort's default variable list, keyed by ``"global"`` and by obs level.
DEFAULT_VARS = VARS["corr_defaults"]["default_vars"]
