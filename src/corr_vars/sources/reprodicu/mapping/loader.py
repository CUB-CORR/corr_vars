import copy
import json
from importlib import resources

from collections.abc import Callable
from typing import Any


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


UNITS = load_json("units")
