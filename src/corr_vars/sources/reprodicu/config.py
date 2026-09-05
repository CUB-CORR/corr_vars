from typing import TypedDict


class SourceDictPartial(TypedDict, total=False):
    path: str | None
    exclude_datasets: list[str]
    include_datasets: list[str]


class SourceDict(TypedDict):
    path: str | None
    exclude_datasets: list[str]
    include_datasets: list[str]


DEFAULTS: SourceDict = {
    "path": None,
    "exclude_datasets": [],
    "include_datasets": [],
}
