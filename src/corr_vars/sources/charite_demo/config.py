"""Configuration schema and defaults for the ``charite_demo`` source.

``charite_demo`` reads the public, fully synthetic CHARITÉ mock dataset — a
handful of Parquet tables shaped like the Charité Health Data Platform — into an
in-process DuckDB. The data lives either behind a URL (the default, read lazily
over HTTP range requests) or in a local directory produced by
:func:`corr_vars.sources.charite_demo.download`:

.. code-block:: python

    Cohort(obs_level="icu_stay", sources={"charite_demo": {}})
    Cohort(obs_level="icu_stay", sources={"charite_demo": {"path": "~/data/charite_mockdata"}})
"""

from datetime import datetime

from corr_vars.sources.charite_demo.mapping import TABLES

from typing import Final, Literal, TypedDict, get_args

#: Where the published dataset lives. ``path`` defaults to it.
DEFAULT_PATH: Final[str] = "https://demo-data.corr.icu/v1"

#: Schema the dataset's tables are registered under in DuckDB.
DEFAULT_DATABASE: Final[str] = "charite_mockdata"

ExtraIcuColumns = Literal[
    "lfd_nr",
    "pflege_oe_nicename",
    "fachliche_oe_nicename",
    "bw_typ",
    "bw_kategorie",
    "room_number",
    "bed_number",
]

_extra_icu_columns_literal_values = set(get_args(ExtraIcuColumns))
_extra_icu_columns_tables_values = {
    col.removeprefix("attributes_")
    for col in TABLES["it_ishmed_bewegung"]
    if col.startswith("attributes_")
    and col.removeprefix("attributes_")
    not in [
        "pflege_oe_id",  # Already added as icu_id
        "fachliche_oe_id",  # Already added as bw_fach_oe
        "bw_art",  # Already added as bw_patient_origin
    ]
}
assert (
    _extra_icu_columns_literal_values == _extra_icu_columns_tables_values
), f"The ExtraIcuColumns literal values {sorted(_extra_icu_columns_literal_values)} don't match {sorted(_extra_icu_columns_tables_values)}"


class FilterDictPartial(TypedDict, total=False):
    extraction_start_date: str | datetime | None
    extraction_end_date: str | datetime | None
    additional_filters: str
    include_adults_only: bool
    exclude_brain_death: bool


class IcuDictPartial(TypedDict, total=False):
    merge_consecutive: Literal["any", "same_icu_only"] | bool
    extra_columns: list[ExtraIcuColumns]


class SourceDictPartial(TypedDict, total=False):
    path: str
    database: Literal["charite_mockdata"]
    filter: FilterDictPartial
    icu_stay: IcuDictPartial


class FilterDict(TypedDict):
    extraction_start_date: str | datetime | None
    extraction_end_date: str | datetime | None
    additional_filters: str
    include_adults_only: bool
    exclude_brain_death: bool


class IcuDict(TypedDict):
    merge_consecutive: Literal["any", "same_icu_only"] | bool
    extra_columns: list[ExtraIcuColumns]


class SourceDict(TypedDict):
    path: str
    database: Literal["charite_mockdata"]
    filter: FilterDict
    icu_stay: IcuDict


DEFAULTS: SourceDict = {
    "path": DEFAULT_PATH,
    "database": DEFAULT_DATABASE,
    "filter": {
        "additional_filters": "",
        "include_adults_only": True,
        "exclude_brain_death": True,
        "extraction_end_date": None,
        "extraction_start_date": None,
    },
    "icu_stay": {"merge_consecutive": True, "extra_columns": []},
}

#: Legacy top-level keys, relocated to their canonical nested position by
#: ``config_loader.load_default_config_data()``.
MIGRATIONS = {
    "merge_consecutive": ("icu_stay", "merge_consecutive"),
}
