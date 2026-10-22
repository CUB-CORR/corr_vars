"""The module namespace a served charite_demo ``py`` snippet executes against.

A ``py`` snippet served by the Concepts API is bare function source: it carries
no imports, so the names it uses have to be waiting for it in its execution
globals. This module *is* that header. It mirrors the namespace of the
production source the demo concepts were ported from, so a ported snippet finds
the same names. It holds exactly three things, and no variable functions:

- the import header,
- the module-level constants variable functions reference, and
- the shared top-level helpers that are called by variable functions but are not
  variables themselves.

Two production names are absent: ``comorb_calc`` (a vendored calculator that is
not published) and ``classify_catheters`` (not needed by the demo concepts).
``comorbidity`` comes from the optional ``pycomorb`` package (``pip install
'corr_vars[charite-demo]'``); without it, calling it raises an ``ImportError``
saying so.

:mod:`corr_vars.sources.charite_demo.py_env` declares which of these names make
up the namespace and builds it from this module; nothing else imports it.
"""

from __future__ import annotations

import os  # noqa: F401
import pickle  # noqa: F401
import tempfile  # noqa: F401

# Every import below is deliberately "unused": binding these names *is* what
# this module is for, so each carries a noqa.
from datetime import datetime, timedelta  # noqa: F401
from functools import partial  # noqa: F401
from pathlib import Path  # noqa: TC003

import ibis  # noqa: F401
import numpy as np  # noqa: F401
import pandas as pd
import polars as pl

import corr_vars.sources.charite_demo.helpers as helpers  # noqa: F401
import corr_vars.utils as utils  # noqa: F401
from corr_vars import __, logger  # noqa: F401
from corr_vars.sources.charite_demo.classify import (  # noqa: F401
    classify_airway_devices,
)
from corr_vars.utils import interval_bucket_agg, merge_consecutive  # noqa: F401

from collections.abc import Callable  # noqa: TC002
from corr_vars.definitions.typing import VariableContext  # noqa: F401
from polars._typing import AsofJoinStrategy  # noqa: F401
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from corr_vars.core import Cohort

try:
    from pycomorb import comorbidity  # noqa: F401
except ImportError:  # pragma: no cover - depends on the optional extra

    def comorbidity(*args: Any, **kwargs: Any) -> Any:
        """Stand-in for ``pycomorb.comorbidity`` when it is not installed.

        Raises:
            ImportError: Always, naming the extra that provides it.
        """
        raise ImportError(
            "Comorbidity scores need the optional 'pycomorb' package: "
            "pip install 'corr_vars[charite-demo]'"
        )


# Data files are addressed by uuid, not by path: a variable function reads its
# mapping file with ``getfile("<uuid>")``. The accessor is injected into the
# snippet's globals by the concepts compiler
# (:func:`corr_vars.concepts.compile.make_getfile`), which resolves the uuid
# against the file manifest served with the config, so the pinned version of the
# file is the one that is read. Annotation only — it binds nothing at runtime,
# which is why ``py_env`` does not declare it in NAMESPACE_NAMES.
getfile: Callable[[str], Path]


# --- Blood gas / oxygenation -------------------------------------------------
# Alveolar gas equation at sea level (mmHg): PAO2 = FiO2 * (Patm - PH2O) - PaCO2 / R
ATMOSPHERIC_PRESSURE_MMHG = 760.0
WATER_VAPOUR_PRESSURE_MMHG = 47.0
RESPIRATORY_QUOTIENT = 0.8
# Oxygen fraction of air, assumed wherever a patient has no ventilator FiO2.
ROOM_AIR_FIO2 = 0.21

# FiO2 is taken from the ventilator (the ABG FiO2 is almost always left at 0.21)
# and is a step function, so the last setting before a sample applies.
FIO2_LOOKBACK = "1h"


def _postcode_state(
    cohort: Cohort, state_col: str, mapping_path: Path, *, alias: str | None = None
) -> pl.DataFrame:
    """Map each observation's postcode (first two digits) to a list of states.

    The mapping (``postcode_mapping.csv``) is not unique — a two-digit prefix can
    span several states — so the states are aggregated into a list per prefix.

    The mapping file is passed in rather than resolved here: this helper is
    shared package code, so it is handed to a served snippet as the module's own
    function object and cannot see the snippet's ``getfile``. The caller — the
    variable function, i.e. the snippet — does the ``getfile`` lookup, which is
    also what makes the file reference visible to the Concepts API.

    Args:
        cohort: The cohort (uses ``obs.postcode`` and the primary key).
        state_col: Mapping column to return (``"state_de"`` or ``"state_en"``).
        mapping_path: Location of ``postcode/postcode_mapping.csv``.
        alias: The name of the output column (defaults to `"value"` if unspecified).

    Returns:
        One row per primary key with a ``value`` list column of state names.
    """
    name = alias or "value"
    mapping = (
        pl.scan_csv(mapping_path, schema_overrides={"postal_code_digits12": pl.Utf8})
        .group_by("postal_code_digits12")
        .agg(pl.col(state_col).unique().sort().alias(name))
    )

    return (
        cohort.obs.lazy()
        .select(cohort.primary_key, "postcode")
        .with_columns(
            pl.col("postcode")
            .cast(pl.Utf8)
            .str.slice(0, 2)
            .alias("postal_code_digits12")
        )
        .join(mapping, on="postal_code_digits12", how="left")
        .select(cohort.primary_key, name)
        .collect()
    )


def fluids_all(var, cohort):
    df = var.required_vars["fluids"].data

    return (
        df.filter(
            (pl.col("attributes").struct.field("volume_unit") == "ml")
            & (pl.col("attributes").struct.field("volume_unit").is_not_null())
        )
        .with_columns(
            [
                pl.col("attributes")
                .struct.field("volume")
                .cast(pl.Float64, strict=False)
                .alias("volume"),
                pl.col("attributes").struct.field("volume_unit").alias("volume_unit"),
            ]
        )
        .select([cohort.primary_key, "recordtime", "volume", "volume_unit"])
    )


def propofol_amount_mg(var, cohort):
    var.data = var.required_vars["propofol"].data

    # Clean up the unit strings
    var.data["value_unit"] = var.data["value_unit"].str.strip()

    # Convert value to float
    var.data["value"] = pd.to_numeric(var.data["value"], errors="coerce")

    # Create new column with standardized value
    var.data["propofol_amount_mg"] = var.data["value"]

    # Return result with primary key and converted value
    return var.data[[cohort.primary_key, "propofol_amount_mg"]]


def propofol_rate_mg_kg_h(var, cohort):
    # Get original data from required_vars
    var.data = var.required_vars["propofol"].data

    # Unnest attributes column
    attributes_df = pd.json_normalize(var.data["attributes"])
    var.data = pd.concat(
        [var.data.reset_index(drop=True), attributes_df.reset_index(drop=True)], axis=1
    )

    # Convert rate_value to float
    var.data["rate_value"] = pd.to_numeric(var.data["rate_value"], errors="coerce")

    # Convert mg/kg/min → mg/kg/h
    mask_min = var.data["rate_unit"] == "mg/kg/min  "
    var.data.loc[mask_min, "rate_value"] = var.data.loc[mask_min, "rate_value"] * 60
    var.data.loc[mask_min, "rate_unit"] = "mg/kg/h    "

    # Convert mg/kg/d → mg/kg/h
    mask_day = var.data["rate_unit"] == "mg/kg/d    "
    var.data.loc[mask_day, "rate_value"] = var.data.loc[mask_day, "rate_value"] / 24
    var.data.loc[mask_day, "rate_unit"] = "mg/kg/h    "

    # Keep only rows with target unit
    var.data = var.data[var.data["rate_unit"] == "mg/kg/h    "].copy()

    # Create new column with standardized value
    var.data["propofol_rate_mg_kg_h"] = var.data["rate_value"]

    # Filter for valid values
    var.data = var.data[(var.data["propofol_rate_mg_kg_h"] <= 50)]

    # Return result
    return var.data[[cohort.primary_key, "propofol_rate_mg_kg_h"]]
