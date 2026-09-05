"""A reprodICU config served by the Concepts API must build the same Variable a bundled one does.

The served form carries two keys the packaged ``mapping/vars.json`` never had: ``type``,
which the importer derives from whether ``calculation`` is present and whether the entry is
dynamic, and ``unit``, folded in from ``units.json``. Both arrive inside ``json_def`` and are
passed to the source's ``VariableLoader`` as keyword arguments alongside the real fields, so
the loader has to consume them rather than forward them into ``Variable.__init__``.
"""

from __future__ import annotations

import pytest

from corr_vars.sources.reprodicu.extract import Variable, VariableLoader
from corr_vars.utils.time import TimeWindow


def _load(**config) -> Variable:
    return VariableLoader(TimeWindow(), var_name="blood_sodium", **config)


SERVED = {
    "type": "native_dynamic",
    "path": "timeseries_labs",
    "column": "Sodium",
    "is_struct": True,
    "filter": "(pl.col('Sodium').struct.field('value').is_not_null())",
    "unit": "mmol/L",
}


def test_served_config_builds_a_variable():
    """The whole json_def goes through as kwargs; `type` and `unit` must not reach __init__."""
    var = _load(**SERVED)

    assert var.path == "timeseries_labs"
    assert var.column == "Sodium"
    assert var.is_struct is True
    assert var.filter == SERVED["filter"]
    assert var.unit == "mmol/L"
    assert not hasattr(
        var, "type"
    ), "`type` restates `dynamic`; it is not a Variable field"


@pytest.mark.parametrize(
    "type_, dynamic",
    [
        ("native_dynamic", True),
        ("derived_dynamic", True),
        ("native_static", False),
        ("derived_static", False),
    ],
)
def test_type_sets_dynamic(type_, dynamic):
    """The four derived types are the cross product of calculation-present and dynamic."""
    config = {"path": "t", "column": "c", "type": type_}
    if not dynamic:
        config["dynamic"] = False
    if type_.startswith("derived"):
        config["calculation"] = "pl.col('x') * 2"

    assert _load(**config).dynamic is dynamic


def test_bundled_config_is_unchanged():
    """A packaged definition carries neither key and must keep working untouched."""
    var = _load(path="timeseries_labs", column="Sodium", is_struct=False)

    assert (
        var.unit is None
    ), "no unit in the definition falls back to the packaged units.json"
    assert var.dynamic is True


def test_unit_falls_back_to_packaged_units():
    """helpers reads var.unit first, then UNITS — so a served unit wins, absent one defers."""
    from corr_vars.sources.reprodicu.mapping import UNITS

    served = _load(**SERVED)
    bundled = _load(path="t", column="c")

    assert (served.unit or UNITS.get("blood_sodium")) == "mmol/L"
    assert (bundled.unit or UNITS.get("blood_sodium")) == UNITS.get("blood_sodium")
