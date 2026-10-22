"""A tiny synthetic charite_demo dataset, written to a temporary folder.

The tables follow the published layout (``manifest.json`` + Parquet under
``charite_mockdata/``) and the HDP column types: ids and free-text values are
strings, timestamps are naive local times, engine ids are BIGINT. Timestamps
*inside* ``it_copra6_hierarchy_v2.c_value`` are UTC strings, as in production.

Patients and what they exercise:

- ``P1`` / case ``F1``: ED admission, two ICU movements 2 h apart (merged into one
  stay by ``merge_consecutive``), a ward stay, a third ICU movement days later.
  Carries the labs, the CoPRA hierarchy records and a surgery.
- ``P2`` / case ``F2``: one cardiac-ICU stay ending in death, plus a companion
  movement (``BM``) on an ICU ward that must not become a stay.
- ``P3`` / case ``F3``: ward-only stay — neither an ICU nor a hospital-stay
  observation.
- ``P4`` / case ``F4``: a child on the paediatric ICU — excluded by default.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from corr_vars.sources.charite_demo import helpers

P1, P2, P3, P4 = "0000000001", "0000000002", "0000000003", "0000000004"
F1, F2, F3, F4 = "0000000101", "0000000102", "0000000103", "0000000104"


def dt(text: str) -> datetime:
    """Parse ``YYYY-MM-DD HH:MM`` into a naive datetime."""
    return datetime.strptime(text, "%Y-%m-%d %H:%M")


STR, TS, INT = pl.Utf8, pl.Datetime("us"), pl.Int64


def _patient() -> pl.DataFrame:
    schema = {
        "c_patnr": STR,
        "c_gender": STR,
        "c_birthdate": TS,
        "c_nationality": STR,
        "c_countryofbirth": STR,
        "c_country": STR,
        "c_postcode": STR,
        "c_datetimeofdeath": TS,
    }
    rows = [
        (P1, "M", dt("1960-05-01 00:00"), "DE", "DE", "DE", "10117", None),
        (
            P2,
            "W",
            dt("1950-03-01 00:00"),
            "DE",
            "DE",
            "DE",
            "13353",
            dt("2024-02-03 06:00"),
        ),
        (P3, "W", dt("1980-01-01 00:00"), "DE", "DE", "DE", "10115", None),
        (P4, "M", dt("2014-01-01 00:00"), "DE", "DE", "DE", "10117", None),
    ]
    return pl.DataFrame(rows, schema=schema, orient="row")


def _fall() -> pl.DataFrame:
    schema = {
        "c_patnr": STR,
        "c_falnr": STR,
        "c_aufnahme": TS,
        "c_entlassung": TS,
        "c_entlassungsart": STR,
        "c_admission_hospital": STR,
        "c_discharge_hospital": STR,
        "c_age": INT,
        "c_dod": TS,
    }
    rows = [
        (
            P1,
            F1,
            dt("2024-01-01 08:00"),
            dt("2024-01-20 12:00"),
            "E",
            None,
            None,
            63,
            None,
        ),
        (
            P2,
            F2,
            dt("2024-02-01 09:00"),
            dt("2024-02-03 06:00"),
            "T",
            None,
            None,
            73,
            dt("2024-02-03 06:00"),
        ),
        (
            P3,
            F3,
            dt("2024-03-01 09:00"),
            dt("2024-03-05 09:00"),
            "E",
            None,
            None,
            44,
            None,
        ),
        (
            P4,
            F4,
            dt("2024-04-01 09:00"),
            dt("2024-04-05 09:00"),
            "E",
            None,
            None,
            10,
            None,
        ),
    ]
    return pl.DataFrame(rows, schema=schema, orient="row")


def _bewegung() -> pl.DataFrame:
    schema = {
        "c_patnr": STR,
        "c_falnr": STR,
        "c_laufende_nummer": INT,
        "c_begin": TS,
        "c_ende": TS,
        "c_pflege_oe_id": STR,
        "c_fachliche_oe_id": STR,
        "c_pflege_oe": STR,
        "c_fachliche_oe": STR,
        "c_bewegungsart": STR,
        "c_bewegungstyp": STR,
        "c_bewegungskategory": STR,
        "c_zimmer": STR,
        "c_bett": STR,
    }
    rows = [
        (
            P1,
            F1,
            1,
            dt("2024-01-01 08:00"),
            dt("2024-01-01 10:00"),
            "ED-1",
            "INT",
            "ED 1",
            "Internal",
            "1A",
            "3",
            "A",
            "1",
            "1",
        ),
        (
            P1,
            F1,
            2,
            dt("2024-01-01 10:00"),
            dt("2024-01-03 10:00"),
            "ICU-MED1",
            "INT",
            "ICU Med 1",
            "Internal",
            "V",
            "1",
            "V",
            "2",
            "1",
        ),
        (
            P1,
            F1,
            3,
            dt("2024-01-03 12:00"),
            dt("2024-01-05 08:00"),
            "ICU-SURG1",
            "SURG",
            "ICU Surg 1",
            "Surgery",
            "V",
            "1",
            "V",
            "3",
            "1",
        ),
        (
            P1,
            F1,
            4,
            dt("2024-01-05 08:00"),
            dt("2024-01-12 08:00"),
            "WARD-MED1",
            "INT",
            "Ward Med 1",
            "Internal",
            "V",
            "1",
            "V",
            "4",
            "1",
        ),
        (
            P1,
            F1,
            5,
            dt("2024-01-12 08:00"),
            dt("2024-01-14 08:00"),
            "ICU-MED2",
            "INT",
            "ICU Med 2",
            "Internal",
            "V",
            "1",
            "V",
            "5",
            "1",
        ),
        (
            P2,
            F2,
            1,
            dt("2024-02-01 09:00"),
            dt("2024-02-03 06:00"),
            "ICU-CARD",
            "CARD",
            "ICU Card",
            "Cardiology",
            "A",
            "3",
            "A",
            "1",
            "1",
        ),
        (
            P2,
            F2,
            2,
            dt("2024-02-01 09:00"),
            dt("2024-02-03 06:00"),
            "ICU-CARD",
            "CARD",
            "ICU Card",
            "Cardiology",
            "BM",
            "1",
            "V",
            "1",
            "2",
        ),
        (
            P3,
            F3,
            1,
            dt("2024-03-01 09:00"),
            dt("2024-03-05 09:00"),
            "WARD-SURG1",
            "SURG",
            "Ward Surg 1",
            "Surgery",
            "A",
            "3",
            "A",
            "1",
            "1",
        ),
        (
            P4,
            F4,
            1,
            dt("2024-04-01 09:00"),
            dt("2024-04-05 09:00"),
            "PICU",
            "PED",
            "PICU",
            "Paediatrics",
            "A",
            "3",
            "A",
            "1",
            "1",
        ),
    ]
    return pl.DataFrame(rows, schema=schema, orient="row")


def _diagnose() -> pl.DataFrame:
    schema = {
        "c_patnr": STR,
        "c_falnr": STR,
        "c_diagnose_timestamp": TS,
        "c_diagnose_1": STR,
        "c_flag_fachabteilungshauptdiagnose": STR,
        "c_laufende_nummer": INT,
    }
    rows = [
        (P1, F1, dt("2024-01-01 11:00"), "I21.0", "X", 1),
        (P1, F1, dt("2024-01-03 13:00"), "J96.0", "X", 2),
        (P1, F1, dt("2024-01-03 13:00"), "E11.9", None, 3),
        (P2, F2, dt("2024-02-01 10:00"), "I46.9", "X", 1),
    ]
    return pl.DataFrame(rows, schema=schema, orient="row")


def _labor() -> pl.DataFrame:
    schema = {
        "c_patnr": STR,
        "c_falnr": STR,
        "c_dokument_timestamp": TS,
        "c_wert_timestamp": TS,
        "c_wert": STR,
        "c_wert_einheit": STR,
        "c_leistungnr": STR,
        "c_katalog_leistungtext": STR,
    }
    rows = [
        (
            P1,
            F1,
            dt("2024-01-02 06:00"),
            dt("2024-01-02 07:00"),
            "1.2",
            "mg/dl",
            "KREA",
            "Kreatinin",
        ),
        (
            P1,
            F1,
            dt("2024-01-02 18:00"),
            dt("2024-01-02 19:00"),
            "folgt",
            "mg/dl",
            "KREA",
            "Kreatinin",
        ),
        (
            P1,
            F1,
            dt("2024-01-02 06:00"),
            dt("2024-01-02 07:00"),
            "140",
            "mmol/l",
            "NA",
            "Natrium",
        ),
        (
            P2,
            F2,
            dt("2024-02-02 06:00"),
            dt("2024-02-02 07:00"),
            "2.5",
            "mg/dl",
            "KREA",
            "Kreatinin (enz.)",
        ),
        (
            P3,
            F3,
            dt("2024-03-02 06:00"),
            dt("2024-03-02 07:00"),
            "0.8",
            "mg/dl",
            "KREA",
            "Kreatinin",
        ),
    ]
    return pl.DataFrame(rows, schema=schema, orient="row")


def _hierarchy() -> pl.DataFrame:
    schema = {
        "c_patnr": STR,
        "c_falnr": STR,
        "c_var_id": INT,
        "c_var_name": STR,
        "c_id": INT,
        "c_parent_id": INT,
        "c_var_timestamp": TS,
        "c_value": STR,
    }
    ts = dt("2024-01-02 10:00")
    rows = [
        # SOFA score: two parents, the second without a total.
        (P1, F1, 500, "Score_SOFA", 1000, None, ts, None),
        (P1, F1, 501, "Score_SOFA_Date", 1001, 1000, ts, "2024-01-02 09:00:00"),
        (P1, F1, 502, "Score_SOFA_Total", 1002, 1000, ts, "7"),
        (P1, F1, 500, "Score_SOFA", 1010, None, ts, None),
        (P1, F1, 501, "Score_SOFA_Date", 1011, 1010, ts, "2024-01-02 21:00:00"),
        # A case outside the ICU cohort carries the same concept.
        (P3, F3, 500, "Score_SOFA", 1020, None, ts, None),
        (P3, F3, 501, "Score_SOFA_Date", 1021, 1020, ts, "2024-03-02 09:00:00"),
        (P3, F3, 502, "Score_SOFA_Total", 1022, 1020, ts, "3"),
        # Behandlung -> OP-Zeiten -> incision event (UTC strings in c_value).
        (P1, F1, 30, "Behandlung", 2000, None, ts, None),
        (P1, F1, 1263, "Behandlung_Beginn", 2002, 2000, ts, "2024-01-04 07:00:00"),
        (P1, F1, 1264, "Behandlung_Ende", 2003, 2000, ts, "2024-01-04 10:00:00"),
        (P1, F1, 101322, "OP_Zeiten", 2001, 2000, ts, None),
        (P1, F1, 101324, "OP_Zeiten_Ereignis", 2004, 2001, ts, "SCHNI"),
        (P1, F1, 101323, "OP_Zeiten_Zeit", 2005, 2001, ts, "2024-01-04 08:00:00"),
    ]
    return pl.DataFrame(rows, schema=schema, orient="row")


def _procedure() -> pl.DataFrame:
    schema = {
        "c_patnr": STR,
        "c_falnr": STR,
        "c_lfd_bewegung": INT,
        "c_prozedur_code": STR,
        "c_prozedur_begin": TS,
        "c_prozedur_ende": TS,
    }
    rows = [
        (P1, F1, 3, "5-361.0", dt("2024-01-04 09:30"), dt("2024-01-04 10:30")),
        (P1, F1, 3, "5-362.1", dt("2024-01-04 09:45"), dt("2024-01-04 10:30")),
        (P1, F1, 3, "8-931.0", dt("2024-01-04 09:00"), dt("2024-01-04 10:00")),
    ]
    return pl.DataFrame(rows, schema=schema, orient="row")


def write_dataset(root: Path) -> Path:
    """Write the synthetic dataset under `root` and return it.

    ``it_copra6_hierarchy_v2`` is split into two part files to exercise the
    multi-file layout of large tables.
    """
    db = root / "charite_mockdata"
    db.mkdir(parents=True)
    tables: dict[str, dict] = {}

    single = {
        "it_ishmed_patient": _patient(),
        "it_ishmed_fall": _fall(),
        "it_ishmed_bewegung": _bewegung(),
        "it_ishmed_diagnose": _diagnose(),
        "it_ishmed_labor": _labor(),
        "it_ishmed_procedure": _procedure(),
    }
    for name, df in single.items():
        rel = f"charite_mockdata/{name}.parquet"
        df.write_parquet(root / rel, compression="zstd")
        tables[name] = {"files": [rel], "rows": len(df)}

    hierarchy = _hierarchy()
    (db / "it_copra6_hierarchy_v2").mkdir()
    parts = []
    for i, part in enumerate((hierarchy.head(8), hierarchy.slice(8))):
        rel = f"charite_mockdata/it_copra6_hierarchy_v2/part-{i:03d}.parquet"
        part.write_parquet(root / rel, compression="zstd")
        parts.append(rel)
    tables["it_copra6_hierarchy_v2"] = {"files": parts, "rows": len(hierarchy)}

    manifest = {
        "dataset": "charite_mockdata",
        "version": "1",
        "generated_at": "2026-10-22T00:00:00Z",
        "reference_now": "2025-12-31T23:59:59",
        "tables": tables,
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


@pytest.fixture(scope="session")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The synthetic dataset, written once per test session."""
    return write_dataset(tmp_path_factory.mktemp("charite_mockdata"))


@pytest.fixture(autouse=True)
def _fresh_connections():
    """Drop cached connectors so every test starts from a clean DuckDB."""
    helpers.clear_connections()
    yield
    helpers.clear_connections()


@pytest.fixture
def no_concepts_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the Concepts API credential check a Cohort runs at construction.

    charite_demo routes to the demo endpoint; these tests build cohorts from
    hand-made variables only and must not touch the network.
    """
    from corr_vars.concepts.resolver import ConceptResolver

    monkeypatch.setattr(ConceptResolver, "authenticate", lambda self, sources: None)
