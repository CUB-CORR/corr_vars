"""The charite_demo source against a tiny synthetic dataset, fully offline.

Everything runs on the folder written by ``conftest.write_dataset``; the HTTP
tests serve that folder from a local, range-capable server.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import http.server
import json
import os
import threading
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import duckdb
import httpx
import polars as pl
import pytest

from conftest import F1, F2, F3, F4, dt
from corr_vars import Cohort
from corr_vars.sources.charite_demo import (
    Co6HierarchyDynamic,
    NativeDynamic,
    VariableLoader,
    cohort,
    config,
    download,
    helpers,
)
from corr_vars.sources.charite_demo.connector import DuckDBConnector
from corr_vars.sources.config_loader import _load_source_defaults
from corr_vars.utils.time import TimeWindow

from corr_vars.definitions.typing import ObsLevel


def source_config(path: Path | str, **overrides) -> config.SourceDict:
    """The resolved source config for `path`, as ``Cohort`` would build it."""
    return _load_source_defaults(
        {"path": str(path), **overrides}, config.DEFAULTS, config.MIGRATIONS
    )


@pytest.fixture
def demo_cohort(dataset_dir: Path, no_concepts_auth: None) -> Cohort:
    """An ICU-stay cohort over the synthetic dataset."""
    return Cohort(
        obs_level="icu_stay",
        sources={"charite_demo": {"path": str(dataset_dir)}},
        load_default_vars=False,
    )


# --------------------------------------------------------------------------
# Connector
# --------------------------------------------------------------------------
class TestConnector:
    def test_registers_a_view_per_manifest_table(self, dataset_dir: Path) -> None:
        conn = DuckDBConnector(str(dataset_dir))
        assert conn.tables == sorted(
            [
                "it_copra6_hierarchy_v2",
                "it_ishmed_bewegung",
                "it_ishmed_diagnose",
                "it_ishmed_fall",
                "it_ishmed_labor",
                "it_ishmed_patient",
                "it_ishmed_procedure",
            ]
        )
        views = conn.execute(
            "SELECT view_name FROM duckdb_views() WHERE schema_name = 'charite_mockdata'"
        )
        assert sorted(v for (v,) in views) == conn.tables

        # The multi-file table reads all of its parts.
        hierarchy = conn.conn.table(
            "it_copra6_hierarchy_v2", database="charite_mockdata"
        )
        assert hierarchy.count().execute() == 14
        conn.close()

    def test_columns_keep_their_types(self, dataset_dir: Path) -> None:
        conn = DuckDBConnector(str(dataset_dir))
        fall = conn.df_from_expr(
            conn.conn.table("it_ishmed_fall", database="charite_mockdata")
        )
        assert fall.schema["c_falnr"] == pl.Utf8
        assert fall.schema["c_aufnahme"] == pl.Datetime("us")
        assert fall.schema["c_age"] == pl.Int64
        conn.close()

    def test_expands_user_home(self, dataset_dir: Path, monkeypatch) -> None:
        monkeypatch.setenv("HOME", str(dataset_dir.parent))
        conn = DuckDBConnector(f"~/{dataset_dir.name}")
        assert conn.path == str(dataset_dir.resolve())
        conn.close()

    def test_missing_manifest_names_the_fix(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="download"):
            DuckDBConnector(str(tmp_path))

    def test_keep_and_datetime_cols(self, dataset_dir: Path) -> None:
        conn = DuckDBConnector(str(dataset_dir))
        df = conn.df_read_sql(
            "SELECT c_falnr, '2024-01-01 10:00:00' AS ts, 1 AS other "
            "FROM charite_mockdata.it_ishmed_fall",
            datetime_cols=["ts"],
            keep_cols=["ts", "c_falnr"],
            return_polars=True,
        )
        assert df.columns == ["ts", "c_falnr"]
        assert df.schema["ts"] == pl.Datetime("us")
        with pytest.warns(UserWarning, match="nope"):
            conn.df_read_sql(
                "SELECT 1 AS x", datetime_cols=["nope"], return_polars=True
            )
        conn.close()


class TestReinitConn:
    def test_one_connector_per_dataset(
        self, dataset_dir: Path, tmp_path: Path, make_dataset
    ) -> None:
        cfg = source_config(dataset_dir)
        same = source_config(f"{dataset_dir}/")
        assert helpers.reinit_conn(cfg) is helpers.reinit_conn(same)

        other = source_config(make_dataset(tmp_path / "other"))
        assert helpers.reinit_conn(other) is not helpers.reinit_conn(cfg)

    def test_databases(self, dataset_dir: Path) -> None:
        dbs = helpers.get_databases(source_config(dataset_dir))
        assert dbs == {"prepared": "charite_mockdata", "workbench": "temp.main"}
        # Ported snippets keep working under the production names.
        assert helpers.get_cub_hdp_database is helpers.get_databases


class TestTemporaryTable:
    def test_round_trip(self, dataset_dir: Path) -> None:
        stub = SimpleNamespace(
            _obs=pl.DataFrame({"case_id": [F1, F2, F2, None]}),
            sources={"charite_demo": source_config(dataset_dir)},
        )
        conn = helpers.reinit_conn(stub)
        dbs = helpers.get_databases(stub)
        with helpers.temporary_table(stub, id_column="case_id") as tmp:
            ids = conn.df_from_expr(conn.conn.table(tmp, database=dbs["workbench"]))
            assert sorted(ids["case_id"]) == [F1, F2]

            fall = conn.conn.table("it_ishmed_fall", database=dbs["prepared"])
            filtered = helpers.filter_by_tmp_table(
                conn,
                fall.select(case_id=fall.c_falnr),
                tmp_tbl=tmp,
                workbench_db=dbs["workbench"],
                id_column="case_id",
            )
            assert sorted(conn.df_from_expr(filtered)["case_id"]) == [F1, F2]

        remaining = conn.execute(
            f"SELECT count(*) FROM duckdb_tables() WHERE table_name = '{tmp}'"
        )
        assert remaining == [(0,)]


# --------------------------------------------------------------------------
# Raw SQL fragments
# --------------------------------------------------------------------------
class TestTryCast:
    @pytest.mark.parametrize(
        ("fragment", "expected"),
        [
            ("CAST(c_wert AS float)", "TRY_CAST(c_wert AS float)"),
            ("cast (c_wert as int) > 3", "TRY_CAST(c_wert as int) > 3"),
            ("TRY_CAST(c_wert AS float)", "TRY_CAST(c_wert AS float)"),
            ("my_cast(c_wert)", "my_cast(c_wert)"),
            ("c_value = 'CAST'", "c_value = 'CAST'"),
        ],
    )
    def test_rewrite(self, fragment: str, expected: str) -> None:
        assert helpers.try_cast_sql(fragment) == expected

    def test_bad_numeric_string_becomes_null(self, dataset_dir: Path) -> None:
        conn = helpers.reinit_conn(source_config(dataset_dir))
        labor = conn.conn.table("it_ishmed_labor", database="charite_mockdata")

        selected = helpers.apply_sql_select(
            labor,
            {"case_id": "c_falnr", "value": "CAST(c_wert AS float)"},
            where="c_katalog_leistungtext = 'Kreatinin' AND CAST(c_wert AS float) > 0 "
            "OR c_wert = 'folgt'",
        )
        values = conn.df_from_expr(selected).sort("value", nulls_last=True)["value"]
        assert values.to_list() == pytest.approx([0.8, 1.2, None], nan_ok=True)

        expr, _, _ = helpers.build_native_extractor_expr(
            labor,
            helpers.get_table_info("it_ishmed_labor"),
            id_column="case_id",
            value_dtype="float64",
        )
        out = conn.df_from_expr(expr)
        assert out.filter(pl.col("value").is_null()).height == 1
        assert out.schema["value"] == pl.Float64


# --------------------------------------------------------------------------
# Observation frames
# --------------------------------------------------------------------------
class TestLoadData:
    def test_icu_stays_merged(self, dataset_dir: Path) -> None:
        obs = cohort.load_data(ObsLevel.ICU_STAY, source_config(dataset_dir))
        obs = obs.sort("icu_stay_id")

        # ED/ward/OR movements, the companion (BM) and the paediatric ICU never
        # become stays; the two F1 ICU movements 2 h apart merge into one.
        assert obs["icu_stay_id"].to_list() == [f"{F1}_1", f"{F1}_2", f"{F2}_1"]
        assert obs["icu_id"].to_list() == [
            ["ICU-MED1", "ICU-SURG1"],
            ["ICU-MED2"],
            ["ICU-CARD"],
        ]
        assert obs["icu_admission"].to_list() == [
            dt("2024-01-01 10:00"),
            dt("2024-01-12 08:00"),
            dt("2024-02-01 09:00"),
        ]
        assert obs["icu_discharge"][0] == dt("2024-01-05 08:00")
        assert obs["icu_admission_diagnoses"][0].to_list() == ["I21.0", "J96.0"]
        assert obs["inhospital_death"].to_list() == [False, False, True]
        assert set(obs["case_id"]) == {F1, F2}

    def test_icu_stays_unmerged_with_extra_columns(self, dataset_dir: Path) -> None:
        cfg = source_config(
            dataset_dir,
            icu_stay={"merge_consecutive": False, "extra_columns": ["lfd_nr"]},
        )
        obs = cohort.load_data(ObsLevel.ICU_STAY, cfg).sort("icu_stay_id")
        assert obs.height == 4
        assert obs["icu_id"].to_list() == [
            ["ICU-MED1"],
            ["ICU-SURG1"],
            ["ICU-MED2"],
            ["ICU-CARD"],
        ]
        assert obs["lfd_nr"].to_list() == [["2"], ["3"], ["5"], ["1"]]
        assert obs.schema["icu_admission_diagnoses"] == pl.List(pl.Utf8)
        assert obs["icu_admission_diagnoses"][1].to_list() == ["J96.0"]

    def test_children_included_on_request(self, dataset_dir: Path) -> None:
        cfg = source_config(dataset_dir, filter={"include_adults_only": False})
        obs = cohort.load_data(ObsLevel.ICU_STAY, cfg)
        assert F4 in obs["case_id"].to_list()

    def test_hospital_stays(self, dataset_dir: Path) -> None:
        obs = cohort.load_data(ObsLevel.HOSPITAL_STAY, source_config(dataset_dir))
        # F3 never reached an ICU ward, F4 is a child.
        assert sorted(obs["case_id"]) == [F1, F2]
        f1 = obs.filter(pl.col("case_id").eq(F1)).row(0, named=True)
        assert f1["hospital_admission"] == dt("2024-01-01 08:00")
        assert f1["hospital_status"] == "Finalised"
        assert f1["age_on_admission"] == 63

    def test_extraction_window(self, dataset_dir: Path) -> None:
        cfg = source_config(
            dataset_dir,
            filter={"extraction_start_date": "2024-01-15", "extraction_end_date": None},
        )
        obs = cohort.load_data(ObsLevel.HOSPITAL_STAY, cfg)
        assert obs["case_id"].to_list() == [F2]

    def test_patients(self, dataset_dir: Path) -> None:
        obs = cohort.load_data(ObsLevel.PATIENT, source_config(dataset_dir))
        assert obs.height == 2
        assert obs.schema["censordate"] == pl.Datetime("us")
        # A living patient is censored at the dataset's frozen reference_now.
        living = obs.filter(pl.col("death_timestamp").is_null())
        assert living["censordate"].to_list() == [datetime(2025, 12, 31, 23, 59, 59)]

    def test_procedure_level_is_not_supported(self, dataset_dir: Path) -> None:
        with pytest.raises(NotImplementedError, match="procedure"):
            cohort.load_data(ObsLevel.PROCEDURE, source_config(dataset_dir))

    def test_cohort_uses_the_source(self, demo_cohort: Cohort) -> None:
        assert demo_cohort.obs.height == 3
        assert demo_cohort.obs["data_source"].unique().to_list() == ["charite_demo"]
        # Sex is passed through as coded in the dataset (M/F, like production).
        assert set(demo_cohort.obs["sex"].to_list()) <= {"M", "F"}


# --------------------------------------------------------------------------
# Variables
# --------------------------------------------------------------------------
class TestVariables:
    def test_native_dynamic_with_where_and_cast(self, demo_cohort: Cohort) -> None:
        demo_cohort.add_variable(
            NativeDynamic(
                var_name="creatinine",
                dynamic=True,
                table="it_ishmed_labor",
                where="c_katalog_leistungtext LIKE 'Kreatinin%'",
                columns={"value": "CAST(c_wert AS float)"},
                tmin="hospital_admission",
                tmax="hospital_discharge",
            )
        )
        data = demo_cohort.obsm["creatinine"].sort("icu_stay_id", "recordtime")
        # F1's two values reach both of its ICU stays (screened at case level);
        # 'folgt' is not a number and becomes null instead of failing the query.
        f1 = data.filter(pl.col("icu_stay_id").eq(f"{F1}_1"))
        assert f1["value"].to_list() == pytest.approx([1.2, None], nan_ok=True)
        assert data.filter(pl.col("icu_stay_id").eq(f"{F2}_1"))["value"].to_list() == [
            pytest.approx(2.5)
        ]
        assert F3 not in data["icu_stay_id"].str.slice(0, 10).to_list()

    def test_legacy_pandas_py_function(self, demo_cohort: Cohort) -> None:
        def to_float(var, cohort):
            df = var.data
            df["value"] = df["value"].astype(float) * 2
            return df

        demo_cohort.add_variable(
            NativeDynamic(
                var_name="sodium_doubled",
                dynamic=True,
                table="it_ishmed_labor",
                where="c_katalog_leistungtext = 'Natrium'",
                py=to_float,
                tmin="hospital_admission",
                tmax="hospital_discharge",
            )
        )
        values = demo_cohort.obsm["sodium_doubled"]["value"].unique().to_list()
        assert values == [280.0]

    def test_co6_hierarchy(self, demo_cohort: Cohort) -> None:
        demo_cohort.add_variable(
            Co6HierarchyDynamic(
                var_name="sofa",
                parent_name="Score_SOFA",
                suffixes={"_Date": "recordtime", "_Total": "value"},
                not_strict=["_Total"],
                value_dtype="Float64",
                tmin="hospital_admission",
                tmax="hospital_discharge",
            )
        )
        data = demo_cohort.obsm["sofa"].filter(pl.col("icu_stay_id").eq(f"{F1}_1"))
        data = data.sort("recordtime")
        # c_value timestamps are UTC; Berlin is UTC+1 in January.
        assert data["recordtime"].to_list() == [
            datetime(2024, 1, 2, 10, 0),
            datetime(2024, 1, 2, 22, 0),
        ]
        assert data["value"].to_list() == [7.0, None]

    def test_co6_hierarchy_strict_join_drops_incomplete_parents(
        self, demo_cohort: Cohort
    ) -> None:
        demo_cohort.add_variable(
            Co6HierarchyDynamic(
                var_name="sofa_strict",
                parent_name="Score_SOFA",
                suffixes={"_Date": "recordtime", "_Total": "value"},
                tmin="hospital_admission",
                tmax="hospital_discharge",
            )
        )
        data = demo_cohort.obsm["sofa_strict"]
        assert data.filter(pl.col("icu_stay_id").eq(f"{F1}_1"))["value"].to_list() == [
            "7"
        ]

    def test_build_procedures(self, dataset_dir: Path) -> None:
        conn = helpers.reinit_conn(source_config(dataset_dir))
        proc = helpers.build_procedures(conn, "charite_mockdata").collect()
        row = proc.row(0, named=True)
        assert row["case_id"] == F1
        assert row["ops_code"] == ["5-361.0", "5-362.1"] or row["ops_code"] == [
            "5-362.1",
            "5-361.0",
        ]
        assert row["or_time_begin"] == dt("2024-01-04 08:00")
        assert row["or_time_incision"] == dt("2024-01-04 09:00")
        assert row["bw_pflege_oe"] == "ICU-SURG1"

    def test_variable_loader(self) -> None:
        window = TimeWindow("hospital_admission", "hospital_discharge")
        var = VariableLoader(
            var_name="sofa",
            time_window=window,
            type="co6_hierarchy",
            parent_name="Score_SOFA",
            suffixes={"_Date": "recordtime", "_Total": "value"},
        )
        assert isinstance(var, Co6HierarchyDynamic)
        assert var.dynamic
        assert var.guessed_source == "charite_demo"

        with pytest.raises(NotImplementedError, match="medication"):
            VariableLoader(var_name="propofol", time_window=window, type="medication")


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Static file handler answering ``Range: bytes=a-b`` with 206."""

    def log_message(self, format: str, *args: object) -> None:
        pass

    def send_head(self):  # type: ignore[override]
        path = Path(self.translate_path(self.path))
        rng = self.headers.get("Range")
        if not path.is_file() or not rng or not rng.startswith("bytes="):
            return super().send_head()
        size = path.stat().st_size
        start_s, _, end_s = rng.removeprefix("bytes=").partition("-")
        start = int(start_s) if start_s else size - int(end_s)
        end = int(end_s) if start_s and end_s else size - 1
        end = min(end, size - 1)
        fh = path.open("rb")
        fh.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self._remaining = end - start + 1
        return fh

    def copyfile(self, source, outputfile) -> None:  # type: ignore[override]
        remaining = getattr(self, "_remaining", None)
        if remaining is None:
            return super().copyfile(source, outputfile)
        outputfile.write(source.read(remaining))
        self._remaining = None


@contextlib.contextmanager
def _serve(directory: Path):
    """Serve `directory` over HTTP on localhost; yields the base URL."""
    handler = functools.partial(_RangeHandler, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def served_dataset(dataset_dir: Path):
    """Serve the dataset folder over HTTP on localhost; yields the base URL."""
    with _serve(dataset_dir) as url:
        yield url


def _hashed_release(src: Path, dest: Path) -> Path:
    """Copy `src` as a release with content-hashed names and a sha256 index.

    Mirrors the published layout: ``<stem>-<sha256[:12]>.parquet``, the manifest
    listing the hashed paths and ``files: {path: {sha256, bytes}}``, plus a
    ``ground_truth`` sidecar that is not a table.
    """
    manifest = json.loads((src / "manifest.json").read_text())
    index: dict[str, dict] = {}
    for info in manifest["tables"].values():
        renamed = []
        for rel in info["files"]:
            data = (src / rel).read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            new = f"{rel.removesuffix('.parquet')}-{digest[:12]}.parquet"
            (dest / new).parent.mkdir(parents=True, exist_ok=True)
            (dest / new).write_bytes(data)
            index[new] = {"sha256": digest, "bytes": len(data)}
            renamed.append(new)
        info["files"] = renamed
    (dest / "ground_truth").mkdir(parents=True, exist_ok=True)
    (dest / "ground_truth" / "stays-000000000000.parquet").write_bytes(b"x")
    manifest["ground_truth"] = {
        "stays": {"files": ["ground_truth/stays-000000000000.parquet"]}
    }
    manifest["files"] = index
    (dest / "manifest.json").write_text(json.dumps(manifest))
    return dest


def _httpfs_available() -> bool:
    try:
        duckdb.connect().execute("LOAD httpfs")
    except Exception:
        return False
    return True


class TestHttp:
    def test_download_then_load(self, served_dataset: str, tmp_path: Path) -> None:
        target = download(tmp_path / "copy", base_url=served_dataset)
        assert Path(target, "manifest.json").is_file()

        # A second run skips complete files and resumes a truncated one.
        part = Path(target, "charite_mockdata", "it_ishmed_fall.parquet")
        full = part.read_bytes()
        part.unlink()
        part.with_name(part.name + ".part").write_bytes(full[:10])
        download(target, base_url=served_dataset)
        assert part.read_bytes() == full

        obs = cohort.load_data(ObsLevel.HOSPITAL_STAY, source_config(target))
        assert sorted(obs["case_id"]) == [F1, F2]

    def test_download_verifies_sha256_and_removes_stale_files(
        self, dataset_dir: Path, tmp_path: Path
    ) -> None:
        release = _hashed_release(dataset_dir, tmp_path / "release")
        manifest = json.loads((release / "manifest.json").read_text())
        target = tmp_path / "copy"
        db = target / "charite_mockdata"
        # leftovers of an older release + files that must survive
        db.mkdir(parents=True)
        (db / "it_ishmed_fall-0123456789ab.parquet").write_bytes(b"old")
        (db / "it_copra6_hierarchy_v2").mkdir()
        (db / "it_copra6_hierarchy_v2" / "part-000-aaaaaaaaaaaa.parquet").write_bytes(
            b"old"
        )
        (db / "gone-bbbbbbbbbbbb.parquet.part").write_bytes(b"old")
        (db / "notes.txt").write_text("mine")
        (target / "elsewhere.parquet").write_bytes(b"mine")

        with _serve(release) as url:
            download(target, base_url=url)

            listed = {f for t in manifest["tables"].values() for f in t["files"]}
            on_disk = {str(p.relative_to(target)) for p in db.rglob("*.parquet*")}
            assert on_disk == listed
            assert (db / "notes.txt").read_text() == "mine"
            assert (target / "elsewhere.parquet").read_bytes() == b"mine"
            assert not (target / "ground_truth").exists()
            local = json.loads((target / "manifest.json").read_text())
            assert "ground_truth" not in local
            assert set(local["files"]) == listed
            obs = cohort.load_data(ObsLevel.HOSPITAL_STAY, source_config(target))
            assert sorted(obs["case_id"]) == [F1, F2]

            # a corrupted copy of the same size fails its checksum and is fetched again
            rel = manifest["tables"]["it_ishmed_fall"]["files"][0]
            good = (target / rel).read_bytes()
            (target / rel).write_bytes(bytes(len(good)))
            download(target, base_url=url)
            assert (target / rel).read_bytes() == good

            # a server file that does not match the manifest's sha256 is rejected
            (release / rel).write_bytes(bytes(len(good)))
            (target / rel).unlink()
            with pytest.raises(httpx.HTTPError, match="Checksum mismatch"):
                download(target, base_url=url)
            assert not (target / rel).exists()
            assert not (target / (rel + ".part")).exists()

    def test_download_subset(self, served_dataset: str, tmp_path: Path) -> None:
        target = download(
            tmp_path / "subset", base_url=served_dataset, tables=["it_ishmed_fall"]
        )
        assert DuckDBConnector(target).tables == ["it_ishmed_fall"]
        with pytest.raises(ValueError, match="Unknown table"):
            download(tmp_path / "x", base_url=served_dataset, tables=["nope"])

    @pytest.mark.skipif(
        not _httpfs_available() or os.environ.get("CI") == "true",
        reason="DuckDB's httpfs extension is not installed (installing needs network)",
    )
    def test_read_over_http(self, served_dataset: str) -> None:
        obs = cohort.load_data(ObsLevel.ICU_STAY, source_config(served_dataset))
        assert obs.height == 3
