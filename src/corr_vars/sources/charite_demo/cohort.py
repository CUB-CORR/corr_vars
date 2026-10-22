"""Observation-level data loading for the ``charite_demo`` source.

A port of the production ``cub_hdp`` loaders to the DuckDB-backed mock dataset:
the queries are the same ibis expressions over the same tables and columns.
``patient``, ``hospital_stay`` and ``icu_stay`` are supported; ``procedure`` is
not (see :func:`load_data`).
"""

from __future__ import annotations

import ibis
import polars as pl

from corr_vars import __, logger
from corr_vars.sources.charite_demo import helpers
from corr_vars.sources.charite_demo.mapping import GLOBAL_VARS, TABLES
from corr_vars.utils.base import as_expr, row_length
from corr_vars.utils.time import (
    TimeWindow,
    WindowRelation,
    filter_with_time_window,
)

from corr_vars.definitions.typing import ObsLevel, PolarsFrame
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from datetime import datetime

    from corr_vars.sources.charite_demo.config import ExtraIcuColumns, SourceDict
    from corr_vars.sources.charite_demo.connector import DuckDBConnector

    from collections.abc import Iterable, Sequence


def load_data(
    obs_level: ObsLevel,
    source_kwargs: SourceDict,
) -> pl.DataFrame:
    """Build the cohort's observation frame from the charite_demo dataset.

    Args:
        obs_level (ObsLevel): Level of observation the frame is keyed at.
        source_kwargs (SourceDict): This source's resolved configuration.

    Returns:
        pl.DataFrame: One row per observation.

    Raises:
        NotImplementedError: For ``ObsLevel.PROCEDURE``.
        ValueError: For any other unsupported observation level.
    """
    ####################
    #  Parse arguments #
    ####################
    # Database arguments
    database = source_kwargs["database"]

    # Filter arguments
    filter_kwargs = source_kwargs["filter"]
    extraction_start_date = filter_kwargs["extraction_start_date"]
    extraction_end_date = filter_kwargs["extraction_end_date"]
    additional_filters = helpers.parse_filters(filter_kwargs["additional_filters"])
    include_adults_only = filter_kwargs["include_adults_only"]
    exclude_brain_death = filter_kwargs["exclude_brain_death"]

    # Parse extraction filter dates
    if isinstance(extraction_start_date, str):
        extraction_start_date = helpers.parse_date(extraction_start_date)
    if isinstance(extraction_end_date, str):
        extraction_end_date = helpers.parse_date(extraction_end_date)

    # ICU specific arguments
    icu_kwargs = source_kwargs["icu_stay"]
    merge_consecutive = icu_kwargs["merge_consecutive"]
    icu_extra_columns = sorted(set(icu_kwargs["extra_columns"]))

    ####################
    #   Load ICU IDs   #
    ####################
    icu_ids = _load_icu_ids(include_adults_only=include_adults_only)

    ####################
    #    Load Data     #
    ####################
    conn = helpers.reinit_conn(source_kwargs)
    match obs_level:
        case ObsLevel.PATIENT:
            lf_obs = _load_ishmed_patient(
                conn,
                database,
                icu_ids=icu_ids,
                require_movement=True,
                include_adults_only=include_adults_only,
                exclude_brain_death=exclude_brain_death,
                start_date=extraction_start_date,
                end_date=extraction_end_date,
                where=additional_filters,
            )
        case ObsLevel.HOSPITAL_STAY:
            lf_obs = _load_ishmed_fall(
                conn,
                database,
                icu_ids=icu_ids,
                require_movement=True,
                include_adults_only=include_adults_only,
                exclude_brain_death=exclude_brain_death,
                start_date=extraction_start_date,
                end_date=extraction_end_date,
                where=additional_filters,
            )
        case ObsLevel.ICU_STAY:
            lf_bew = _load_ishmed_bewegung(
                conn,
                database,
                icu_ids=icu_ids,
                exclude_brain_death=exclude_brain_death,
                start_date=extraction_start_date,
                end_date=extraction_end_date,
                extra_extract_columns=icu_extra_columns,
            )
            lf_bew = _load_ishmed_diagnosis(conn, database, lf_bew)
            lf_bew = _merge_bew(lf_bew, merge_consecutive, icu_extra_columns)
            lf_fall = _load_ishmed_fall(
                conn,
                database,
                # ward restriction and movement-art filter handled by
                # _load_ishmed_bewegung for icu_stay
                icu_ids=[],
                require_movement=False,
                include_adults_only=include_adults_only,
                exclude_brain_death=exclude_brain_death,
                start_date=extraction_start_date,
                end_date=extraction_end_date,
                where=additional_filters,
            )
            lf_obs = _load_icu_stays(
                lf_bew, lf_fall, extra_keep_columns=icu_extra_columns
            )
        case ObsLevel.PROCEDURE:
            raise NotImplementedError(
                "charite_demo does not support obs_level='procedure'. Use "
                "'icu_stay', 'hospital_stay' or 'patient'; surgeries are still "
                "available as a variable (e.g. 'surgery')."
            )
        case _:
            raise ValueError(f"Unsupported observation level: {obs_level}")

    return lf_obs.collect()


def _load_icu_ids(*, include_adults_only: bool) -> list[str]:
    """Load the list of ICU ward ids (c_pflege_oe_id) based on the given filters."""
    ward_ids = GLOBAL_VARS["hospital_data"]["ward_ids"]
    icu_ids = list(ward_ids["adult_icu_ids"])
    if not include_adults_only:
        icu_ids += ward_ids["ped_neo_icu_ids"]
    return icu_ids


def _excluded_movement_arts(exclude_brain_death: bool) -> list[str]:
    """c_bewegungsart codes excluded when identifying stays.

    Companion-person admissions (``Begleitpersonen``) are always excluded; brain-death /
    organ-explantation re-admissions are excluded only when requested.
    """
    mv_types = GLOBAL_VARS["hospital_data"]["mv_types"]
    excluded = list(mv_types["Companion"])
    if exclude_brain_death:
        excluded += mv_types["Brain_death"]
    return excluded


def _filter_within_hospital_stay(
    df: PolarsFrame, start_col: str, end_col: str
) -> PolarsFrame:
    """Keep rows where the given interval falls within hospital admission and discharge."""
    record_window = TimeWindow(start_col, end_col)
    hospital_window = TimeWindow.from_obs_level(ObsLevel.HOSPITAL_STAY)
    return df.pipe(
        filter_with_time_window,
        time_window=record_window,
        reference_window=hospital_window,
        relation=WindowRelation.CONTAINED,
        keep_null=True,
    )  # type: ignore


def _sequential_id(on: str | pl.Expr, by: str | pl.Expr | None = None) -> pl.Expr:
    """Assign a 1-based sequential ``expr_N`` identifier per `expr` group.

    ``cum_count`` is already 1-based, so the first row of a group gets ``_1``.
    """
    expr = as_expr(on)
    return (
        expr.cast(pl.Utf8)
        .add(pl.lit("_"))
        .add(expr.cum_count().over(partition_by=expr, order_by=by).cast(pl.Utf8))
    )


def _load_ishmed_patient(
    conn: DuckDBConnector,
    database: str,
    *,
    icu_ids: Iterable[str] = (),
    require_movement: bool = True,
    include_adults_only: bool = True,
    exclude_brain_death: bool = True,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    where: str | None = None,
) -> pl.LazyFrame:
    """Build a lazy query for the it_ishmed_patient table.

    When *require_movement* is ``True`` the case must have a qualifying ICU
    movement (see :func:`helpers.filtered_movements`) — this defines the
    patient cohort.  Set it to ``False`` when the result is only used as a
    per-case metadata lookup, so cases without such a movement keep their
    patient data instead of being dropped.
    """
    now = helpers.now_expr(end_date, reference_now=conn.reference_now)
    p = conn.conn.table("it_ishmed_patient", database=database)

    # Filter for valid IDs (known HDP issue)
    expr = p.filter(helpers.is_present(p.c_patnr))

    # Bound the patient timeline (birth/death) to the extraction window; rows with
    # a missing bound (e.g. living patients) are kept for the user to handle.
    expr = expr.filter(
        helpers.within_date_window(
            expr.c_birthdate,
            expr.c_datetimeofdeath,
            start_date=start_date,
            end_date=end_date,
        )
    )

    # Age filter
    if include_adults_only:
        # it_ishmed_patient has no age column, so derive it: adult = at least 18
        # at censordate (death, or now if living).  Null birthdate is kept (age
        # unknown) for the user to handle.
        censordate = expr.c_datetimeofdeath.coalesce(now)
        expr = expr.filter(
            expr.c_birthdate.isnull()
            | (expr.c_birthdate <= censordate - ibis.interval(years=18))
        )

    # EXISTS over qualifying movements (always applied), correlated on the patient
    # id.  The movement-art filter always applies; the ICU-ward restriction only
    # kicks in when icu_ids is non-empty (see helpers.filtered_movements).
    if require_movement:
        exclude_movement_ids = _excluded_movement_arts(exclude_brain_death)
        movements = helpers.filtered_movements(
            conn.conn.table("it_ishmed_bewegung", database=database),
            icu_ids=icu_ids,
            exclude_movement_ids=exclude_movement_ids,
            start_date=start_date,
            end_date=end_date,
        )
        expr = expr.filter(expr.c_patnr.isin(movements.c_patnr))

    if where:
        expr = helpers.apply_sql_where(expr, where)

    expr = expr.select(
        expr.c_patnr.name("patient_id"),
        expr.c_gender.name("sex"),
        expr.c_birthdate.name("birthdate"),
        expr.c_nationality.name("nationality"),
        expr.c_countryofbirth.name("country_of_birth"),
        expr.c_country.name("country"),
        expr.c_postcode.name("postcode"),
        expr.c_datetimeofdeath.name("death_timestamp"),
        expr.c_datetimeofdeath.coalesce(now).name("censordate"),
    ).distinct()

    lf = conn.lf_from_expr(
        expr,
        datetime_cols=["birthdate", "death_timestamp"],
    )

    logger.info(
        __(
            "SUCCESS: Extracted ISHMED_PATIENT data for {n_obs} patients",
            n_obs=row_length(lf),
        )
    )
    return lf


def _load_ishmed_fall(
    conn: DuckDBConnector,
    database: str,
    *,
    include_adults_only: bool = True,
    exclude_brain_death: bool = True,
    icu_ids: Iterable[str] = (),
    require_movement: bool = True,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    where: str | None = None,
) -> pl.LazyFrame:
    """Build a lazy query for the it_ishmed_fall table.

    When *require_movement* is ``True`` the case must have a qualifying ICU
    movement (see :func:`helpers.filtered_movements`) — this defines the
    hospital-stay cohort.  Set it to ``False`` when the result is only used as a
    per-case metadata lookup (e.g. the procedure obs level joins fall onto
    procedures), so cases without such a movement keep their hospital data
    instead of being dropped.
    """
    now = helpers.now_expr(end_date, reference_now=conn.reference_now)
    f = conn.conn.table("it_ishmed_fall", database=database)

    # additional_filters reference raw fall columns; apply while they are valid
    # (before any aliasing).  As they only touch fall columns, applying them
    # before the LEFT JOIN is equivalent to the outer WHERE.
    if where:
        f = helpers.apply_sql_where(f, where)

    p = conn.conn.table("it_ishmed_patient", database=database)

    # Determine the join key shared by the fall and patient tables (replaces a
    # runtime DESCRIBE round-trip).
    # TODO: Is c_falnr/c_patnr unique on it_ishmed_patient?
    common_id = "c_falnr" if "c_falnr" in p.columns else "c_patnr"
    joined = f.join(p, common_id, how="left")

    # Filter for valid IDs (known HDP issue)
    expr = joined.filter(helpers.is_present(f.c_falnr))

    # Age filter
    if include_adults_only:
        expr = expr.filter(f.c_age >= ibis.literal(18))

    # Bound the hospital timeline (admission/discharge) to the extraction window; rows with
    # a missing bound (e.g. ongoing hospital cases) are kept for the user to handle.
    expr = expr.filter(
        helpers.within_date_window(
            f.c_aufnahme,
            f.c_entlassung,
            start_date=start_date,
            end_date=end_date,
        )
    )

    # EXISTS over qualifying movements, correlated on the case id.  The
    # movement-art filter always applies; the ICU-ward restriction only kicks in
    # when icu_ids is non-empty (see helpers.filtered_movements).  Skipped when
    # fall is only a metadata lookup (require_movement=False).
    if require_movement:
        exclude_movement_ids = _excluded_movement_arts(exclude_brain_death)
        movements = helpers.filtered_movements(
            conn.conn.table("it_ishmed_bewegung", database=database),
            icu_ids=icu_ids,
            exclude_movement_ids=exclude_movement_ids,
            start_date=start_date,
            end_date=end_date,
        )
        expr = expr.filter(f.c_falnr.isin(movements.c_falnr))

    expr = expr.select(
        f.c_patnr.name("patient_id"),
        f.c_falnr.name("case_id"),
        ibis.cases((f.c_aufnahme <= now, f.c_aufnahme), else_=None).name(
            "hospital_admission"
        ),
        ibis.cases((f.c_entlassung <= now, f.c_entlassung), else_=None).name(
            "hospital_discharge"
        ),
        helpers.stay_status_expr(f.c_aufnahme, f.c_entlassung, now).name(
            "hospital_status"
        ),
        # Facility category the patient was admitted from / discharged to
        # (hospice, rehabilitation, nursing home, hospital, …).
        f.c_admission_hospital.name("admission_origin"),
        f.c_discharge_hospital.name("discharge_destination"),
        f.c_age.name("age_on_admission"),
        p.c_gender.name("sex"),
        p.c_birthdate.name("birthdate"),
        p.c_nationality.name("nationality"),
        p.c_countryofbirth.name("country_of_birth"),
        p.c_country.name("country"),
        p.c_postcode.name("postcode"),
        f.c_dod.name("death_timestamp"),
        f.c_entlassungsart.isin(
            GLOBAL_VARS["hospital_data"]["discharge_types"]["Discharge_death"]
        ).name("inhospital_death"),
    )

    lf = conn.lf_from_expr(
        expr,
        datetime_cols=[
            "hospital_admission",
            "hospital_discharge",
            "birthdate",
            "death_timestamp",
        ],
    )

    logger.info(
        __(
            "SUCCESS: Extracted ISHMED_FALL data for {n_obs} cases",
            n_obs=row_length(lf),
        )
    )
    return lf


def _load_ishmed_bewegung(
    conn: DuckDBConnector,
    database: str,
    *,
    icu_ids: list[str],
    exclude_brain_death: bool,
    start_date: datetime | None,
    end_date: datetime | None,
    extra_extract_columns: list[ExtraIcuColumns] | None = None,
) -> pl.LazyFrame:
    """Build a lazy query for raw ICU movement records from the it_ishmed_bewegung table."""
    now = helpers.now_expr(end_date, reference_now=conn.reference_now)

    exclude_movement_ids = _excluded_movement_arts(exclude_brain_death)
    bew = helpers.filtered_movements(
        conn.conn.table("it_ishmed_bewegung", database=database),
        icu_ids=icu_ids,
        exclude_movement_ids=exclude_movement_ids,
        start_date=start_date,
        end_date=end_date,
    )

    # Filter for valid IDs (known HDP issue)
    bew = bew.filter(helpers.is_present(bew.c_falnr))

    extra_selects = [
        bew[TABLES["it_ishmed_bewegung"][f"attributes_{col}"]].name(col)
        for col in (extra_extract_columns or [])
    ]

    expr = bew.select(
        bew.c_falnr.name("case_id"),
        ibis.cases((bew.c_begin <= now, bew.c_begin), else_=None).name("icu_admission"),
        ibis.cases((bew.c_ende <= now, bew.c_ende), else_=None).name("icu_discharge"),
        helpers.stay_status_expr(bew.c_begin, bew.c_ende, now).name("icu_status"),
        bew.c_bewegungsart.name("bw_patient_origin"),
        bew.c_fachliche_oe_id.name("bw_fach_oe"),
        bew.c_pflege_oe_id.name("icu_id"),
        *extra_selects,
    )

    return conn.lf_from_expr(
        expr,
        datetime_cols=["icu_admission", "icu_discharge"],
    )


def _merge_bew(
    lf_bew: pl.LazyFrame,
    merge_consecutive: Literal["any", "same_icu_only"] | bool,
    extra_columns: list[ExtraIcuColumns],
) -> pl.LazyFrame:
    """Collect the movement LazyFrame, optionally merging consecutive ICU stays."""
    if not merge_consecutive:
        logger.info("SUCCESS: Extracted ICU stays")
        # NOTE: Workaround for variables / code relying on type list for these variables
        return lf_bew.with_columns(
            pl.concat_list(pl.col(col).cast(pl.Utf8)).alias(col)
            for col in (
                "bw_patient_origin",
                "bw_fach_oe",
                "icu_id",
                "icu_admission_diagnoses",
                *extra_columns,
            )
        )

    logger.info("Merging consecutive ICU stays...")
    lf_merged = helpers.merge_consecutive_stays(
        lf_bew,
        strategy="any" if merge_consecutive is True else merge_consecutive,
    )
    original_count = row_length(lf_bew)
    merged_count = row_length(lf_merged)
    logger.info(__("Original number of stays: {n}", n=original_count))
    logger.info(__("Number of stays after merging: {n}", n=merged_count))
    logger.info(
        __(
            "Percentage of stays after merging: {pct:.2%}",
            pct=merged_count / original_count,
        )
    )
    logger.info("SUCCESS: Extracted ICU stays")
    return lf_merged


def _load_ishmed_diagnosis(
    conn: DuckDBConnector, database: str, lf_bew: pl.LazyFrame
) -> pl.LazyFrame:
    diag = conn.conn.table("it_ishmed_diagnose", database=database)
    diag_expr = diag.filter(diag.c_flag_fachabteilungshauptdiagnose.like("X")).select(
        diag.c_falnr.name("case_id"),
        diag.c_diagnose_1.name("icu_admission_diagnoses"),
        diag.c_laufende_nummer,
    )

    diagnoses = conn.lf_from_expr(diag_expr).join(lf_bew, on="case_id", how="semi")

    # Add an identifyer to the diagnoses. We sort by c_laufende_nummer within a case.
    # 1) Diagnosis: stable order + index (per case)
    diagnoses = diagnoses.with_columns(
        pl.arg_sort_by("c_laufende_nummer").over("case_id").alias("diag_idx")
    ).drop("c_laufende_nummer")

    # 2) Identify fachabteilung-change blocks on the movement stream
    # fach_idx increments whenever bw_fach_oe (c_fachliche_oe_id) changes within a case.
    movements = (
        lf_bew.sort("case_id", "icu_admission")
        .with_columns(
            pl.col("bw_fach_oe").shift(1).over("case_id").alias("prev_bw_fach_oe")
        )
        .with_columns(
            pl.col("bw_fach_oe")
            .ne(pl.col("prev_bw_fach_oe"))
            .fill_null(False)
            .alias("new_bw_fach_oe")
        )
        .with_columns(pl.cum_sum("new_bw_fach_oe").over("case_id").alias("fach_idx"))
        .drop("prev_bw_fach_oe", "new_bw_fach_oe")
    )

    # 3) Build a per-case mapping: fach_idx (order by first begin) -> diag_idx (0..n-1)
    # This does NOT drop any movement rows; it just computes a lookup and merges it back.
    # Assign sequential diag_idx per case to the ordered fachabteilung blocks.
    block_order = (
        movements.group_by("case_id", "fach_idx")
        .agg(pl.min("icu_admission"))
        .sort("case_id", "icu_admission")
        .with_columns(pl.int_range(0, pl.len()).over("case_id").alias("diag_idx"))
    )

    # 4) Annotate EVERY movement row with its diag_idx (no structure change)
    movement_annotated = movements.join(
        block_order.select("case_id", "fach_idx", "diag_idx"),
        how="left",
        on=["case_id", "fach_idx"],
        validate="m:1",
    )

    # 5) Bring diagnosis columns onto EACH movement row
    movements_with_diagnoses = movement_annotated.join(
        diagnoses,  # contains diag_idx per case
        how="left",
        on=["case_id", "diag_idx"],
        validate="m:1",  # each movement row maps to at most one diagnosis step
    )

    # mov_with_diag now has:
    # - exact original movement granularity (every row preserved)
    # - fach_idx (block id), diag_idx (aligned to diagnosis sequence)
    # - all diagnosis columns merged onto each movement row belonging to that fachabteilung
    return movements_with_diagnoses.drop("fach_idx", "diag_idx")


def _load_icu_stays(
    lw_bew: pl.LazyFrame, lf_fall: pl.LazyFrame, *, extra_keep_columns: Sequence[str]
) -> pl.LazyFrame:
    """Merge ICU stays with hospital-stay data to produce one row per ICU stay."""
    lf = lw_bew.join(lf_fall, on="case_id", how="inner")
    lf = _filter_within_hospital_stay(lf, "icu_admission", "icu_discharge")
    lf = lf.select(
        "patient_id",
        "case_id",
        _sequential_id(on="case_id", by="icu_admission").alias("icu_stay_id"),
        "age_on_admission",
        "sex",
        "birthdate",
        "nationality",
        "country_of_birth",
        "country",
        "postcode",
        "death_timestamp",
        "inhospital_death",
        "hospital_admission",
        "hospital_discharge",
        "hospital_status",
        "admission_origin",
        "discharge_destination",
        "icu_admission",
        "icu_discharge",
        "icu_status",
        "icu_id",
        "icu_admission_diagnoses",
        "bw_fach_oe",
        "bw_patient_origin",
        *extra_keep_columns,
    )
    logger.info(__("Number of ICU stays: {n}", n=row_length(lf)))
    logger.info("SUCCESS: Merged ICU stays with static data")
    return lf
