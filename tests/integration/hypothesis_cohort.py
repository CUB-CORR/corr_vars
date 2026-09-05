import re
from datetime import datetime, timedelta

import hypothesis.strategies as st
import polars as pl
from polars.testing.parametric import column, dataframes

from corr_vars.definitions import ObsLevel
from corr_vars.utils.frames import time_difference

from collections.abc import Callable

# Artificially capped to 300 unique combinations
ID_REGEX = re.compile(r"py-0{8}[0-2][0-9]-[0-9]")


@st.composite
def obs_dataframe(
    draw: Callable[[st.SearchStrategy[pl.DataFrame]], pl.DataFrame],
    size: int | tuple[int, int] | None = None,
    obs_level: ObsLevel = ObsLevel.HOSPITAL_STAY,
    data_source: str = "dummy",
) -> pl.DataFrame:
    if size is None:
        _min_size = _max_size = 100
    elif isinstance(size, int):
        _min_size = _max_size = size
    else:
        _min_size, _max_size = size

    # Enforce Regex cap
    _min_size = min(max(_min_size, 1), 300)
    _max_size = min(max(_min_size, 1), 100)

    # Enforce _max_size >= _min_size
    _max_size = min(_min_size, _max_size)

    now = datetime.now()
    df: pl.DataFrame = draw(
        dataframes(
            cols=[
                column(
                    "data_source",
                    strategy=st.just(data_source),
                ),
                column(
                    "id",
                    strategy=st.from_regex(ID_REGEX, fullmatch=True),
                    unique=True,
                ),
                column(
                    "hospital_admission",
                    strategy=st.datetimes(
                        min_value=datetime(2005, 7, 18),  # Actual min in real cohort
                        max_value=datetime(now.year, now.month, now.day),
                    ),
                ),
                column(
                    # TODO: Improve this
                    "hospital_status",
                    strategy=st.just("Finalised"),
                ),
                column(
                    "hospital_lenth_of_stay",
                    strategy=st.timedeltas(
                        min_value=timedelta(days=18),
                        max_value=timedelta(days=180),
                    ),
                ),
                column(
                    # TODO: Improve this
                    "icu_stays",
                    strategy=st.integers(min_value=1, max_value=5),
                ),
                column(
                    # TODO: Improve this
                    "procedures",
                    strategy=st.integers(min_value=1, max_value=3),
                ),
                column(
                    "age_on_admission",
                    strategy=st.integers(min_value=18, max_value=120),
                ),
                column("sex", strategy=st.sampled_from(["M", "F"])),
                column(
                    "birthday_offset", strategy=st.integers(min_value=0, max_value=364)
                ),
                column("sex", strategy=st.sampled_from(["M", "F"])),
                column(
                    "inhospital_death", strategy=st.sampled_from([False] * 3 + [True])
                ),
            ],
            min_size=_min_size,
            max_size=_max_size,
        )
    )

    obs = df.with_columns(
        # IDs
        pl.col("id").add("-p").alias("patient_id"),
        pl.col("id").alias("case_id"),
        # Timestamps
        pl.col("hospital_admission")
        .add(pl.col("hospital_lenth_of_stay"))
        .alias("hospital_discharge"),
        pl.col("hospital_admission")
        .sub(
            pl.duration(
                days=pl.col("age_on_admission").mul(365).add(pl.col("birthday_offset"))
            )
        )
        .alias("birthdate"),
    ).with_columns(
        # Death timestamp
        pl.when("inhospital_death")
        .then("hospital_discharge")
        .otherwise(None)
        .alias("death_timestamp"),
    )

    select_cols = [
        "data_source",
        "patient_id",
        "case_id",
        "hospital_admission",
        "hospital_discharge",
        "hospital_status",
        "age_on_admission",
        "sex",
        "birthdate",
        "death_timestamp",
        "inhospital_death",
    ]

    # TODO: Improve this
    if obs_level == ObsLevel.ICU_STAY:
        select_cols += [
            "icu_stay_id",
            "icu_id",
            "icu_admission",
            "icu_discharge",
            "icu_status",
            "bw_fach_oe",
            "bw_patient_origin",
        ]
        # Generates ID and tmin, tmax
        icu_obs = _icu_stays(obs)
        obs = obs.join(icu_obs, on="case_id").with_columns(
            pl.lit(["PY-TEST"]).alias("icu_id"),
            pl.lit("Finalised").alias("icu_status"),
            pl.lit(["TEST-PY"]).alias("bw_fach_oe"),
            pl.lit(["PY"]).alias("bw_patient_origin"),
        )
    elif obs_level == ObsLevel.PROCEDURE:
        select_cols += [
            "procedure_id",
            "ops_code",
            "bw_pflege_oe",
            "bw_fach_oe",
            "or_time_ops",
            "or_time_begin",
            "or_time_end",
            "or_time_anes_begin",
            "or_time_anes_ready",
            "or_time_anes_end",
            "or_time_incision",
            "or_time_suture",
        ]

        # Generates ID and tmin, tmax
        procedure = _procedures(obs)
        obs = obs.join(procedure, on="case_id").with_columns(
            pl.lit(["0-000.0A", "0-000.0", "4-20.0"]).alias("ops_code"),
            pl.lit(["TEST"]).alias("bw_pflege_oe"),
            pl.lit(["TEST-PY"]).alias("bw_fach_oe"),
        )

    return obs.select(select_cols)


def _icu_stays(obs: pl.DataFrame) -> pl.DataFrame:
    """Generates a dataframe with ICU stays.

    An icu stay should be between the hospital admission and discharge
    Each icu stay should have a length of stay (LOS) smaller 30 days (prefer long LOS)
    with a gap of at least 2 days to previous and following stays.
    """
    icu_df = (
        obs.select("case_id", "hospital_admission", "hospital_discharge", "icu_stays")
        .with_columns(
            time_difference(
                "hospital_discharge", "hospital_admission", unit="d", total=True
            ).alias("hospital_lenth_of_stay_days"),
            pl.int_ranges(0, pl.col("icu_stays")).alias("icu_stay_num"),
        )
        .explode("icu_stay_num")
        .with_columns(
            pl.col("hospital_lenth_of_stay_days")
            .sub(pl.col("icu_stays").sub(1).mul(2))
            .truediv(pl.col("icu_stays"))
            .clip(upper_bound=30)
            .alias("icu_los_days"),
        )
        .with_columns(
            pl.col("icu_stay_num")
            .mul(pl.col("icu_los_days").add(2))
            .alias("icu_los_offset")
        )
        .with_columns(
            (
                pl.col("hospital_admission")
                + pl.duration(days=pl.col("icu_los_offset"))
            ).alias("icu_admission")
        )
        .with_columns(
            (pl.col("icu_admission") + pl.duration(days=pl.col("icu_los_days"))).alias(
                "icu_discharge"
            )
        )
        .with_columns(
            pl.col("case_id")
            .add(pl.lit("-"))
            .add(pl.col("icu_stay_num").cast(pl.Utf8))
            .alias("icu_stay_id")
        )
    )
    return icu_df.select(
        "case_id",
        "icu_stay_id",
        "icu_admission",
        "icu_discharge",
    )


def _procedures(obs: pl.DataFrame) -> pl.DataFrame:
    """Generates a dataframe with procedures.

    An procedure should be between the hospital admission and discharge
    Each procedure should have a duration smaller 24 hours
    with a gap of at least 12 hours to previous and following procedures.
    """
    procedure_df = (
        obs.select("case_id", "hospital_admission", "hospital_discharge", "procedures")
        .with_columns(
            time_difference(
                "hospital_discharge", "hospital_admission", unit="d", total=True
            ).alias("hospital_lenth_of_stay_hours"),
            pl.int_ranges(0, pl.col("procedures")).alias("procedure_num"),
        )
        .explode("procedure_num")
        .with_columns(
            pl.col("hospital_lenth_of_stay_hours")
            .sub(pl.col("procedures").sub(1).mul(12))
            .truediv(pl.col("procedures"))
            .clip(upper_bound=24)
            .alias("procedure_duration_hours"),
        )
        .with_columns(
            pl.col("procedure_num")
            .mul(pl.col("procedure_duration_hours").add(12))
            .alias("procedure_duration_offset")
        )
        .with_columns(
            (
                pl.col("hospital_admission")
                + pl.duration(hours=pl.col("procedure_duration_offset"))
            ).alias("or_time_begin")
        )
        .with_columns(
            (
                pl.col("or_time_begin")
                + pl.duration(hours=pl.col("procedure_duration_offset"))
            ).alias("or_time_end")
        )
        .with_columns(
            pl.col("case_id")
            .add(pl.lit("-"))
            .add(pl.col("procedure_num").cast(pl.Utf8))
            .alias("procedure_id")
        )
    )
    return procedure_df.select(
        "case_id",
        "procedure_id",
        "or_time_begin",
        "or_time_end",
        # TODO: Simplified approach for other times
        pl.col("or_time_begin").dt.truncate("1d").alias("or_time_ops"),
        pl.col("or_time_begin").alias("or_time_anes_begin"),
        pl.col("or_time_begin").alias("or_time_anes_ready"),
        pl.col("or_time_end").alias("or_time_anes_end"),
        pl.col("or_time_end").alias("or_time_incision"),
        pl.col("or_time_end").alias("or_time_suture"),
    )
