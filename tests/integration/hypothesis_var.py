import re
from datetime import datetime

import hypothesis.strategies as st
import polars as pl
from polars.testing.parametric import column, dataframes

from corr_vars.definitions import ObsLevel

from collections.abc import Callable
from typing import Literal

# Artificially capped to 300 unique combinations
ID_REGEX = re.compile(r"py-0{8}[0-2][0-9]-[0-9]")


@st.composite
def var_dataframe(
    draw: Callable[[st.SearchStrategy[pl.DataFrame]], pl.DataFrame],
    id_strategy: st.SearchStrategy[str] | None = None,
    size: int | tuple[int, int] | None = None,
    vtype: Literal["static", "dynamic_ts_snapshot", "dynamic_ts_interval"] = "static",
    obs_level: ObsLevel = ObsLevel.HOSPITAL_STAY,
    tmin: datetime | None = None,
    tmax: datetime | None = None,
) -> pl.DataFrame:
    if size is None:
        _min_size = _max_size = 100
    elif isinstance(size, int):
        _min_size = _max_size = size
    else:
        _min_size, _max_size = size

    interval_duration_column = column(
        "interval_duration",
        strategy=st.integers(min_value=60, max_value=24 * 60 * 60),
    )

    df: pl.DataFrame = draw(
        dataframes(
            cols=[
                column(
                    obs_level.primary_key,
                    strategy=(id_strategy or st.from_regex(ID_REGEX, fullmatch=True)),
                    unique=(vtype == "static"),
                ),
                column(
                    "recordtime",
                    strategy=st.datetimes(
                        min_value=tmin or datetime(2000, 1, 1),
                        max_value=tmax or datetime(datetime.now().year, 1, 1),
                    ),
                ),
                column(
                    "value",
                    strategy=st.floats(min_value=0, max_value=100, allow_nan=False),
                ),
                *([interval_duration_column] if vtype == "dynamic_ts_interval" else []),
            ],
            min_size=_min_size,
            max_size=_max_size,
            allow_null=False,
        )
    )

    return df.with_columns(
        (
            pl.col("recordtime").add(pl.duration(seconds=pl.col("interval_duration")))
            if vtype == "dynamic_ts_interval"
            else pl.lit(None)
        ).alias("recordtime_end"),
    ).select(
        obs_level.primary_key,
        *(
            [
                "recordtime",
                "recordtime_end",
            ]
            if vtype != "static"
            else []
        ),
        "value",
    )
