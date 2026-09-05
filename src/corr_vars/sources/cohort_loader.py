import polars as pl

import corr_vars.sources as corr
from corr_vars import __, logger, utils
from corr_vars.definitions import ObsLevel, SourceDict

from typing import cast


def load_cohort_data(sources: SourceDict, obs_level: ObsLevel) -> pl.DataFrame:
    """Load cohort data from multiple sources.

    Args:
        sources (dict): Dictionary of data sources with conigurations to use for data extraction.
        obs_level (ObsLevel): Observation level to load data for.

    Returns:
        obs (pl.DataFrame): Static data for each observation. Contains one row per observation (e.g., ICU stay)
        with columns for static variables like demographics and outcomes.
    """
    # Load data for each source
    obs_dfs = []
    for source, source_kwargs in sources.items():
        src_cohort = corr.get_src_module(source=source, submodule="cohort")
        # load_data should accept ObsLevel and the respective SourceDict as arguments
        # and return a polars DataFrame
        obs = cast("pl.DataFrame", src_cohort.load_data(obs_level, source_kwargs))
        # Add data_source as first column to the polars DataFrame
        obs.insert_column(0, pl.lit(source).alias("data_source"))
        obs_dfs.append(obs)
        logger.info(
            __(
                "SUCCESS: Loaded {n_obs} rows from {source}",
                n_obs=len(obs),
                source=source,
            )
        )

    # Concatenate all polars DataFrames
    if not obs_dfs:
        raise ValueError("No source returned data.")

    obs_dfs = utils.harmonize_str_list_cols(obs_dfs)
    return pl.concat(obs_dfs, how="diagonal_relaxed")
