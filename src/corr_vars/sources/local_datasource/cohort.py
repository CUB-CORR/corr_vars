"""Observation-level data loading for the ``local_datasource`` source.

Like :mod:`~corr_vars.sources.local_datasource.extract`'s ``NativeExtractor``,
this is the seam between the source skeleton and a deployment's own store: the
query that builds the observation frame is deployment-specific and is not part of
the published package.
"""

from __future__ import annotations

import polars as pl

from corr_vars.definitions.typing import ObsLevel

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corr_vars.sources.local_datasource.config import SourceDict


def load_data(obs_level: ObsLevel, source_kwargs: SourceDict) -> pl.DataFrame:
    """Build the cohort's observation frame from the backing store.

    Args:
        obs_level (ObsLevel): Level of observation the frame is keyed at.
        source_kwargs (SourceDict): This source's resolved configuration.

    Returns:
        pl.DataFrame: One row per observation, carrying at least the level's
        primary key and its ``t_min`` / ``t_max`` anchor columns.

    Raises:
        NotImplementedError: Always. ``local_datasource`` is a source skeleton;
            replace this function to bind it to a store.
    """
    raise NotImplementedError(
        f"local_datasource ships no query for obs_level {obs_level.lower_name!r}: "
        "it is a source skeleton carrying the variable classes and the extraction "
        "pipeline, not a connection to any particular store. Replace "
        "corr_vars.sources.local_datasource.cohort.load_data to bind it to one."
    )
