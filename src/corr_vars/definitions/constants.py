from datetime import datetime, timezone

from .typing import ObsLevel

from typing import Final

# The timestamps an unbounded window bound is filled with. They stand for
# "no bound" rather than for a point in time, so they shouldn't be shifted.
UNBOUNDED_TMIN: Final = datetime.min.replace(tzinfo=None)
UNBOUNDED_TMAX: Final = datetime.max.replace(tzinfo=None)

UNIX_EPOCH: Final = datetime(1970, 1, 1, tzinfo=timezone.utc)

DYN_COLUMNS: Final = (
    "recordtime",
    "recordtime_end",
    "recordtime_relative",
    "recordtime_end_relative",
    "value",
    "value_unit",
    "attributes",
)

PRIMARY_KEYS: Final = tuple(ObsLevel.primary_keys())

COL_ORDER: Final = (*PRIMARY_KEYS, *DYN_COLUMNS)
