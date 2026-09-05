"""Dependency-free primitives shared by the other :mod:`corr_vars.utils` modules.

Everything here operates on a single frame, expression or offset and pulls in
nothing from the rest of ``corr_vars.utils``, so any module may import it without
risking an import cycle.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from corr_vars.definitions import OffsetLike

    from polars._typing import ColumnNameOrSelector

__all__ = [
    "as_expr",
    "col_length",
    "column_selector",
    "columns",
    "is_empty",
    "normalize_offset",
    "row_length",
    "struct_fields",
]


def as_expr(value: pl.Expr | str) -> pl.Expr:
    if isinstance(value, pl.Expr):
        return value
    return pl.col(value)


def columns(value: pl.DataFrame | pl.LazyFrame) -> list[str]:
    if isinstance(value, pl.DataFrame):
        return value.columns
    return value.collect_schema().names()


def struct_fields(
    value: pl.DataFrame | pl.LazyFrame, field: ColumnNameOrSelector
) -> list[str]:
    return columns(value.select(field).unnest(field))


def col_length(value: pl.DataFrame | pl.LazyFrame) -> int:
    return len(columns(value))


def row_length(value: pl.DataFrame | pl.LazyFrame) -> int:
    if isinstance(value, pl.DataFrame):
        return len(value)
    return cast("int", value.select(pl.len()).collect().item())


def is_empty(value: pl.DataFrame | pl.LazyFrame) -> bool:
    return row_length(value) == 0


def column_selector(value: str) -> pl.Expr:
    splits = value.split(".")
    curr = pl.col(splits[0])
    if len(splits) > 1:
        for part in splits[1:]:
            curr = curr.struct.field(part)
    return curr


def normalize_offset(offset: OffsetLike) -> str:
    """Coerce an offset to the polars duration language.

    A polars-duration string (``"28d"``, ``"6h"``) is returned unchanged; a
    :class:`~datetime.timedelta` is rendered in microseconds. This is the single place
    that turns an :data:`~corr_vars.definitions.typing.OffsetLike` into the string
    ``dt.offset_by`` needs — which does not accept a timedelta.
    """
    if isinstance(offset, timedelta):
        return f"{offset // timedelta(microseconds=1)}us"
    return offset
