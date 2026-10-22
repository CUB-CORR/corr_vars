"""Helpers shared by this source's variable classes."""

from __future__ import annotations


def parse_select(select_str: str):
    """Parse the select syntax to extract function name, parameters and columns.

    Args:
        select_str (str): The select string.

    Returns:
        tuple: (function_name, params, columns)

    Examples:
        - !first value, recordtime
        - !closest(timestamp, 1d, 2d) value, recordtime
        - !any value
        - !last value, recordtime
    """
    select_str = select_str.strip()

    # Extract function name
    if select_str.startswith("!"):
        if "(" in select_str:
            # Handle function with parameters
            func_name = select_str[1 : select_str.find("(")].strip()
            params_str = select_str[
                select_str.find("(") + 1 : select_str.find(")")
            ].strip()
            params = [p.strip() for p in params_str.split(",")]
            cols_str = select_str[select_str.find(")") + 1 :].strip()
        else:
            # Handle function without parameters
            parts = select_str[1:].split(" ", 1)
            func_name = parts[0].strip()
            params = []
            cols_str = parts[1] if len(parts) > 1 else ""
    else:
        func_name, params = "", []
        cols = select_str.split(",")
        return func_name, params, cols

    # Extract columns to select
    columns = [c.strip() for c in cols_str.split(",")] if cols_str else []

    return func_name, params, columns


def parse_where_function(where_str: str) -> tuple[str, str, list[str]]:
    """Parse a magic ``where`` filter of the form ``!func(column, [values])``.

    :func:`parse_select` splits its parameters on every comma, which breaks a
    list argument apart (``"['I50'"``, ``"'I42.0']"``, ...). This parser keeps
    the list intact.

    Args:
        where_str (str): The where string, e.g.
            ``"!startswith(value, ['I50', 'I42.0'])"``.

    Returns:
        tuple: ``(function_name, column, values)``. A single value given
        without brackets becomes a one-element list.

    Raises:
        ValueError: If the string is not of that form, or its value list is
            empty.

    Examples:
        >>> parse_where_function("!startswith(value, ['I50', 'I42.0'])")
        ('startswith', 'value', ['I50', 'I42.0'])
        >>> parse_where_function("!isin(value, 'A')")
        ('isin', 'value', ['A'])
    """
    import ast
    import re

    match = re.fullmatch(
        r"\s*!\s*(\w+)\s*\(\s*([^,()]+?)\s*,\s*(.*)\)\s*", where_str, flags=re.DOTALL
    )
    if match is None:
        raise ValueError(
            f"Cannot parse where filter {where_str!r}; expected '!func(column, [values])'"
        )
    func_name, column, values_str = match.groups()
    try:
        values = ast.literal_eval(values_str.strip())
    except (ValueError, SyntaxError):
        # Bare, unquoted values: "!isin(value, A, B)"
        values = [v.strip().strip("'\"") for v in values_str.split(",") if v.strip()]
    if isinstance(values, (str, int, float)):
        values = [values]
    if not values:
        raise ValueError(
            f"Where filter {where_str!r} has an empty value list; "
            "give at least one value, e.g. '!func(column, ['A'])'"
        )
    return func_name, column.strip(), [str(v) for v in values]
