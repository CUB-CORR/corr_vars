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
