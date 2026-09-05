# tests/conftest.py
import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--var_names",
        action="store",
        help="Comma-separated list of variable names to test",
    )
