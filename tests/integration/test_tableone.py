import subprocess
from pathlib import Path

import pandas as pd
import pytest
from tableone import TableOne

from corr_vars.utils.tableone import (
    tableone_to_html,
    tableone_to_latex,
    tableone_to_markdown,
    tableone_to_pdf,
)

from typing import Protocol


@pytest.fixture
def iris_tableone() -> TableOne:
    """Fixture to create a sample TableOne object for testing."""
    data = pd.read_csv(Path(__file__).parent.parent / "assets" / "data" / "iris.csv")
    return TableOne(data=data, columns=data.columns.tolist(), groupby="species")


class TableOneConverter(Protocol):
    def __call__(self, tableone: TableOne, file_path: Path | None = None) -> str: ...


@pytest.mark.parametrize(
    "func, extension, substrings",
    [
        (tableone_to_latex, "tex", [r"\documentclass", "TABLE1 START"]),
        (tableone_to_markdown, "md", ["|"]),
        (tableone_to_html, "html", ["<html>", "</table>"]),
    ],
)
def test_tableone_conversions(
    iris_tableone: TableOne,
    tmp_path: Path,
    func: TableOneConverter,
    extension: str,
    substrings: list[str],
) -> None:
    # Logic is the same as the helper above
    result_str = func(iris_tableone)
    assert isinstance(result_str, str)
    assert all(s in result_str for s in substrings)

    base_path = tmp_path / "output"
    expected_file = tmp_path / f"output.{extension}"

    func(iris_tableone, file_path=base_path)
    assert expected_file.read_text(encoding="utf-8") == result_str


def test_tableone_to_pdf_success(
    iris_tableone: TableOne, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tests successful PDF generation and full cleanup."""

    # 1. Mock subprocess.run to simulate success
    def mock_run(args, **kwargs):
        # args[2] is -output-directory, args[3] is the .tex path
        # Simulate pdflatex creating the files
        base = Path(args[3]).with_suffix("")
        base.with_suffix(".pdf").touch()
        base.with_suffix(".aux").touch()
        base.with_suffix(".log").touch()

    monkeypatch.setattr(subprocess, "run", mock_run)

    file_path = tmp_path / "subdir" / "my_table"

    # 2. Run the function
    tableone_to_pdf(iris_tableone, file_path)

    # 3. Assertions
    pdf_file = tmp_path / "subdir" / "my_table.pdf"
    assert pdf_file.exists()

    # Verify cleanup: .tex, .aux, and .log should be unlinked on success
    for ext in [".tex", ".aux", ".log"]:
        assert not (tmp_path / "subdir" / f"my_table{ext}").exists()


def test_tableone_to_pdf_compilation_failure(
    iris_tableone: TableOne, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tests that .log and .aux are preserved if pdflatex fails."""

    # 1. Mock subprocess.run to raise an error
    def mock_run_fail(args, **kwargs):
        # Even if it fails, pdflatex usually leaves a .log and .aux behind
        base = Path(args[3]).with_suffix("")
        base.with_suffix(".aux").touch()
        base.with_suffix(".log").touch()
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(subprocess, "run", mock_run_fail)

    file_path = tmp_path / "fail_table"

    # 2. Run function (it catches the error and logs it, so no pytest.raises)
    tableone_to_pdf(iris_tableone, file_path)

    # 3. Assertions
    # .tex should still be cleaned up (per your finally block logic)
    assert not (tmp_path / "fail_table.tex").exists()

    # .aux and .log should still exist because we updated the 'cleanup' list
    assert (tmp_path / "fail_table.aux").exists()
    assert (tmp_path / "fail_table.log").exists()
