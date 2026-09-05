import subprocess
import textwrap
from os import PathLike
from pathlib import Path

from tableone import TableOne

from corr_vars import logger

from typing import overload

__all__ = [
    "tableone_to_html",
    "tableone_to_latex",
    "tableone_to_markdown",
    "tableone_to_pdf",
]


@overload
def tableone_to_latex(tableone: TableOne, file_path: None = ...) -> str: ...


@overload
def tableone_to_latex(tableone: TableOne, file_path: str | PathLike[str]) -> None: ...


def tableone_to_latex(
    tableone: TableOne, file_path: str | PathLike[str] | None = None
) -> str | None:
    """Convert a TableOne object to a LaTeX table string.

    Args:
        tableone (TableOne): The TableOne object to convert.
        file_path (str | PathLike[str] | None): The path to the file where the LaTeX table will be saved.

    Returns:
        str: A LaTeX table string representing the TableOne object if file_path is None, otherwise None.
    """
    LATEX_TEMPLATE_START = textwrap.dedent(r"""
        \documentclass[border = 50pt]{standalone}
        % Table stuff
        \usepackage{tabularx}
        \usepackage{booktabs}
        \usepackage{multirow}
        % Remove page num
        \pagestyle{empty}
        \setlength\extrarowheight{1mm}
        \begin{document}
        % TABLE1 START
        """).lstrip()

    LATEX_TEMPLATE_END = textwrap.dedent(r"""
        % TABLE1 END
        \end{document}
        """).rstrip()

    latex_string = (
        LATEX_TEMPLATE_START
        + tableone.tabulate(tablefmt="latex_booktabs")
        + LATEX_TEMPLATE_END
    )

    if file_path:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".tex").write_text(latex_string, encoding="utf-8")
        return None
    return latex_string


@overload
def tableone_to_markdown(tableone: TableOne, file_path: None = ...) -> str: ...


@overload
def tableone_to_markdown(
    tableone: TableOne, file_path: str | PathLike[str]
) -> None: ...


def tableone_to_markdown(
    tableone: TableOne, file_path: str | PathLike[str] | None = None
) -> str | None:
    """Convert a TableOne object to a Markdown table string.

    Args:
        tableone (TableOne): The TableOne object to convert.
        file_path (str | PathLike[str] | None): The path to the file where the Markdown table will be saved.

    Returns:
        str: A Markdown table string representing the TableOne object if file_path is None, otherwise None.
    """
    markdown_string = tableone.tabulate(tablefmt="github")

    if file_path:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".md").write_text(markdown_string, encoding="utf-8")
        return None
    return markdown_string


@overload
def tableone_to_html(tableone: TableOne, file_path: None = ...) -> str: ...


@overload
def tableone_to_html(tableone: TableOne, file_path: str | PathLike[str]) -> None: ...


def tableone_to_html(
    tableone: TableOne, file_path: str | PathLike[str] | None = None
) -> str | None:
    """Convert a TableOne object to an HTML table string.

    Args:
        tableone (TableOne): The TableOne object to convert.
        file_path (str | PathLike[str] | None): The path to the file where the HTML table will be saved.

    Returns:
        str: An HTML table string representing the TableOne object if file_path is None, otherwise None.
    """
    HTML_TEMPLATE_START = textwrap.dedent(r"""
        <html>
            <head>
                <title>TableOne</title>
            </head>
            <body>
        """).lstrip()

    HTML_TEMPLATE_END = textwrap.dedent(r"""
            </body>
        </html>
        """).rstrip()

    html_string = (
        HTML_TEMPLATE_START
        + textwrap.indent(tableone.tabulate(tablefmt="html"), "    " * 2)
        + HTML_TEMPLATE_END
    )

    if file_path:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".html").write_text(html_string, encoding="utf-8")
        return None
    return html_string


def tableone_to_pdf(tableone: TableOne, file_path: str | PathLike[str]) -> None:
    """Convert a TableOne object to a PDF file.
    This function generates a LaTeX file from the TableOne object and compiles it to PDF using pdflatex.
    Note that pdflatex must be installed and available in the system PATH for this function to work.

    Args:
        tableone (TableOne): The TableOne object to convert.
        file_path (str | PathLike[str]): The path to the file where the PDF table will be saved.

    Returns:
        None
    """
    pdf_path = Path(file_path).with_suffix(".pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    tex_path = Path(file_path).with_suffix(".tex")
    tableone_to_latex(tableone, tex_path)

    # Call pdflatex
    cleanup = [".aux", ".log", ".tex"]
    try:
        # -interaction=nonstopmode prevents the script from hanging on errors
        subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                f"-output-directory={pdf_path.parent.as_posix()}",
                tex_path.as_posix(),
            ],
            check=True,
        )
    except subprocess.CalledProcessError:
        cleanup = [".tex"]  # Don't delete .log and .aux if compilation fails
        logger.error("Compilation failed. Check the .log and .aux files for details.")
    finally:
        # Cleanup auxiliary files
        for suffix in cleanup:
            if pdf_path.with_suffix(suffix).exists():
                pdf_path.with_suffix(suffix).unlink()
