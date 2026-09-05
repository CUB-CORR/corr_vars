from __future__ import annotations

import os
import subprocess
import sys
import textwrap

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corr_vars.core.cohort import Cohort

__all__ = ["print_cohort_debug_info", "print_debug_info"]


def print_cohort_debug_info(cohort: Cohort) -> None:
    """Prints useful information about Cohort objects."""
    print(
        textwrap.dedent("""\
            Cohort (repr)
            {cohort}

            Cohort (info)
            - t_eligible: {t_eligible}
            - t_outcome: {t_outcome}

            ObsLevel (repr)
            {obs_level}

            Obs (info):
            - obs.shape: {obs_shape}
            - obs.columns:
                {obs_columns}

            Obsm (repr)
            {obsm}
            """).format_map(
            {
                "cohort": repr(cohort),
                "t_eligible": cohort.t_eligible,
                "t_outcome": cohort.t_outcome,
                "obs_level": repr(cohort.obs_level),
                "obsm": repr(cohort.obsm),
                "obs_shape": cohort._obs.shape,
                "obs_columns": textwrap.indent(
                    textwrap.fill(", ".join(cohort._obs.columns)), " " * 4
                ).lstrip(),
            }
        )
    )


def print_debug_info() -> None:
    """Prints the current package version, interpreter path, and other useful information."""
    try:
        # Get the current package version
        from corr_vars import __version__ as version

        print(f"Package version: {version}")
    except ImportError:
        print("Package version: Not available")

    # Get the interpreter path
    interpreter_path = sys.executable
    print(f"Interpreter path: {interpreter_path}")

    # Get the current commit hash
    try:
        package_path = os.path.dirname(os.path.abspath(__file__))
        commit_info = (
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    package_path,
                    "log",
                    "-1",
                    "--format=%h \n - Timestamp: %ad \n - Author: %an",
                ]
            )
            .strip()
            .decode("utf-8")
        )
        print(f"Current commit: {commit_info}")
    except Exception as e:
        print(f"Current commit: Not available ({e})")

    # Print the current working directory
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")

    # Print the platform information
    platform_info = sys.platform
    print(f"Platform: {platform_info}")

    # Print the Python version
    python_version = sys.version
    print(f"Python version: {python_version}")
