import glob
import os
import re
import tempfile
import warnings

from corr_vars import __, logger

from typing import Final, cast


class TemporaryDirectoryManager:
    """Manages the creation and cleanup of a temporary directory."""

    variable_file_prefix: Final = "var_c_"
    variable_file_pattern: Final = re.compile(
        r"^var_c_(?P<var_name>[^\.]+?)(?::(?P<cache_key>[^\.]+?))?\..+$", re.IGNORECASE
    )
    _tmpdir: tempfile.TemporaryDirectory | None
    _path: str | None

    def __init__(self) -> None:
        self._tmpdir = None
        self._path = None

    @property
    def path(self) -> str:
        """The path to the temporary directory.
        Calls create_tmpdir(check_existing=True) to ensure the directory is created
        or validated before the path is returned.
        """
        self.create_tmpdir(check_existing=True)
        if self._path is None:
            raise RuntimeError("Temporary directory path could not be established.")
        return self._path

    def _check_exist(self) -> bool:
        return bool(
            self._path
            and os.path.exists(self._path)
            and os.access(self._path, os.W_OK)
            and self._tmpdir
            and isinstance(self._tmpdir, tempfile.TemporaryDirectory)
            and self._tmpdir.name == self._path
        )

    def create_tmpdir(
        self, tmpdir_path: str | None = None, check_existing: bool = False
    ) -> str:
        """Creates a temporary directory for storing intermediary data.

        Args:
            tmpdir_path (str | None): Path to the temporary directory (default: None).
            check_existing (bool): Checks if an existing temporary directory is valid (default: False).

        Returns:
            The created or reused temporary directory path.
        """
        if check_existing and self._check_exist():
            logger.debug(
                __("Using existing temporary directory {path}", path=self._path)
            )
            return cast("str", self._path)

        # Create temporary directory for storing intermediary data
        # NOTE: Default path is already set for the corr_vars module by the global tempfile.tempdir
        tmp_path = tmpdir_path
        if tmp_path is not None and not os.path.exists(tmp_path):
            warnings.warn(
                f"Specified temporary directory at {tmp_path} does not exist. Will proceed with default path.",
                UserWarning,
            )
            tmp_path = None

        self._tmpdir = tempfile.TemporaryDirectory(dir=tmp_path)
        self._path = cast("str", self._tmpdir.name)
        logger.debug(__("Created temporary directory at {path}", path=self._path))
        return self._path

    def delete_tmpdir(self) -> None:
        """Cleans up the temporary directory."""
        if isinstance(self._tmpdir, tempfile.TemporaryDirectory):
            self._tmpdir.cleanup()
            self._tmpdir = None
            self._path = None

    def delete_tmpdir_variable(self, var_name: str) -> None:
        """Cleans up the files of variable `var_name` in the temporary directory.

        Removes every contributor's copy — a source files its cache under
        ``<var_name>@<contributor_key>`` when several contributors share the
        name — but never a different variable that merely starts with the same
        characters.
        """
        if self._check_exist():
            candidates = [
                file_path
                for file_path in glob.glob(
                    f"{self.variable_file_prefix}{var_name}*",
                    root_dir=self._path,
                )
                if self._matches_variable(os.path.basename(file_path), var_name)
            ]
            for file_path in candidates:
                os.unlink(os.path.join(cast("str", self._path), file_path))
                logger.info(__("Deleted: {path}", path=file_path))

    def _matches_variable(self, file_name: str, var_name: str) -> bool:
        """Whether `file_name` is a file of variable `var_name`."""
        match = self.variable_file_pattern.match(file_name)
        if match is None:
            return False
        matched = match.group("var_name")
        return matched == var_name or matched.startswith(f"{var_name}@")

    def create_tmpdir_variable_path(
        self, var_name: str, extension: str | None = "parquet"
    ) -> str:
        """Returns the expected file path to a variable called `var_name` in the temporary directory."""
        return os.path.join(
            self.path,
            f"{self.variable_file_prefix}{var_name}{'.' + extension if extension else ''}",
        )

    @property
    def tmpdir_variables(self) -> list[str]:
        """Returns a list of variable parquet files in the temporary directory."""
        if self._check_exist():
            return [
                file_path
                for file_path in glob.glob(
                    f"{self.variable_file_prefix}*.parquet",
                    root_dir=self._path,
                )
                if self.variable_file_pattern.match(os.path.basename(file_path))
            ]
        return []
