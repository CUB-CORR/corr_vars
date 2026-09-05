import os
import tempfile

import pytest

from corr_vars.core.file_manager import TemporaryDirectoryManager


@pytest.fixture
def temp_dir_manager() -> TemporaryDirectoryManager:
    """Fixture to create a TemporaryDirectoryManager instance."""
    return TemporaryDirectoryManager()


def test_create_and_delete_tmpdir(temp_dir_manager: TemporaryDirectoryManager) -> None:
    """Test the creation and deletion of the temporary directory."""
    path = temp_dir_manager.create_tmpdir()
    assert os.path.exists(path)
    temp_dir_manager.delete_tmpdir()
    assert not os.path.exists(path)


def test_check_exist(temp_dir_manager: TemporaryDirectoryManager) -> None:
    """Test the _check_exist method."""
    assert not temp_dir_manager._check_exist()
    temp_dir_manager.create_tmpdir()
    assert temp_dir_manager._check_exist()
    temp_dir_manager.delete_tmpdir()
    assert not temp_dir_manager._check_exist()


def test_create_tmpdir_with_path() -> None:
    """Test creating a temporary directory with a specified path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = TemporaryDirectoryManager()
        path = manager.create_tmpdir(tmpdir_path=tmpdir)
        assert os.path.dirname(path) == tmpdir
        manager.delete_tmpdir()


def test_create_tmpdir_with_not_existing_path(
    temp_dir_manager: TemporaryDirectoryManager,
) -> None:
    """Test creating a temporary directory with a non-existing path."""
    not_existing_path = "/tmp/not_existing_path"
    with pytest.warns(
        UserWarning,
        match=f"Specified temporary directory at {not_existing_path} does not exist. Will proceed with default path.",
    ):
        path = temp_dir_manager.create_tmpdir(tmpdir_path=not_existing_path)
    assert os.path.exists(path)
    assert path != not_existing_path
    temp_dir_manager.delete_tmpdir()


def test_delete_tmpdir_variable(temp_dir_manager: TemporaryDirectoryManager) -> None:
    """Test the deletion of variable files."""
    _ = temp_dir_manager.create_tmpdir()
    var_name = "test_var"
    file_path = temp_dir_manager.create_tmpdir_variable_path(var_name)
    with open(file_path, "w") as f:
        f.write("test")

    assert os.path.exists(file_path)
    temp_dir_manager.delete_tmpdir_variable(var_name)
    assert not os.path.exists(file_path)
    temp_dir_manager.delete_tmpdir()


def test_delete_tmpdir_variable_matches_the_name_exactly(
    temp_dir_manager: TemporaryDirectoryManager,
) -> None:
    """A variable whose name merely starts the same is left alone."""
    _ = temp_dir_manager.create_tmpdir()
    kept = temp_dir_manager.create_tmpdir_variable_path("test_var_extended")
    for path in (temp_dir_manager.create_tmpdir_variable_path("test_var"), kept):
        with open(path, "w") as f:
            f.write("test")

    temp_dir_manager.delete_tmpdir_variable("test_var")

    assert os.path.exists(kept)
    temp_dir_manager.delete_tmpdir()


def test_delete_tmpdir_variable_removes_every_contributor(
    temp_dir_manager: TemporaryDirectoryManager,
) -> None:
    """Per-contributor and per-cache-key files of the variable all go."""
    _ = temp_dir_manager.create_tmpdir()
    paths = [
        temp_dir_manager.create_tmpdir_variable_path(name)
        for name in ("test_var@src#1:case_id", "test_var@src#2:case_id")
    ]
    for path in paths:
        with open(path, "w") as f:
            f.write("test")

    temp_dir_manager.delete_tmpdir_variable("test_var")

    assert not any(os.path.exists(path) for path in paths)
    temp_dir_manager.delete_tmpdir()


def test_create_tmpdir_variable_path(
    temp_dir_manager: TemporaryDirectoryManager,
) -> None:
    """Test the creation of a variable path."""
    var_name = "test_var"
    path = temp_dir_manager.create_tmpdir_variable_path(var_name)
    expected_path = os.path.join(
        temp_dir_manager.path,
        f"{temp_dir_manager.variable_file_prefix}{var_name}.parquet",
    )
    assert path == expected_path
    temp_dir_manager.delete_tmpdir()


def test_tmpdir_variables(temp_dir_manager: TemporaryDirectoryManager) -> None:
    """Test the listing of variable files."""
    # Should work without created tmpdir
    variables = temp_dir_manager.tmpdir_variables
    assert variables == []

    _ = temp_dir_manager.create_tmpdir()
    var_name1 = "test_var1"
    var_name2 = "test_var2"
    file_path1 = temp_dir_manager.create_tmpdir_variable_path(var_name1)
    file_path2 = temp_dir_manager.create_tmpdir_variable_path(var_name2)

    # Should work with created tmpdir, but without variables
    variables = temp_dir_manager.tmpdir_variables
    assert variables == []

    with open(file_path1, "w") as f:
        f.write("test1")
    with open(file_path2, "w") as f:
        f.write("test2")

    # Should work with created tmpdir and variables
    variables = temp_dir_manager.tmpdir_variables
    assert len(variables) == 2
    assert os.path.basename(file_path1) in variables
    assert os.path.basename(file_path2) in variables

    temp_dir_manager.delete_tmpdir()


def test_path_property_runtime_error(
    temp_dir_manager: TemporaryDirectoryManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test if the path property raises a RuntimeError if the path is not established."""
    monkeypatch.setattr(temp_dir_manager, "create_tmpdir", lambda check_existing: None)
    with pytest.raises(
        RuntimeError, match="Temporary directory path could not be established."
    ):
        _ = temp_dir_manager.path
