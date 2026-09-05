import builtins
import subprocess

import pytest

from corr_vars.utils.debug import print_cohort_debug_info, print_debug_info


def test_print_cohort_debug_info(capsys: pytest.CaptureFixture, dummy_cohort) -> None:
    """Test the output of the cohort debug info."""
    print_cohort_debug_info(dummy_cohort)
    captured = capsys.readouterr()
    assert "Cohort (repr)" in captured.out
    assert "Cohort (info)" in captured.out
    assert "ObsLevel (repr)" in captured.out
    assert "Obs (info)" in captured.out
    assert "Obsm (repr)" in captured.out


def test_print_debug_info_basic(capsys: pytest.CaptureFixture) -> None:
    """Smoke test: normal run prints the expected sections."""
    print_debug_info()
    captured = capsys.readouterr()
    assert "Package version:" in captured.out
    assert "Interpreter path:" in captured.out
    assert "Current working directory:" in captured.out
    assert "Platform:" in captured.out
    assert "Python version:" in captured.out


def test_print_debug_info_import_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Simulate ImportError when importing corr_vars.__version__."""
    orig_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "corr_vars" or name.startswith("corr_vars."):
            raise ImportError("simulated missing package")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        print_debug_info()
    finally:
        # ensure import behavior restored in case of test failure before monkeypatch teardown
        monkeypatch.setattr(builtins, "__import__", orig_import)
    captured = capsys.readouterr()
    assert "Package version: Not available" in captured.out


def test_print_debug_info_git_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Simulate subprocess.check_output raising an exception (git not available / repo missing)."""

    def raise_git(*args, **kwargs):
        raise Exception("git error simulated")

    monkeypatch.setattr(subprocess, "check_output", raise_git)
    print_debug_info()
    captured = capsys.readouterr()
    assert "Current commit: Not available" in captured.out
    assert "git error simulated" in captured.out
