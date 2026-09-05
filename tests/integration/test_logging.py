import contextlib
import logging
from pathlib import Path

import pytest

from corr_vars.utils.logging import (
    CustomFormatter,
    configure_logger_level_and_handlers,
    log_collection,
    log_dict,
    log_multiline_string,
)


def _clean_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # Important to catch the log
    logger.propagate = True
    return logger


def test_configure_logger_stream_colored_capsys(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = _clean_logger("test_cfg_stream_colored")
    caplog.set_level(logging.INFO, logger=logger.name)

    configure_logger_level_and_handlers(
        logger=logger,
        level=logging.INFO,
        file_path=None,
        colored_output=False,
        formatted_numbers=False,
    )

    logger.info("This is SUCCESS and DROP")

    message = caplog.records[0].getMessage()
    assert " SUCCESS " in message
    assert " DROP" in message
    assert CustomFormatter.green + "SUCCESS" + CustomFormatter.reset not in message
    assert CustomFormatter.red + "DROP" + CustomFormatter.reset not in message

    configure_logger_level_and_handlers(
        logger=logger,
        level=logging.INFO,
        file_path=None,
        colored_output=True,
        formatted_numbers=False,
    )

    logger.info("This is SUCCESS and DROP")

    message = caplog.records[1].getMessage()
    levelname = caplog.records[1].levelname
    assert (
        CustomFormatter.LEVEL_COLORS["INFO"] + "INFO" + CustomFormatter.reset
        in levelname
    )
    assert " SUCCESS " not in message
    assert " DROP" not in message
    assert CustomFormatter.green + "SUCCESS" + CustomFormatter.reset in message
    assert CustomFormatter.red + "DROP" + CustomFormatter.reset in message


def test_configure_logger_numbers_capsys(caplog: pytest.LogCaptureFixture) -> None:
    logger = _clean_logger("test_cfg_numbers")
    caplog.set_level(logging.INFO, logger=logger.name)

    configure_logger_level_and_handlers(
        logger=logger,
        level=logging.INFO,
        file_path=None,
        colored_output=False,
        formatted_numbers=False,
    )

    logger.info("There are 42 items")
    message = caplog.records[0].getMessage()
    assert " 42 " in message
    assert CustomFormatter.underline + "42" + CustomFormatter.reset not in message

    configure_logger_level_and_handlers(
        logger=logger,
        level=logging.INFO,
        file_path=None,
        colored_output=False,
        formatted_numbers=True,
    )

    logger.info("There are 42 items")
    message = caplog.records[1].getMessage()
    assert " 42 " not in message
    assert CustomFormatter.underline + "42" + CustomFormatter.reset in message


def test_configure_logger_writes_to_file(tmp_path: Path) -> None:
    logger = _clean_logger("test_cfg_file")

    logfile = tmp_path / "test_logging_output.log"
    configure_logger_level_and_handlers(
        logger=logger,
        level=logging.INFO,
        file_path=str(logfile),
        file_mode="w",
        colored_output=False,
        formatted_numbers=False,
    )

    logger.info("hello file")

    # ensure handlers flushed
    for h in logger.handlers:
        with contextlib.suppress(Exception):
            h.flush()

    content = logfile.read_text()
    assert "INFO - hello file" in content


def test_custom_formatter_colored_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure NO_COLOR is not set
    monkeypatch.delenv("NO_COLOR", raising=False)
    fmt = CustomFormatter(colored_output=None, formatted_numbers=False)
    assert fmt.colored is True


def test_custom_formatter_respects_no_color(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    fmt = CustomFormatter(colored_output=None, formatted_numbers=False)
    assert fmt.colored is False


def test_log_collection_tree_output(caplog: pytest.LogCaptureFixture) -> None:
    logger = _clean_logger("test_log_collection_tree_output")
    caplog.set_level(logging.INFO, logger=logger.name)

    items = ["a", "b", "c"]
    log_collection(logger=logger, collection=items, level=logging.INFO, as_tree=True)

    messages = [rec.getMessage() for rec in caplog.records]
    assert messages == ["├── a", "├── b", "└── c"]


def test_log_collection_no_tree_output(caplog: pytest.LogCaptureFixture) -> None:
    logger = _clean_logger("test_log_collection_no_tree_output")
    caplog.set_level(logging.INFO, logger=logger.name)

    items = ["x", "y"]
    log_collection(logger=logger, collection=items, level=logging.INFO, as_tree=False)

    messages = [rec.getMessage() for rec in caplog.records]
    assert messages == ["x", "y"]


def test_log_multiline_string(caplog: pytest.LogCaptureFixture) -> None:
    logger = _clean_logger("test_log_multiline_string")
    caplog.set_level(logging.INFO, logger=logger.name)

    multiline = "first line\nsecond line\nthird line"
    log_multiline_string(logger=logger, multiline=multiline, level=logging.INFO)

    messages = [rec.getMessage() for rec in caplog.records]
    assert messages == ["first line", "second line", "third line"]


def test_log_dict_output(caplog: pytest.LogCaptureFixture) -> None:
    logger = _clean_logger("test_log_dict_output")
    caplog.set_level(logging.INFO, logger=logger.name)

    dictionary = {"a": 1, "b": 2, "c": 3}
    log_dict(
        logger=logger,
        dictionary=dictionary,
        level=logging.INFO,
        json_indent=0,
        indent=0,
    )

    messages = [rec.getMessage() for rec in caplog.records]
    assert messages == ["{", '"a": 1,', '"b": 2,', '"c": 3', "}"]
