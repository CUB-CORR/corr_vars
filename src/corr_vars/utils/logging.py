import json
import logging
import os
import re
import textwrap

from collections.abc import Callable, Collection, Iterable, Mapping
from typing import Any, Final, Literal, TypeVar

T = TypeVar("T")

__all__ = [
    "CustomFormatter",
    "configure_logger_level_and_handlers",
    "log_collection",
    "log_dict",
    "log_multiline_string",
    "pretty_join",
    "text_indent",
    "text_tree_indent",
]


class CustomFormatter(logging.Formatter):
    """Logging formatter that optionally colorizes output and highlights numbers.

    Features:
    - Optionally colorizes level names and certain keywords (e.g. SUCCESS, DROP).
    - Optionally underlines numeric tokens in messages for emphasis.
    - Respects NO_COLOR environment variable when deciding to colorize.
    """

    white = "\x1b[37m"
    green = "\x1b[32m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    underline = "\x1b[4m"

    LEVEL_COLORS = {
        "DEBUG": white,
        "INFO": white,
        "WARNING": yellow,
        "ERROR": red,
        "CRITICAL": bold_red,
    }

    def __init__(
        self,
        colored_output: bool | None = None,
        formatted_numbers: bool = True,
        fmt: str | None = None,
        datefmt: str | None = None,
        style: Literal["%", "{", "$"] = "%",
        validate: bool = True,
        *,
        defaults: Mapping[str, Any] | None = None,
    ) -> None:
        """Create a CustomFormatter.

        Args:
            colored_output: If True/False forces colored output on/off. If None, color is
                enabled unless NO_COLOR env var is set.
            formatted_numbers: If True, numeric tokens in messages are underlined.
            fmt: Format string passed to logging.Formatter.
            datefmt: Date format string passed to logging.Formatter.
            style: Format style passed to logging.Formatter.
            validate: Validation flag passed to logging.Formatter.
            defaults: Defaults mapping passed to logging.Formatter.
        """
        # Colored by default (except for NO_COLOR)
        # See https://no-color.org/
        if colored_output is not None:
            self.colored = colored_output
        elif "NO_COLOR" in os.environ:
            self.colored = False
        else:
            self.colored = True
        self.format_numbers = formatted_numbers
        super().__init__(fmt, datefmt, style, validate, defaults=defaults)

    def format(self, record: logging.LogRecord) -> str:
        """Format a LogRecord.

        Performs the following extra steps before delegating to the base formatter:

        - Optionally underline numeric tokens in record.msg for emphasis.
        - Optionally colorize the level name and highlight certain keywords
          (e.g. 'SUCCESS' in green, 'DROP' in red).

        Args:
            record: logging.LogRecord to be formatted.

        Returns:
            The formatted log message string.
        """

        def number_repl(match):
            return self.underline + match.group(0) + self.reset

        msg = str(record.msg)

        if self.format_numbers:
            msg = re.sub(
                r"\b\d+(?:(?:[\.,:+-]|(?:e[+-]))\d+)*%?",
                number_repl,
                msg,
            )

        if self.colored:
            levelname = record.levelname
            record.levelname = f"{self.LEVEL_COLORS[levelname]}{levelname}{self.reset}"

            msg = msg.replace("SUCCESS", f"{self.green}SUCCESS{self.reset}")
            msg = msg.replace("DROP", f"{self.red}DROP{self.reset}")

        record.msg = msg
        return super().format(record)


def configure_logger_level_and_handlers(
    logger: logging.Logger,
    level: int | str = logging.INFO,
    file_path: str | None = None,
    file_mode: str = "a",
    verbose_fmt: bool = False,
    colored_output: bool | None = None,
    formatted_numbers: bool = True,
) -> None:
    """Configure logger level and attach standard handlers.

    This helper:
    - Sets the logger level.
    - Removes existing handlers.
    - Adds a StreamHandler using CustomFormatter.
    - Optionally adds a FileHandler when file_path is provided.

    Args:
        logger: Logger instance to configure.
        level: Logging level (int or string).
        file_path: Optional filesystem path to write logs to.
        file_mode: File mode for the file handler (default 'a').
        verbose_fmt: If True, use a verbose formatter including timestamps and source file.
        colored_output: If True/False forces colored output for the stream handler.
        formatted_numbers: If True, numbers in messages are underlined.
    """
    logger.setLevel(level)

    for handler in logger.handlers:
        logger.removeHandler(handler)

    fmt = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
        if verbose_fmt
        else "%(levelname)s - %(message)s"
    )

    if file_path:
        _file_handler = logging.FileHandler(filename=file_path, mode=file_mode)
        _file_handler.setLevel(level)
        _file_handler.setFormatter(logging.Formatter(fmt=fmt))

        logger.addHandler(_file_handler)

    _stream_handler = logging.StreamHandler()
    _stream_handler.setLevel(level)
    _stream_handler.setFormatter(
        CustomFormatter(
            fmt=fmt, colored_output=colored_output, formatted_numbers=formatted_numbers
        )
    )

    logger.addHandler(_stream_handler)


# Third-party loggers a source's database driver may install, which emit
# per-cursor/per-request bookkeeping at INFO — once per cursor, and therefore
# once per query. Add an entry here when a source pulls in a chatty driver.
NOISY_LOGGERS: Final[dict[str, int]] = {
    "impala": logging.WARNING,
}


def quiet_noisy_loggers(loggers: dict[str, int] = NOISY_LOGGERS) -> None:
    """Raise the level of chatty third-party loggers.

    Args:
        loggers: Mapping of logger name to the minimum level it should emit.
    """
    for name, level in loggers.items():
        logging.getLogger(name).setLevel(level)


def text_indent(
    text: str | Collection[str],
    indent: str | int = 0,
) -> str:
    """Indent a block of text.

    Args:
        text: The input text block or collection of lines to indent.
        indent: String or number of spaces to use for indentation.

    Returns:
        The indented text block.
    """
    prefix = indent if isinstance(indent, str) else " " * indent
    multiline = text if isinstance(text, str) else "\n".join(text)
    return textwrap.indent(text=multiline, prefix=prefix, predicate=lambda line: True)


def text_tree_indent(
    text: str | Collection[str],
) -> str:
    """Indent a block of text with tree-style prefixes.

    Each line is prefixed with '├── ' except the last line, which is prefixed with '└── '.

    Args:
        text: The input text block or collection of lines to indent.

    Returns:
        The tree-indented text block.
    """
    lines = text.splitlines() if isinstance(text, str) else text
    indented_lines = []
    for i, line in enumerate(lines):
        prefix = "└── " if i == len(lines) - 1 else "├── "
        indented_lines.append(f"{prefix}{line}")
    return "\n".join(indented_lines)


def log_collection(
    logger: logging.Logger,
    collection: Collection[T],
    level: int = logging.INFO,
    as_tree: bool = True,
    indent: str | int = 0,
    serialiser: Callable[[T], str] = str,
) -> None:
    """Log elements of a collection in a readable form.

    When `as_tree=True` each element is prefixed with tree-style markers:
      ├── for all but the last item
      └── for the last item

    When `as_tree=False` each item is indented uniformly without tree markers.

    Args:
        logger: Logger to emit the messages.
        collection: Iterable collection of items to log.
        level: Logging level to use for each emitted message.
        as_tree: Whether to render the collection with tree markers.
        indent: String or number of spaces to use for indentation when `as_tree=False`.
        serialiser: Function to convert each item to a string.
    """
    serialised = [serialiser(value) for value in collection]
    indented = (
        text_indent(text=serialised, indent=indent)
        if not as_tree
        else text_tree_indent(serialised)
    )
    for line in indented.splitlines():
        logger.log(level, line)


def log_multiline_string(
    logger: logging.Logger,
    multiline: str,
    level: int = logging.INFO,
    indent: str | int = 0,
) -> None:
    """Log a multiline string as separate lines.

    Splits the input on newline characters and logs each line separately.
    Useful for showing multi-line messages in logs.

    Args:
        logger: Logger to emit the messages.
        multiline: The multi-line string to log.
        level: Logging level to use for each emitted message.
    """
    log_collection(
        logger=logger,
        collection=multiline.splitlines(),
        level=level,
        as_tree=False,
        indent=indent,
    )


def log_dict(
    logger: logging.Logger,
    dictionary: dict[Any, Any],
    level: int = logging.INFO,
    json_indent: str | int = 0,
    indent: str | int = 0,
) -> None:
    """Log a dictionary in pretty-printed JSON format.

    Args:
        logger: Logger to emit the messages.
        dictionary: The dictionary to log.
        level: Logging level to use for each emitted message.
    """
    multiline = json.dumps(dictionary, indent=json_indent, sort_keys=True, default=str)
    log_multiline_string(logger=logger, multiline=multiline, level=level, indent=indent)


def pretty_join(
    items: Iterable[T],
    sep: str = ", ",
    width: int = 120,
    indent: str | int = 0,
    sort: bool = True,
    serialiser: Callable[[T], str] = str,
) -> str:
    """Join an iterable into a pretty-formatted, indented block.

    Args:
        items: Iterable of items to join.
        sep: Separator string to use between items.
        width: Maximum line width for wrapping.
        indent: String or number of spaces to indent the output.
        sort: Whether to sort the input strings before joining.
        serialiser: Function to convert each input item to a string.

    Returns:
        A single string with the joined, wrapped, and indented content.
    """
    iterable = [serialiser(item) for item in items]
    if sort:
        iterable = sorted(iterable)
    return text_indent(
        text=textwrap.fill(sep.join(iterable), width=width), indent=indent
    )
