"""corr_vars - Build clinical study cohorts from ICU and hospital data sources."""

__version__ = "2.0.0"
__author__ = "Moritz Thiele"

import contextlib
import logging
import os

from collections.abc import Callable
from typing import Final


# Logging setup
class FStringCallableLogMessage:
    def __init__(
        self,
        message: str,
        /,
        *args: object | Callable[[], object],
        **kwargs: object | Callable[[], object],
    ) -> None:
        self.message = message
        self.args = args
        self.kwargs = kwargs

    def __str__(self) -> str:
        args = (i() if callable(i) else i for i in self.args)
        kwargs = {k: v() if callable(v) else v for k, v in self.kwargs.items()}

        return self.message.format(*args, **kwargs)


__ = FStringCallableLogMessage


def setup_logging(logger: logging.Logger) -> None:
    from .utils.logging import configure_logger_level_and_handlers, quiet_noisy_loggers

    logging.captureWarnings(capture=True)
    configure_logger_level_and_handlers(logger)
    quiet_noisy_loggers()


logger = logging.getLogger(__name__)
setup_logging(logger)


# Temporary directory setup
TMP_DIR: Final[str] = os.environ.get("CORR_VARS_TMP_DIR", "")


def setup_tempdir() -> None:
    import tempfile

    tempfile.tempdir = TMP_DIR
    os.environ["TMPDIR"] = TMP_DIR
    # NOTE: STATA can also read from STATATMP in addition to TMPDIR
    os.environ["STATATMP"] = TMP_DIR


if TMP_DIR and os.path.isdir(TMP_DIR):
    setup_tempdir()

# STATA setup
STATA_DIR: Final[str] = os.environ.get("CORR_VARS_STATA_DIR", "")


def setup_stata() -> None:
    import atexit

    try:
        import stata_setup

        # This will register pystata.config.shutdown() to be called at exit,
        # which can cause a fatal Python error in the C code.
        stata_setup.config(STATA_DIR, "be", splash=False)
        logger.info("STATA is ready to use with Pystata")
    except Exception as e:
        logger.info(f"Stata is not available due to an error:\n {e}")
        return

    # Catch atexit shutdown errors to prevent non-zero exit codes
    # 1. Capture all hidden exit callbacks
    class CallbackCapturer:
        def __init__(self):
            self.captured = []

        def __eq__(self, other):
            # atexit loops through its C-array and compares everything to this object
            self.captured.append(other)
            return False  # Return False so we don't accidentally unregister anything

    capturer = CallbackCapturer()

    atexit.unregister(capturer)

    # 2. Clear the registry completely
    atexit._clear()

    # 3. Re-register everything, replacing the Stata function with a safe wrapper
    for item in reversed(capturer.captured):
        func = item[0] if isinstance(item, tuple) else item
        args = item[1] if isinstance(item, tuple) and len(item) > 1 else ()
        kwargs = item[2] if isinstance(item, tuple) and len(item) > 2 else {}

        # Check if this is the Stata shutdown function
        if hasattr(func, "__name__") and func.__name__ == "shutdown":
            # Define the safe wrapper that swallows the error
            def safe_shutdown(*s_args, **s_kwargs):
                # Silently catch the error so that a script exits with status 0
                with contextlib.suppress(BaseException):
                    func(*s_args, **s_kwargs)

            # Register our wrapper instead of the original
            atexit.register(safe_shutdown, *args, **kwargs)

        else:
            # Re-register all other safe callbacks unmodified
            atexit.register(func, *args, **kwargs)


if STATA_DIR and os.path.isdir(STATA_DIR):
    setup_stata()

# Import commonly used classes and functions
from .core.cohort import Cohort  # noqa: E402
from .sources import local_datasource, reprodicu  # noqa: E402

__all__ = [
    "Cohort",
    ###########
    # Logging #
    ###########
    "logger",
    "__",
    ###########
    # Modules #
    ###########
    # Forgo dynamic import for now.
    # Stub file was not dynamic anyway and was causing
    # Cohort.obs / Cohort.obsm typechecking to not work
    "local_datasource",
    "reprodicu",
]


# Import variables as [source].Variable
# import types
# import pkgutil
# import importlib
# from . import sources
# import sys

# package_name = __name__

# for finder, name, ispkg in pkgutil.iter_modules(sources.__path__):
#     try:
#         extract_module = importlib.import_module(
#             f".sources.{name}.extract", package=package_name
#         )
#         Variable = getattr(extract_module, "Variable", None)
#         if Variable is not None:
#             # Create a submodule
#             mod = types.ModuleType(name)
#             mod.Variable = Variable
#             # Attach submodule to current package namespace
#             globals()[name] = mod
#             __all__.append(name)
#             # Register in sys.modules for import
#             sys.modules[f"{package_name}.{name}"] = mod
#     except (ImportError, AttributeError):
#         continue
