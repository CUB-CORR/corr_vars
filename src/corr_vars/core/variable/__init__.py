"""Variable classes: the source-agnostic base and outcome variable types.

``Variable`` is the base every source's variable classes build on;
``FreeDaysVariable`` is a source-agnostic "free-days-through-day-N" outcome.
"""

from corr_vars.core.variable.base import Variable
from corr_vars.core.variable.free_days import FreeDaysVariable

__all__ = [
    "FreeDaysVariable",
    "Variable",
]
