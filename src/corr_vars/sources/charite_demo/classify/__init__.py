"""Runtime classification helpers shared by served charite_demo ``py`` snippets.

Package code rather than definition code: a snippet served by the Concepts API
finds ``classify_airway_devices`` in the namespace
:mod:`corr_vars.sources.charite_demo.py_env` declares.
"""

from corr_vars.sources.charite_demo.classify.airway import classify_airway_devices

__all__ = ["classify_airway_devices"]
