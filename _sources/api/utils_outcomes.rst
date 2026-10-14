Free-Day Outcomes
=================

Source-agnostic "free-days-through-day-N" outcomes — ventilator-free days,
RRT-free days, vasopressor-free days, and so on. :func:`free_days` counts the days in a
follow-up horizon free of a clinical event and, optionally, emits the competing-risk
components the VFD literature recommends reporting alongside the composite.

The :class:`~corr_vars.core.variable.FreeDaysVariable` variable type wraps this engine
so an outcome is declared entirely in ``vars.json`` (``"type": "free_days"``) with a
single episode dependency, and :meth:`~corr_vars.core.cohort.Cohort.add_horizon_variable`
adds one over a ``[t0, t0 + horizon]`` window.

.. currentmodule:: corr_vars.utils.outcomes

.. autofunction:: free_days

Type Aliases
------------

.. autodata:: FreeDaysMethod

.. autodata:: DeathRule

.. autodata:: EventType
