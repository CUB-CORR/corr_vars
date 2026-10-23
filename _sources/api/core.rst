Core Module
===========

The core module contains the fundamental classes for managing cohorts and variables in CORR-Vars.

Variable Architecture
---------------------

Variables in CORR-Vars follow a hierarchical structure that supports different types of data extraction and computation:

.. image:: ../_static/cv_var_hierarchy.png
   :width: 50%
   :align: center

Variable Types
^^^^^^^^^^^^^^

**Base Variable Class**
  The foundational class that all variables inherit from. Provides common functionality for data processing, cleaning, and time filtering. Source-specific subclasses (e.g. ``Variable`` in ``sources/reprodicu``) add source-specific extraction and optimizations.

  ``corr_vars.core.variable`` is a package: :class:`~corr_vars.core.variable.Variable`
  lives in ``base`` and the source-agnostic outcome type
  :class:`~corr_vars.core.variable.FreeDaysVariable` in ``free_days``; both are
  re-exported from the package.

**Free-Day Outcome Variable**
  :class:`~corr_vars.core.variable.FreeDaysVariable` (``"type": "free_days"``) computes
  "free-days-through-day-N" outcomes (ventilator-, RRT-, vasopressor-free days, …)
  source-agnostically, delegating to the :func:`~corr_vars.utils.outcomes.free_days`
  engine. See :doc:`utils_outcomes`.

Variable Processing Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each variable follows a standardized processing pipeline:

1. **Extraction**: Data is retrieved from the specified source
2. **Time Filtering**: Data is filtered based on tmin/tmax constraints  
3. **Cleaning**: Invalid values are removed based on cleaning rules
4. **Column Ordering**: Columns are standardized according to predefined order
5. **Relative Time Calculation**: Relative timestamps are computed for dynamic variables

Examples
--------

Working with Variables
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from corr_vars.core.variable import Variable
    from corr_vars import Cohort
    
    # Initialize a cohort
    cohort = Cohort(obs_level="icu_stay", load_default_vars=False)
    
    # Variables are typically added through the cohort interface
    cohort.add_variable("blood_sodium")
    
    # Access the variable data
    sodium_data = cohort.obsm["blood_sodium"]
    print(f"Sodium measurements: {len(sodium_data)} records")

Custom Variable Creation
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from corr_vars.sources.local_datasource import NativeStatic

    # Aggregate an existing dynamic variable into one value per observation.
    # The variable classes are source-agnostic: base_var resolves through
    # whichever sources the cohort is configured with.
    cohort.add_variable(NativeStatic(
        var_name="max_temperature_24h",
        select="!max value",
        base_var="body_temperature",
        tmin="icu_admission",
        tmax=("icu_admission", "+24h"),
    ))

**Declarative form**

The same variable can be written as a definition dict, which is the shape used in a
source's ``vars.json`` and in the Concepts API:

.. code-block:: python

    cohort.add_variable_definition("max_temperature_24h", {
        "type": "native_static",
        "select": "!max value",
        "base_var": "body_temperature",
    })
    cohort.add_variable(
        "max_temperature_24h",
        tmin="icu_admission",
        tmax=("icu_admission", "+24h"),
    )

This only works if one of the cohort's sources has a variable loader that understands
the aggregation type — ``local_datasource`` or a deployment source built on it. A
``reprodicu``-only cohort does not, so use direct construction there. See
:doc:`aggregation`.

Time Constraints
^^^^^^^^^^^^^^^^

.. code-block:: python

    # Add variable with custom time constraints
    cohort.add_variable(
        "blood_lactate",
        tmin=("icu_admission", "-2h"),  # 2 hours before ICU admission
        tmax=("icu_admission", "+6h")   # 6 hours after ICU admission
    )

Class Reference
---------------

.. currentmodule:: corr_vars.core.variable

.. autoclass:: Variable
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: FreeDaysVariable
   :members:
   :undoc-members:
   :show-inheritance:

