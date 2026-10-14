Aggregated Variables
====================

Aggregation variables create new variables by computing statistics or transformations
from variables that are already in the cohort. They never touch a backing store, so they
work on **any** cohort, whatever source supplies the underlying data.

The classes live in :mod:`corr_vars.sources.local_datasource` and are imported from there:

.. code-block:: python

    from corr_vars.sources.local_datasource import (
        DerivedDynamic,
        DerivedStatic,
        NativeStatic,
    )

Overview
--------

- **NativeStatic**: one value per observation, aggregated from one dynamic ``base_var``
  (e.g. first, last, mean, max)
- **DerivedStatic**: one value per observation, computed from existing columns via an
  SQL-like ``expression`` or a ``py`` function
- **DerivedDynamic**: a new time series computed from the variables in ``requires`` via a
  ``py`` function

Variable Types
--------------

NativeStatic Variables
^^^^^^^^^^^^^^^^^^^^^^

NativeStatic collapses a dynamic variable into a single value per observation.

**Available Aggregation Functions:**

- ``!first [columns]``: First recorded row
- ``!last [columns]``: Last recorded row
- ``!any``: True if any value exists
- ``!sum [column]``: Sum of values
- ``!mean [column]``: Mean value
- ``!median [column]``: Median value
- ``!perc(quantile) [column]``: Percentile, e.g. ``!perc(75) value``
- ``!closest(to_column[, timedelta[, plusminus]]) [columns]``: Value closest to a
  reference time

.. note::

   The class docstring lists the functions above, but the implementation supports more.
   ``!min``, ``!max``, ``!count``, ``!std`` and ``!sum_interval`` are also available:

   - ``!min [column]`` / ``!max [column]``: Minimum / maximum value
   - ``!count [column]``: Number of rows in the window
   - ``!std [column]``: Standard deviation
   - ``!sum_interval [column]``: Sum of interval values weighted by the overlap of each
     interval with the time window. Requires a ``recordtime_end`` column on the base
     variable.

**Examples:**

.. code-block:: python

    from corr_vars import Cohort
    from corr_vars.sources.local_datasource import NativeStatic

    cohort = Cohort(
        obs_level="icu_stay",
        sources={"reprodicu": {"path": "/path/to/reprodICU_files"}},
        project="my_project",
    )

    # First blood pressure measurement
    cohort.add_variable(NativeStatic(
        var_name="first_blood_pressure",
        select="!first value",
        base_var="blood_pressure_sys",
    ))

    # Maximum heart rate during the ICU stay
    cohort.add_variable(NativeStatic(
        var_name="max_heart_rate",
        select="!max value",
        base_var="heart_rate",
        tmin="icu_admission",
        tmax="icu_discharge",
    ))

    # Blood pressure closest to admission, within two hours
    cohort.add_variable(NativeStatic(
        var_name="admission_blood_pressure",
        select="!closest(icu_admission, 0, 2h) value",
        base_var="blood_pressure_sys",
    ))

``base_var`` names an existing dynamic variable. It is resolved through the cohort's own
sources, so ``blood_pressure_sys`` above comes from ``reprodicu``. Pass ``tmin``/``tmax``
inside the constructor — passing them to ``add_variable()`` for an object emits a
``UserWarning``.

DerivedStatic Variables
^^^^^^^^^^^^^^^^^^^^^^^

DerivedStatic computes one value per observation from columns that are already on the
observation frame, either with an ``expression`` (parsed by ``polars.sql_expr``) or with a
``py`` function.

**Expression-Based Variables:**

.. code-block:: python

    from corr_vars.sources.local_datasource import DerivedStatic

    # Body mass index
    cohort.add_variable(DerivedStatic(
        var_name="bmi",
        requires=["weight_on_admission", "height"],
        expression="weight_on_admission / (height / 100) ** 2",
    ))

    # Threshold flag on an aggregated value
    cohort.add_variable(DerivedStatic(
        var_name="hypernatremia",
        requires=["max_sodium_icu"],
        expression="max_sodium_icu > 145",
    ))

**Custom Function Variables:**

For anything an expression cannot express, pass a ``py`` function. It receives the
variable and the cohort, and returns a frame keyed on the cohort's primary key:

.. code-block:: python

    import polars as pl

    def lactate_clearance(var, cohort):
        data = var.required_vars["blood_lactate"].data
        agg = data.group_by("icu_stay_id").agg(
            pl.col("value").first().alias("first_lactate"),
            pl.col("value").last().alias("last_lactate"),
        )
        return agg.select(
            "icu_stay_id",
            ((pl.col("first_lactate") - pl.col("last_lactate")) / pl.col("first_lactate"))
            .alias("lactate_clearance"),
        )

    cohort.add_variable(DerivedStatic(
        var_name="lactate_clearance",
        requires=["blood_lactate"],
        py=lactate_clearance,
        py_ready_polars=True,
    ))

DerivedDynamic Variables
^^^^^^^^^^^^^^^^^^^^^^^^

DerivedDynamic builds a new time series from the variables in ``requires``. The ``py``
function returns rows carrying the primary key, ``recordtime`` and ``value``:

.. code-block:: python

    import polars as pl

    from corr_vars.sources.local_datasource import DerivedDynamic

    def pf_ratio(var, cohort):
        pao2 = var.required_vars["blood_pao2_arterial"].data
        fio2 = var.required_vars["vent_fio2"].data
        joined = pao2.join_asof(
            fio2.rename({"value": "fio2"}),
            on="recordtime",
            by="icu_stay_id",
            tolerance="1h",
        )
        return joined.select(
            "icu_stay_id",
            "recordtime",
            (pl.col("value") / pl.col("fio2")).alias("value"),
        )

    cohort.add_variable(DerivedDynamic(
        var_name="pf_ratio",
        requires=["blood_pao2_arterial", "vent_fio2"],
        py=pf_ratio,
        py_ready_polars=True,
        cleaning={"value": {"low": 50, "high": 800}},
    ))

Declarative form
----------------

The same variables can be written as declarative definitions instead of objects. This is
the shape a source's bundled ``vars.json`` and the CORR Concepts API use, and
``Cohort.add_variable_definition()`` accepts it for a single project:

.. code-block:: python

    cohort.add_variable_definition("max_sodium_icu", {
        "type": "native_static",
        "select": "!max value",
        "base_var": "blood_sodium",   # NOT "requires"
    })
    cohort.add_variable("max_sodium_icu", tmin="icu_admission", tmax="icu_discharge")

For ``native_static`` the field naming the aggregated variable is ``base_var`` (the legacy
alias ``variable`` is still accepted), never ``requires``.

.. warning::

   A project-local definition is dispatched to the ``VariableLoader`` of **every** source
   configured on the cohort, and only a loader that knows the type can build it.
   :func:`~corr_vars.sources.local_datasource.extract.VariableLoader` knows
   ``native_static``, ``derived_static``, ``derived_dynamic``, ``native_dynamic``,
   ``complex`` and ``free_days``; the ``reprodicu`` loader does not — it raises
   ``TypeError`` on ``select`` / ``base_var`` / ``expression``.

   So on a reprodICU-only cohort, use the declarative form for **overrides of served
   definitions** (``cleaning``, ``filter``, …) and for reprodICU's own fields, and use
   direct construction for the aggregation types. The declarative form for aggregation
   types is for deployments that bound ``local_datasource`` (see
   :doc:`local_datasource`) and for authoring definitions in ``vars.json`` or the
   Concepts API.

Time Constraints and Filtering
------------------------------

All aggregation variables support time constraints. ``tmin``/``tmax`` accept a column name
or a ``(column, offset)`` tuple:

.. code-block:: python

    # Values only from the first 24 hours
    early_lactate = NativeStatic(
        var_name="max_lactate_24h",
        select="!max value",
        base_var="blood_lactate",
        tmin="icu_admission",
        tmax=("icu_admission", "+24h"),
    )

    # Values before a specific event
    pre_intubation_spo2 = NativeStatic(
        var_name="last_spo2_before_intubation",
        select="!last value",
        base_var="spo2",
        tmax="first_intubation_dtime",
    )

Filtering with WHERE Clauses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``where`` filters the base variable's rows before aggregation. It takes an SQL-style
boolean expression, or one of ``!isin`` / ``!startswith`` / ``!endswith``:

.. code-block:: python

    # Only febrile values
    high_temp = NativeStatic(
        var_name="max_fever_temperature",
        select="!max value",
        base_var="body_temperature",
        where="value > 38.0",
    )

    # Specific medication labels
    max_norepinephrine = NativeStatic(
        var_name="max_norepinephrine_dose",
        select="!max value",
        base_var="med_norepinephrine",
        where="!isin(description, ['Norepinephrine', 'Noradrenaline'])",
    )

Advanced Examples
-----------------

Complex Clinical Indicators
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Shock index at admission, from two aggregated values
    shock_index = DerivedStatic(
        var_name="shock_index_admission",
        requires=["first_heart_rate", "first_blood_pressure_sys"],
        expression="first_heart_rate / first_blood_pressure_sys",
    )

    # APACHE II acute physiology components
    apache_temp = NativeStatic(
        var_name="apache_temperature",
        select="!closest(icu_admission, 0, 24h) value",
        base_var="body_temperature",
    )

    apache_map = NativeStatic(
        var_name="apache_mean_arterial_pressure",
        select="!closest(icu_admission, 0, 24h) value",
        base_var="blood_pressure_mean",
    )

Outcome Variables
^^^^^^^^^^^^^^^^^

.. code-block:: python

    # ICU length of stay in days
    icu_los = DerivedStatic(
        var_name="icu_length_of_stay_days",
        requires=["icu_admission", "icu_discharge"],
        expression="(icu_discharge - icu_admission).dt.total_seconds() / 86400",
    )

.. note::

   Free-day outcomes (ventilator-, ICU-, RRT-free days, …) are no longer hand-rolled as
   ``DerivedStatic`` variables. They are declarative ``"type": "free_days"`` variables
   backed by :class:`~corr_vars.core.variable.FreeDaysVariable`; add them with
   :meth:`~corr_vars.core.cohort.Cohort.add_horizon_variable`, e.g.
   ``cohort.add_horizon_variable("vent_free_days", t0="icu_admission", horizon=28)``.
   See :doc:`utils_outcomes`.

Quality Indicators
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Number of blood pressure measurements during the stay
    bp_frequency = NativeStatic(
        var_name="bp_measurement_count",
        select="!count value",
        base_var="blood_pressure_sys",
        tmin="icu_admission",
        tmax="icu_discharge",
    )

    # Time of the first antibiotic administration
    time_to_abx = NativeStatic(
        var_name="first_antibiotic_time",
        select="!first recordtime",
        base_var="any_antibiotic_icu",
    )

Best Practices
--------------

1. **Use appropriate aggregation functions**: choose the one that matches the clinical
   question rather than post-processing a broader one.
2. **Set time constraints**: specify ``tmin``/``tmax`` to avoid temporal bias.
3. **Apply cleaning rules**: use ``cleaning`` to drop physiologically impossible values.
4. **Prefer expressions over ``py``**: an ``expression`` is cheaper and easier to review.
5. **Validate results**: check aggregated values for clinical plausibility before use.

Class Reference
---------------

.. currentmodule:: corr_vars.sources.local_datasource.extract

.. autoclass:: NativeStatic
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: DerivedStatic
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: DerivedDynamic
   :members:
   :undoc-members:
   :show-inheritance:
