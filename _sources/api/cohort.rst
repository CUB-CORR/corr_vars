Cohort
======


Cohort Workflow
----------------

Initialization
^^^^^^^^^^^^^^^

With this snippet, you can initialize a cohort object using the reprodICU data source.

.. code-block:: python

   cohort = Cohort(
       obs_level="icu_stay",     # One of: "patient", "hospital_stay", "icu_stay", "procedure"
       load_default_vars=False,  # Optional, defaults to True
       sources={
           "reprodicu": {"path": "/path/to/reprodICU_files"},
       },
       project="my_project",     # Project on the CORR Concepts API
   )

Specify multiple data sources to combine cohorts.

.. code-block:: python

   cohort = Cohort(
       obs_level="icu_stay",     # One of: "patient", "hospital_stay", "icu_stay", "procedure"
       load_default_vars=False,  # Optional, defaults to True
       sources={
           "reprodicu": {"path": "/path/to/reprodICU_files"},
           # "my_source": {...},
       },
       project="my_project",
   )

Accessing Data
^^^^^^^^^^^^^^

After initialization, cohort data is available through two attributes:

- ``cohort.obs`` — a Polars DataFrame with one row per observation (static variables, demographics, outcomes).
- ``cohort.obsm`` — a dictionary of Polars DataFrames, one per dynamic (time-series) variable.

.. code-block:: python

   # Inspect static data
   print(cohort.obs)
   cohort.obs.select("age_on_admission")
   cohort.obs.filter(pl.col("sex") == "M")

   # Access a dynamic variable
   print(cohort.obsm["blood_sodium"])

   # Filter time-series data for a specific observation
   cohort.obsm["blood_sodium"].filter(
       pl.col(cohort.primary_key) == "12345"
   )

   # Check which dynamic variables have been extracted
   print(list(cohort.obsm.keys()))

You can assign a new DataFrame back to ``cohort.obs`` to add computed columns:

.. code-block:: python

   cohort.obs = cohort.obs.with_columns([
       pl.col('first_sodium_recordtime').eq(pl.col('first_severe_hypernatremia_recordtime'))
       .alias('idx_hypernatremia_was_on_admission')
   ])

   cohort.obs = cohort.obs.with_columns([
       pl.when(pl.col('idx_hypernatremia_was_on_admission'))
       .then(pl.lit('community_acquired'))
       .otherwise(pl.lit('hospital_acquired'))
       .alias('hn_origin')
   ])

Adding Variables
^^^^^^^^^^^^^^^^

.. code-block:: python

   # Use pre-defined variables
   cohort.add_variable("pf_ratio")

   # Load a variable with custom time bounds
   cohort.add_variable(
       variable="anx_dx_covid_19",
       tmin=("hospital_admission", "-1d"),
       tmax=cohort.t_eligible
   )

   # Create a custom variable on the fly
   from corr_vars.sources.local_datasource import NativeStatic

   cohort.add_variable(NativeStatic(
       var_name="median_sodium_before_hn",
       select="!median value",
       base_var="blood_sodium",
       tmin="hospital_admission",
       tmax=cohort.t_eligible,
   ))

   # Save a variable under a different name
   cohort.add_variable(
       variable="any_med_glu",
       save_as="glucose_prior_eligible",
       tmin=(cohort.t_eligible, "-48h"),
       tmax=cohort.t_eligible,
   )

Horizon outcomes (free-day counts, horizon mortality, readmissions) are added with
:meth:`~corr_vars.core.cohort.Cohort.add_horizon_variable`, which derives the
``[t0, t0 + horizon days]`` window from a time-zero column and a horizon in days and
delegates to ``add_variable`` — call it once per (variable, horizon):

.. code-block:: python

   # Ventilator-free days through day 28, saved as vent_free_days_28d
   cohort.add_horizon_variable("vent_free_days", t0="icu_admission", horizon=28)

   # Same template at 90 days; also ICU- and hospital-free days, mortality, readmission
   cohort.add_horizon_variable("vent_free_days", t0="icu_admission", horizon=90)
   cohort.add_horizon_variable("icu_free_days", t0="icu_admission", horizon=28)
   cohort.add_horizon_variable("mortality", save_as="mort_90d", t0="icu_admission", horizon=90)

The underlying ``free_days`` outcomes are declarative ``"type": "free_days"`` variables
(see :doc:`utils_outcomes`).

For large cohorts it is faster to apply filters before loading default variables:

.. code-block:: python

   cohort = Cohort(obs_level="icu_stay", load_default_vars=False, ...)

   # Filter first ...
   cohort.include_list([
       {"variable": "age_on_admission", "operation": ">= 18", "label": "Adults"}
   ])

   # ... then load default variables on the reduced cohort
   cohort.load_default_vars()

Pass ``project_vars`` at construction time to register definitions before any
variables are loaded:

.. code-block:: python

   cohort = Cohort(
       obs_level="icu_stay",
       sources={"reprodicu": {"path": "/path/to/reprodICU_files"}},
       project_vars={
           "my_new_var": {
               "type": "native_dynamic",
               "table": "labs",
               "where": "name LIKE '%new%'",
               "value_dtype": "DOUBLE",
               "cleaning": {"value": {"low": 100, "high": 150}},
           },
           "blood_sodium": {
               "where": "name LIKE '%custom_sodium%'"
           },
       },
   )

To add or override a variable definition at runtime, without touching the
published concept, use ``add_variable_definition()``. Use ``get_variable_definition()`` to inspect the
active definition (merged local override + global defaults) per source:

.. code-block:: python

   # Register a brand-new variable
   cohort.add_variable_definition("my_new_var", {
       "type": "native_dynamic",
       "table": "labs",
       "where": "name LIKE '%new%'",
       "value_dtype": "DOUBLE",
       "cleaning": {"value": {"low": 100, "high": 150}},
   })

   # Partially override an existing variable (merges with global definition)
   cohort.add_variable_definition("blood_sodium", {
       "where": "name LIKE '%custom_sodium%'"
   })

   # Inspect the resolved definition per data source
   defn = cohort.get_variable_definition("blood_sodium")
   # {"reprodicu": {"where": "name LIKE '%custom_sodium%'"}}

The definition is dispatched to the variable loader of every source configured on the
cohort, so an aggregation type (``native_static``, ``derived_static``, ``derived_dynamic``)
in a declarative definition only works if one of those sources has a loader that knows it —
``local_datasource`` or a deployment source built on it. A ``reprodicu``-only cohort does not,
so use direct construction there; see :doc:`aggregation`.

Time Anchors
^^^^^^^^^^^^

``t_eligible`` marks the earliest timepoint a patient is eligible for the study;
``t_outcome`` marks the primary outcome timepoint. Both default to observation-level
column names (e.g. ``icu_admission`` / ``icu_discharge``) but should be overridden
for most study designs.

.. code-block:: python

   # Extract the first SpO2 < 90 % event as the eligibility anchor
   from corr_vars.sources.local_datasource import NativeStatic

   cohort.add_variable(NativeStatic(
       var_name="spo2_lt_90",
       base_var="spo2",
       select="!first recordtime",
       where="value < 90",
   ))
   cohort.set_t_eligible("spo2_lt_90")    # also drops rows where the anchor is null

   # Set the outcome anchor (no rows are dropped)
   cohort.set_t_outcome("hospital_discharge")

   # t_eligible / t_outcome can now be used as tmax/tmin elsewhere
   cohort.add_variable("blood_sodium", tmax=cohort.t_eligible)

Inclusion/Exclusion
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Add multiple inclusion criteria at once
   cohort.include_list([
       {
           "variable": "age_on_admission",
           "operation": ">= 18",
           "label": "Adult patients"
       },
       {
           "variable": "icu_length_of_stay",
           "operation": "> 2",
           "label": "ICU stay > 2 days"
       }
   ])

   # Add exclusion criteria at once
   cohort.exclude_list([
       {
           "variable": "any_dx_covid_19",
           "operation": "== True",
           "label": "Exclude COVID-19 patients"
       }
   ])

Use ``include()`` / ``exclude()`` to apply a single criterion at a later stage:

.. code-block:: python

   cohort.include(
       variable="age_on_admission",
       operation=">= 18",
       label="Adult",
       operations_done="Include only adult patients",
   )

   cohort.exclude(
       variable="elix_total",
       operation="> 20",
       operations_done="Exclude patients with high Elixhauser score",
   )

For grouped criteria that should appear as a single step in the flowchart, use the
``change_tracker()`` context manager:

.. code-block:: python

   with cohort.change_tracker("Adults", mode="include") as track:
       track.filter(pl.col("age_on_admission") >= 18)

Exploration
^^^^^^^^^^^

.. code-block:: python

   # Create a TableOne summary
   tableone = cohort.to_tableone(ignore_cols=["icu_id"])
   print(tableone)

   # Grouped TableOne (e.g. by sex)
   tableone = cohort.to_tableone(groupby="sex", pval=True)
   tableone.to_csv("tableone_sex.csv")

   # Interactive Jupyter widget for obs
   cohort.widget                          # renders the full obs DataFrame
   cohort.to_widget("age_on_admission", "sex")  # select specific columns

   # Interactive widget for dynamic variables
   cohort.obsm.widget                     # renders all obsm variables
   cohort.obsm.to_widget("blood_sodium")  # renders a single variable

   # Inclusion / exclusion flowchart
   cohort.figureone                       # returns a graphviz.Digraph object
   cohort.to_figureone()                  # returns a graphviz.Digraph object

   # Print debug information (helpful when filing a GitHub issue)
   cohort.debug_print()

Data Export
^^^^^^^^^^^

.. code-block:: python

   # Save to CORR archive (recommended)
   cohort.save("my_cohort.corr3")

   # Load from file (.corr2 and .corr3 supported)
   cohort = Cohort.load("my_cohort.corr3")

   # Export obs + all obsm variables as individual CSV files
   cohort.to_csv("path/to/output_folder")

   # Export obs + all obsm variables as individual Parquet files
   cohort.to_parquet("path/to/output_folder")

   # Convert obs to a Stata-compatible pandas DataFrame
   stata_df = cohort.to_stata()

   # Save obs directly as a .dta Stata file
   cohort.to_stata(to_file="path/to/output_folder/my_cohort.dta")

Class Reference
----------------

.. currentmodule:: corr_vars.core.cohort

.. autoclass:: Cohort
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: obs, obsm
