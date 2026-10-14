Local Data Source (skeleton)
============================

``local_datasource`` is the **source skeleton** that ships with CORR-Vars. It carries the
variable classes and the extraction pipeline, but no connection to any store: the query
that reaches a warehouse table, a folder of parquet files, or an HTTP service is
deployment-specific and is not part of the published package.

Two things follow from that:

- The aggregation and derivation classes (:class:`NativeStatic`, :class:`DerivedStatic`,
  :class:`DerivedDynamic`, :class:`ComplexVariable`) compute from data that is already in
  the cohort. They need **no** store and work on any cohort — see :doc:`aggregation` and
  :doc:`../custom_variables`.
- Only :class:`NativeDynamic`, which fetches new rows, needs a store. That is what binding
  the skeleton is for.

The source is pinned as local under ``[local].sources`` in
``src/corr_vars/concepts/routing.toml``: its definitions are never fetched from the
Concepts API, so it works with no network and no API key. Definitions come from the
bundled ``mapping/vars.json``, and ``py`` functions from ``mapping/variables.py``, looked
up by variable name.

The two seams
-------------

Binding the skeleton to a store means replacing exactly two things.

1. ``cohort.load_data`` — the observation frame
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~corr_vars.sources.local_datasource.cohort.load_data` raises
``NotImplementedError``. Replace it to build the cohort's observation frame: one row per
observation, carrying at least the level's primary key and its ``t_min`` / ``t_max`` anchor
columns (for ``icu_stay``: ``icu_stay_id``, ``icu_admission``, ``icu_discharge``).

2. ``NativeExtractor.extract`` — the raw rows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`NativeExtractor.extract` also raises ``NotImplementedError``. Subclass it,
implement ``extract(cohort, id_column)`` to return rows carrying at least ``id_column``,
``recordtime`` and ``value``, and point :attr:`NativeDynamic.extractor_class` at the
subclass. :class:`NativeDynamic` wraps the extractor in the :class:`BaseDynamic` pipeline:
cache → ``extract_from_db`` → time filter → cleaning → optional ``py`` post-processing.

Worked example: a folder of parquet files
-----------------------------------------

The following binds the skeleton to a directory in which each table is one parquet file,
keyed on ``icu_stay_id``:

.. code-block:: python

    from pathlib import Path

    import polars as pl

    import corr_vars.sources.local_datasource.cohort as ld_cohort
    from corr_vars.sources.local_datasource import NativeDynamic, NativeExtractor

    DATA_DIR = Path("/path/to/my_store")


    class MyExtractor(NativeExtractor):
        """Read one parquet file per table and restrict it to the cohort's ids."""

        def extract(self, cohort, id_column):
            frame = pl.scan_parquet(DATA_DIR / f"{self.table}.parquet")

            # `where` is optional; when set it is an SQL-style boolean on the table.
            if self.where:
                frame = frame.filter(pl.sql_expr(self.where))

            ids = cohort.obs.select(id_column).unique()
            frame = frame.join(ids.lazy(), on=id_column, how="inner")

            frame = frame.select(id_column, "recordtime", "value")
            if self.value_dtype:
                frame = frame.with_columns(
                    pl.col("value").cast(getattr(pl, self.value_dtype))
                )
            return frame.collect()


    NativeDynamic.extractor_class = MyExtractor


    def load_data(obs_level, source_kwargs):
        """Build the observation frame for `obs_level`."""
        stays = pl.read_parquet(DATA_DIR / "icu_stays.parquet")
        return stays.select("icu_stay_id", "icu_admission", "icu_discharge")


    ld_cohort.load_data = load_data

With both seams filled, a ``NativeDynamic`` variable extracts from the store like any
other native variable:

.. code-block:: python

    from corr_vars import Cohort

    cohort = Cohort(
        obs_level="icu_stay",
        sources={"local_datasource": {}},
        project="my_project",
    )

    cohort.add_variable(NativeDynamic(
        var_name="blood_lactate",
        dynamic=True,
        table="labs",
        where="analyte = 'lactate'",
        value_dtype="Float64",
        cleaning={"value": {"low": 0.1, "high": 30.0}},
    ))

Configuration
-------------

:mod:`corr_vars.sources.local_datasource.config` defines the ``SourceDict`` and
``SourceDictPartial`` TypedDicts, plus ``DEFAULTS`` and ``MIGRATIONS``. The published
schema is deliberately small — a ``database`` name and ``conn_args``
(``hostname``, ``username``, ``password``, ``password_file``):

.. code-block:: python

    DEFAULTS: SourceDict = {
        "database": None,
        "conn_args": {
            "hostname": None,
            "username": None,
            "password": None,
            "password_file": False,
        },
    }

A deployment extends these with whatever its store needs. ``MIGRATIONS`` maps legacy
top-level keys to their canonical nested position, so old configuration files keep
loading.

Variable definitions
--------------------

Once the store is bound, the source needs definitions. Either fill the bundled mapping, or
route the source to a Concepts API endpoint in ``concepts/routing.toml`` (which unpins it
from ``[local].sources``).

``mapping/vars.json``
^^^^^^^^^^^^^^^^^^^^^

The bundled file declares no variables. Add them under ``variables``, and list the ones
that should load automatically under ``corr_defaults.default_vars`` per observation level.
The main definition fields are:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Meaning
   * - ``type``
     - ``native_dynamic``, ``native_static``, ``derived_static``, ``derived_dynamic``,
       ``complex`` or ``free_days``. Selects the class the loader builds.
   * - ``table``
     - Table in the backing store (native variables only).
   * - ``where``
     - Row filter for the extraction, in the store's dialect (native variables only).
   * - ``value_dtype``
     - Polars dtype name the ``value`` column is cast to, e.g. ``"Float64"``.
   * - ``cleaning``
     - Per-column plausibility bounds, e.g. ``{"value": {"low": 0, "high": 300}}``.
   * - ``requires`` / ``base_var``
     - Dependencies. Derived and complex variables take a ``requires`` list;
       ``native_static`` takes a single ``base_var``.
   * - ``compatible_with``
     - Observation levels the definition is valid for.

``mapping/variables.py``
^^^^^^^^^^^^^^^^^^^^^^^^

One module-level function per variable that needs a ``py`` transformation, named exactly
like its key in ``vars.json``. The loader looks the function up by that name and injects
it into the config as ``"py"``. A variable with no transformation needs no entry.

.. note::

   A variable published through the CORR Concepts API carries its ``py`` source with it,
   so ``variables.py`` is only for the definitions a source bundles itself.

Class Reference
---------------

.. currentmodule:: corr_vars.sources.local_datasource.extract

.. autoclass:: BaseDynamic
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: NativeExtractor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: NativeDynamic
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ComplexVariable
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: VariableLoader

.. currentmodule:: corr_vars.sources.local_datasource.cohort

.. autofunction:: load_data
