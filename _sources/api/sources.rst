Data Sources
============

CORR-Vars supports multiple data sources for extracting clinical data. Each source provides specialized access to different types of healthcare data.

Available Sources
-----------------

.. toctree::
   :maxdepth: 2

   reprodicu
   local_datasource
   aggregation

Overview
--------

Data sources in CORR-Vars are modular components that handle the extraction and preprocessing of clinical data from different healthcare systems. Each source implements a standardized interface while providing source-specific optimizations and features.

**ReprodICU**
  Parquet-based source for the reprodICU collection of intensive care datasets. It is the
  ready-to-use example source: point ``path`` at the files and it works.

**Local data source**
  The source skeleton. It carries the variable classes and the extraction pipeline but no
  connection to a store, so a deployment binds it to its own — see
  :doc:`local_datasource`. Its aggregation and derivation classes need no binding and work
  on any cohort (:doc:`aggregation`).

A source is a package under ``src/corr_vars/sources/``, discovered automatically.
See :doc:`../index` for what a source has to provide.

Multi-Source Configuration
--------------------------

You can combine multiple data sources in a single cohort:

.. code-block:: python

    from corr_vars import Cohort

    # Configure multiple sources
    cohort = Cohort(
        obs_level="icu_stay",
        sources={
            "reprodicu": {"path": "/path/to/reprodICU_files"},
            # "my_source": {...},
        },
        project="my_project",
    )

    # Data source tracking
    print(cohort.obs["data_source"].value_counts())

Source-Specific Variables
-------------------------

Different sources may provide different variables. The variable loader automatically handles source selection:

.. code-block:: python

    # A variable one source publishes and another does not
    cohort.add_variable("apache_ii_score")

    # Variables are automatically tagged with their source
    print("Available sources for this variable:")
    print(cohort.obsm["blood_pressure"]["data_source"].unique())

Variable Loader
---------------

The variable loader provides the core functionality for loading variables from configured sources:

.. currentmodule:: corr_vars.sources.var_loader

.. autofunction:: load_variable

Base Sources Module
-------------------

.. currentmodule:: corr_vars.sources

.. automodule:: corr_vars.sources
   :members:
   :undoc-members:
   :show-inheritance: