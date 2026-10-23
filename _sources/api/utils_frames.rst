DataFrame Utilities
===================

Conversion helpers (Polars ↔ Pandas ↔ Stata ↔ TableOne), time-series joins, value cleaning, and aggregation utilities for clinical data frames.

.. currentmodule:: corr_vars.utils.frames

Conversion
----------

.. autofunction:: convert_to_polars_df

.. autofunction:: convert_to_polars_lf

.. autofunction:: convert_to_pandas_df

.. autofunction:: convert_to_stata

.. autofunction:: convert_to_tableone

Time-Series Operations
-----------------------

.. autofunction:: time_difference

.. autofunction:: remove_asof

.. autofunction:: find_nearest

.. autofunction:: interval_bucket_agg

Value Operations
----------------

.. autofunction:: apply_cleaning

.. autofunction:: absolute_and_relative_value_counts

.. autofunction:: unique_sucessive
