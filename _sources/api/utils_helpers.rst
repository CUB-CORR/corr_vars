Extraction Helpers
==================

Low-level helpers used during variable extraction: filtering by condition, aggregation, time-window expression builders, time-series cleaning, consecutive-stay merging, and miscellaneous utilities.

.. currentmodule:: corr_vars.utils.helpers

Filtering and Aggregation
--------------------------

.. autofunction:: filter_by_condition

.. autofunction:: aggregate_column

.. autofunction:: get_cb

.. autofunction:: extract_df_data

Time Windows
------------


.. autofunction:: add_time_window_expr

Time-Series Cleaning
---------------------

.. autofunction:: clean_timeseries

.. autofunction:: merge_consecutive

Miscellaneous
-------------

.. autofunction:: harmonize_str_list_cols

.. autofunction:: deep_merge

.. autofunction:: flatten_args

.. autofunction:: raise_error_with_nearest_matches
