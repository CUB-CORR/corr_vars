Utils
=====

Generally useful functions used across the library. Some are also helpful when writing custom variable definitions or post-processing cohort data.

.. toctree::
   :maxdepth: 1

   utils_time
   utils_base
   utils_frames
   utils_outcomes
   utils_helpers
   utils_tableone
   utils_logging
   utils_debug

Overview
--------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Submodule
     - Contents
   * - :doc:`utils_time`
     - ``TimeAnchor``, ``TimeWindow``, ``WindowRelation``, ``WindowOverlap`` — time-bound primitives used throughout variable extraction.
   * - :doc:`utils_base`
     - Dependency-free primitives: ``as_expr``, ``column_selector``, frame introspection, ``normalize_offset``.
   * - :doc:`utils_frames`
     - DataFrame conversion (Polars ↔ Pandas ↔ Stata), time-series joins, value cleaning, aggregation helpers.
   * - :doc:`utils_outcomes`
     - The source-agnostic ``free_days`` engine for free-day (VFD-style) outcomes.
   * - :doc:`utils_helpers`
     - Low-level variable-extraction helpers: filtering, aggregation, time-window expressions, time-series cleaning, ``merge_consecutive``.
   * - :doc:`utils_tableone`
     - Export a ``TableOne`` object to LaTeX, Markdown, HTML, or PDF.
   * - :doc:`utils_logging`
     - Logger configuration, custom formatters, and structured log helpers.
   * - :doc:`utils_debug`
     - Debug-info printers for filing GitHub issues.
