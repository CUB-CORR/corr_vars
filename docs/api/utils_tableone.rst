TableOne Export
===============

Export a `TableOne <https://tableone.readthedocs.io/en/latest/>`_ object (produced by :meth:`~corr_vars.core.cohort.Cohort.to_tableone`) to various formats. Each function accepts an optional ``file_path`` — when omitted the formatted string is returned instead of written to disk.

.. currentmodule:: corr_vars.utils.tableone

.. autofunction:: tableone_to_latex

.. autofunction:: tableone_to_markdown

.. autofunction:: tableone_to_html

.. autofunction:: tableone_to_pdf
