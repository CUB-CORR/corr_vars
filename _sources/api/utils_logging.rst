Logging
=======

Logger configuration and structured log helpers used throughout the library. ``CustomFormatter`` adds colour and aligned level labels; ``configure_logger_level_and_handlers`` is called by ``Cohort.__init__`` to apply the ``logger_args`` parameter.

.. currentmodule:: corr_vars.utils.logging

.. autoclass:: CustomFormatter
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: configure_logger_level_and_handlers

.. autofunction:: text_indent

.. autofunction:: text_tree_indent

.. autofunction:: log_collection

.. autofunction:: log_multiline_string

.. autofunction:: log_dict

.. autofunction:: pretty_join
