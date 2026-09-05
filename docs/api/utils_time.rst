Time Primitives
===============

Classes that represent time anchors and time windows used throughout variable extraction. ``TimeAnchor`` wraps a column name and an optional offset; ``TimeWindow`` pairs a ``tmin`` and ``tmax`` anchor into a single object passed through the extraction pipeline.

.. currentmodule:: corr_vars.utils.time

.. autoclass:: TimeAnchor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: TimeWindow
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: WindowRelation
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: WindowOverlap
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: compatible_with_time_anchor

.. autofunction:: compatible_with_time_window

.. autofunction:: resolve_time_anchor

.. autofunction:: truncate_to_day

.. autofunction:: format_delta
