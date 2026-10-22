"""``charite_demo``: the public, fully synthetic CHARITÉ mock dataset.

A port of the production Charité source from Impala to DuckDB over Parquet. It
reads the published dataset straight from its URL (the default) or from a local
copy made with :func:`download`:

.. code-block:: python

    from corr_vars import Cohort

    cohort = Cohort(obs_level="icu_stay", sources={"charite_demo": {}})
"""

from .config import SourceDict, SourceDictPartial
from .dataset import download
from .extract import (
    BaseDynamic,
    Co6HierarchyDynamic,
    Co6HierarchyExtractor,
    ComplexVariable,
    DerivedDynamic,
    DerivedStatic,
    NativeDynamic,
    NativeExtractor,
    NativeStatic,
    VariableLoader,
)

__all__ = [
    "BaseDynamic",
    "Co6HierarchyDynamic",
    "Co6HierarchyExtractor",
    "ComplexVariable",
    "DerivedDynamic",
    "DerivedStatic",
    "NativeDynamic",
    "NativeExtractor",
    "NativeStatic",
    "SourceDict",
    "SourceDictPartial",
    "VariableLoader",
    "download",
]
