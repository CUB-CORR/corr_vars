# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# sys.path.insert(0, os.path.abspath('../corr_vars'))
sys.path.insert(0, os.path.abspath(".."))


# Single source of truth: src/corr_vars/__init__.py, also read by hatch
# (see [tool.hatch.version] in pyproject.toml).
from corr_vars import __version__ as release

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "CORR-Variables"
authors = ["Moritz Thiele", "Noel Kronenberg", "Dario von Wedel"]

if len(authors) > 1:
    author = ", ".join(authors[:-1]) + ", and " + authors[-1]
else:
    author = authors[0]

copyright = f"2025, {author}"
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Try to import optional extensions
optional_extensions = []
try:
    import sphinx_copybutton  # noqa: F401

    optional_extensions.append("sphinx_copybutton")
except ImportError:
    print(
        "Warning: sphinx-copybutton not available. Install with: pip install sphinx-copybutton"
    )

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Google docstrings
    "sphinx.ext.viewcode",  # adds links to highlighted source code
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_design",  # Modern responsive design components
] + optional_extensions

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'sphinx_rtd_theme'
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
autodoc_member_order = "bysource"

html_favicon = "_static/corr_favicon.png"
html_css_files = ["custom.css"]
html_js_files = ["copy-button-enhancement.js"]

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"
copybutton_here_doc_delimiter = "EOF"
copybutton_selector = "div.highlight pre"
copybutton_format_func = None

# Theme options for copy button
html_theme_options = {
    "use_repository_button": True,
    "repository_url": "https://github.com/cub-corr/corr-vars",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_download_button": True,
}


# Configure typehints extension
# autodoc_typehints = "description"  # Show type hints in the description
# autodoc_typehints_format = "short"  # Use short form (e.g., list instead of typing.List)

suppress_warnings = [
    "ref.python",  # Suppresses unresolvable forward references (e.g. Decimal inside polars IntoExpr)
    # Newer sphinx_autodoc_typehints emits its own category for the same
    # unresolvable polars type-alias forward refs (Decimal, Selector, ...).
    "sphinx_autodoc_typehints.forward_reference",
]

# Key settings for automatic default value extraction
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
napoleon_use_param = True
typehints_defaults_print_param_attrs = True
always_document_param_types = True
