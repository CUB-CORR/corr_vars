"""``py`` transformation functions for this source's bundled variables.

One module-level function per variable, named exactly like its key in
``vars.json`` — ``_collect_local_variable_config()`` looks a function up by that
name and injects it into the config as ``"py"``. A variable with no
transformation needs no entry here.

The module is empty because ``vars.json`` declares no variables. See
``AGENTS.md`` for the configuration schema, and note that a variable published
through the CORR Concepts API carries its ``py`` source with it instead — this
file is only for the definitions a source bundles itself.
"""
