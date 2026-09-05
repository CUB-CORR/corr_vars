"""Configuration schema and defaults for the ``local_datasource`` source.

``local_datasource`` is a **source skeleton**: it carries the variable classes
and the extraction pipeline, but not the query that reaches a backing store (see
:class:`~corr_vars.sources.local_datasource.extract.NativeExtractor`). Its
configuration is correspondingly small — a deployment that binds the source to a
real store extends `SourceDict` and `DEFAULTS` with whatever that store needs.
"""

from typing import TypedDict


class ConnDictPartial(TypedDict, total=False):
    hostname: str | None
    username: str | None
    password: str | None
    password_file: str | bool


class SourceDictPartial(TypedDict, total=False):
    database: str | None
    conn_args: ConnDictPartial


class ConnDict(TypedDict):
    hostname: str | None
    username: str | None
    password: str | None
    password_file: str | bool


class SourceDict(TypedDict):
    database: str | None
    conn_args: ConnDict


DEFAULTS: SourceDict = {
    "database": None,
    "conn_args": {
        "hostname": None,
        "username": None,
        "password": None,
        "password_file": False,
    },
}

#: Legacy top-level keys, relocated to their canonical nested position by
#: ``config_loader.load_default_config_data()``.
MIGRATIONS = {
    "password_file": ("conn_args", "password_file"),
}
