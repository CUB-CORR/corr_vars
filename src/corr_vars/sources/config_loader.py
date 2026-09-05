import logging
from copy import deepcopy

import corr_vars.sources as corr
from corr_vars import logger, utils
from corr_vars.definitions import SourceDict, SourceDictPartial

from collections.abc import Mapping, MutableMapping
from typing import Any, TypeVar, cast

T = TypeVar("T", bound=MutableMapping[str, Any])


def _find_unknown_key_paths(
    user_cfg: Mapping[str, Any],
    schema: Mapping[str, Any],
    prefix: str = "",
    unknown_key_paths: list[str] | None = None,
) -> list[str]:
    if unknown_key_paths is None:
        unknown_key_paths = []

    for key, value in user_cfg.items():
        path = f"{prefix}.{key}" if prefix else key

        if key not in schema:
            unknown_key_paths.append(path)
            continue

        if isinstance(value, Mapping) and isinstance(schema.get(key), Mapping):
            _find_unknown_key_paths(value, schema[key], path, unknown_key_paths)

    return unknown_key_paths


def _load_source_defaults(
    source_kwargs: Mapping[str, Any],
    defaults: T,
    migrations: Mapping[str, Any] | None = None,
) -> T:
    # 1. Load source config
    config = dict(source_kwargs)

    # 2. Apply migrations for old keys
    if migrations:
        for old_key, (new_parent, new_key) in migrations.items():
            if old_key in config:
                config.setdefault(new_parent, {})
                config[new_parent][new_key] = config.pop(old_key)

    # 3. Log warnings
    unknown_key_paths = _find_unknown_key_paths(config, defaults)
    if unknown_key_paths:
        logger.warning("Unknown keys in config:")
        utils.log_collection(
            logger=logger, collection=unknown_key_paths, level=logging.WARNING
        )

    # 4. Deep merge overrides on defaults
    result = utils.deep_merge(deepcopy(defaults), config)

    # 5. Now `result` conforms to SourceDictFull — tell the type checker
    return cast("T", result)


def load_default_config_data(sources: SourceDictPartial) -> SourceDict:
    """Load default config data from multiple sources.

    Args:
        sources (SourceDict): Dictionary of data sources with conigurations to use for data extraction.

    Returns:
        default_sources (SourceDictFull): sources filled with default value where it was left unspecified.
    """
    config = {}
    for source, source_kwargs in sources.items():
        src_config = corr.get_src_module(source=source, submodule="config")
        defaults = cast("MutableMapping[str, Any]", getattr(src_config, "DEFAULTS", {}))
        migrations = cast("Mapping[str, Any]", getattr(src_config, "MIGRATIONS", {}))
        result = _load_source_defaults(
            cast("Mapping[str, Any]", source_kwargs), defaults, migrations
        )
        config[source] = result
    return cast("SourceDict", config)
