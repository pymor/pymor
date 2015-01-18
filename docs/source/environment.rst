.. _environment:

*********************
Environment Variables
*********************

pyMOR respects the following environment variables:

PYMOR_CACHE_DISABLE
    If ``1``, disable caching globally, overriding calls to
    :func:`~pymor.core.cache.enable_caching`. This is mainly
    useful for debugging. See :mod:`pymor.core.cache` for more
    details. 

PYMOR_CACHE_MAX_SIZE
    Maximum size of pyMOR's default disk |CacheRegion|. If not set,
    the `pymor.core.cache._setup_default_regions.disk_max_size`
    |default| is used. The suffixes 'k' or 'K', 'm' or 'M' and
    'g' or 'G' can be used to specify the amount as a number of
    kilobytes, megabytes or gigabytes.

PYMOR_CACHE_MEMORY_MAX_KEYS
    Maximum number of keys stored in pyMOR's default memory
    |CacheRegion|. If not set, the
    `pymor.core.cache._setup_default_regions.memory_max_keys`
    |default| is used.

PYMOR_CACHE_PATH
    Location of pyMOR's default disk |CacheRegion|. If not
    set, the `pymor.core.cache._setup_default_regions.disk_path`
    |default| is used.

PYMOR_COLORS_DISABLE
    If ``1``, disable coloring of logging output.

PYMOR_COPY_DOCSTRINGS_DISABLE 
    By default, docstrings of methods in base classes are copied
    to overriding methods, if these do not define their own
    docstring. Setting this variable to `1` disables this feature.
    (We use this for when auto-generating API-documentation with
    sphinx.)

PYMOR_DEFAULTS
    If empty or ``NONE``, do not load any :mod:`~pymor.core.defaults`
    from file. Otherwise, a ``:``-separated list of the paths to a
    Python scripts containing defaults.
