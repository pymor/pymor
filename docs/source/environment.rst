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
