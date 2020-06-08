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

PYMOR_WITH_SPHINX
    This variable is set to `1` during API documentation generation
    using sphinx.

PYMOR_DEFAULTS
    If empty or ``NONE``, do not load any :mod:`~pymor.core.defaults`
    from file. Otherwise, a ``:``-separated list of the paths to a
    Python scripts containing defaults.

PYMOR_HYPOTHESIS_PROFILE
    Controls which profile the hypothesis pytest plugin uses to execute our
    test suites. Defaults to the "dev" profile which runs fewer variations than 
    the "ci" or "ci_pr" which get used in our Gitlab-CI.
