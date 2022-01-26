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

PYMOR_MPI_FINALIZE
    If set controls the value for `mpi4py.rc.finalize`. If `PYMOR_MPI_FINALIZE` is unset the value
    of `mpi4py.rc.finalize` remains unchanged, unless `mpi4py.rc.finalize is None` in which
    case it is defaulted to `True`.

PYMOR_ALLOW_DEADLINE_EXCESS
    If set, test functions decorated with :func:`~pymortests.base.might_exceed_deadline` are allowed
    to exceed the default test deadline set in :mod:`~pymortests.conftest`.
