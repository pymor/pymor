(environment)=

# Environment Variables

pyMOR respects the following environment variables:

```{envvar} PYMOR_CACHE_DISABLE
If `1`, disable caching globally, overriding calls to
{func}`~pymor.core.cache.enable_caching`. This is mainly
useful for debugging. See {mod}`pymor.core.cache` for more
details.
```

```{envvar} PYMOR_COLORS_DISABLE
If `1`, disable coloring of logging output.
```

```{envvar} PYMOR_CONFIG_DISABLE
Whitespace separated list of {mod}`~pymor.core.config` items
which should be reported as missing even though they may be present.
E.g., `PYMOR_CONFIG_DISABLE="SLYCOT"` can be used to prevent pyMOR
from importing the Slycot library.
```

```{envvar} PYMOR_DEFAULTS
If empty or `NONE`, do not load any {mod}`~pymor.core.defaults`
from file. Otherwise, a `:`-separated list of the paths to a
Python scripts containing defaults.
```

```{envvar} PYMOR_FIXTURES_DISABLE_BUILTIN
If set, {{ VectorArray }}, {{ Operator }} and related fixtures only use
external solver backends.
```

```{envvar} PYMOR_FORCE_JUPYTER
If set, forces pyMOR to believe that it is running in a Jupyter
environment even though automatic detection in
{func}`~pymor.core.config.is_jupyter` failed.
```

```{envvar} PYMOR_HYPOTHESIS_PROFILE
Controls which profile the hypothesis pytest plugin uses to execute our
test suites. Defaults to the "dev" profile which runs fewer variations than
the "ci" or "ci_large" which get used in our GitLab CI.
```

```{envvar} PYMOR_MPI_FINALIZE
If set, controls the value for `mpi4py.rc.finalize`. If `PYMOR_MPI_FINALIZE`
is unset the value of `mpi4py.rc.finalize` remains unchanged, unless
`mpi4py.rc.finalize is None` in which case it is defaulted to `True`.
```

```{envvar} PYMOR_MPI_INIT_THREAD
Required threading level to require when pyMOR calls `mpi4py.MPI.Init_thread`.
```
