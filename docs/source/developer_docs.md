# Developer Documentation

## Setting up an Environment for pyMOR Development

### Getting the Source

Clone the pyMOR git repository using:

```
git clone git@github.com:pymor/pymor.git
cd pymor
```

and, optionally, switch to the branch you are interested in, e.g.:

```
git checkout 2022.2.x
```

### Environment with venv

Create and activate a new venv (alternatively, you can use
[`virtualenv`](https://virtualenv.pypa.io/en/latest/) and
[`virtualenvwrapper`](https://virtualenvwrapper.readthedocs.io/en/latest/)):

```
python3 -m venv venv
source venv/bin/activate
```

Then, it may be necessary to upgrade `pip`:

```
pip install -U pip
```

Finally, make an editable installation of pyMOR with minimal dependencies:

```
pip install -e .
```

or, to install all optional dependencies and development tools:

```
pip install -e '.[full-compiled,dev]'
```

Note that the `full-compiled` extra will install `mpi4py` and `slycot`, which will require C and
Fortran compilers as well as MPI and OpenBLAS headers.
Alternatively, use the `full` extra to avoid building these additional packages.

### Environment with CI images

The docker images used in pyMOR's CI pipeline can be pulled and executed using the
`ci_<current|oldest|fenics>_image_<pull|run>` make targets.
E.g., to run the 'current' CI image use

```
make ci_current_image_run
```

This will automatically mount the pyMOR source tree into `/src`. Note that pyMOR itself is not
installed in the image, so you still have to install it using `pip install -e .`.
To launch a notebook server and bind it to port 8888, execute

```
make ci_current_image_run_notebook
```

(not available for the `fenics` image).

## Coding Guidelines and Project Management

### Python Code Style

pyMOR follows the coding style of [PEP8](https://www.python.org/dev/peps/pep-0008/) apart from a few
exceptions.
Configurations for the [ruff](https://github.com/charliermarsh/ruff) code checker are contained in
`pyproject.toml`. To check your code using `ruff`, execute

```
ruff .
```

at the root of pyMOR's source tree.

Further guidelines:

- Functions and classes called or instantiated by users should be
  sufficiently well documented.
- Use keyword arguments for parameters with defaults. This will make your
  code less likely to break, when the called function is extended.
- Generally use verbose identifiers instead of single letter names, also
  for mathematical objects (use `residual` instead of `r`). Exceptions
  are well-established variable names (like A,B,C,D,E for LTI systems)
  or temporary variables.
- Prefer assertions over exceptions in potentially performance-relevant
  code. Assertions can be ignored by invoking Python with the `-O`
  argument. Try to check a single condition in an assertion and add helpful
  error messages.
- Use `warnings.warn` for code-related issues. Use `self.logger.warning`
  for issues related to an algorithm or user input.
- It is generally ok to use builtin names as function parameters
  (e.g. `type`) when there is no other adequate name. There is no need
  to add underscores before or after the name.
- Use the `self.__auto_init(locals())` idiom to initialize instance
  attributes from `__init__` args of the same name.

:::{note}
If you are a first-time contributor, do not worry too much about code
style. The main developers will be happy to help you to bring your code
into proper shape for inclusion in pyMOR.
:::

### Markdown Style

The Markdown style is determined by the
[`markdownlint`](https://github.com/DavidAnson/markdownlint) rules,
specified in `.markdownlint.yml`.
The [`markdownlint-cli2`](https://github.com/DavidAnson/markdownlint-cli2) tool
(or the `pre-commit` hook; see below)
can be used to check for errors.

### GitHub Project

All new code enters pyMOR by means of a pull request. Pull requests (PR) trigger {ref}`automatic tests <ref_testing_ci>`
which are mandatory to pass before the PR can be merged. A PR must also be tagged with either one of the
`pr:new-feature`, `pr:fix`, `pr:removal`, `pr:deprecation` or `pr:change` labels to clearly identify the type of
changes it introduces. See the [labels' descriptions](https://github.com/pymor/pymor/labels?q=pr%3A) for more details.

### `pre-commit` Hooks

pyMOR ships a config for the [pre-commit](https://pre-commit.com/) hook management system.
Using this setup can be a good way to find code style errors before pushing to GitHub, but using
it is not required. Once you have `pre-commit` installed in you environment, run

```
pre-commit install
```

Afterwards, the hooks configured in `.pre-commit-config.yaml` will run on all changed
files prior to committing changes. Errors will block the commit, some
checks will automatically fix the file.

### Updating pyMOR's Dependencies

All required or optional dependencies of pyMOR are specified in `pyproject.toml`.

We use [pip-compile](https://github.com/jazzband/pip-tools), to generate `requirements-ci-*.txt`
files from these specifications, which contain pinned versions of all packaged installed into the
respective GitLab CI images.
The extras included into the images are specified in `Makefile`.
For the `oldest` CI image, `requirements-ci-oldest-pins.in` is used in addition, which pins some of
pyMOR's core dependencies to the oldest version supported by pyMOR.
Similarly to the `pip-compile` workflow, we use [conda-lock](https://github.com/conda/conda-lock)
to create [conda-forge](https://conda-forge.org/) environment lock files that are used for the
GitHub actions CI builds.

If you update pyMOR's dependencies, make sure to execute

```
make ci_requirements
```

and commit the changes made to the lock files to ensure that the updated dependencies are picked up
by CI.

Note that `make ci_requirements` requires [docker](https://www.docker.com/) or a compatible
container runtime such as [podman](https://podman.io/).
The pyMOR main developers will be happy to take care of this step for you.

(ref_testing_ci)=

## Testing / Continuous Integration Setup

### pyMOR's Test Suite

pyMOR uses [pytest](https://pytest.org/) for unit testing.
All tests are contained within the `src/pymortests` directory and can be run
by invoking the `pytest` executable.
Please refer to the [pytest documentation](https://docs.pytest.org/en/latest/how-to/usage.html)
for detailed examples.
To run the entire test suite, it is also possible to execute

```
make test
```

which will use the [Xvfb](https://www.x.org/releases/X11R7.6/doc/man/man1/Xvfb.1.xhtml) virtual
framebuffer X server, to prevent GUI windows from popping up during the test run.

pyMOR uses the [hypothesis](https://hypothesis.works/) for property-based testing.
To select the amount of test samples to create, you can specify the hypothesis profile to use by
setting the PYMOR_HYPOTHESIS_PROFILE.
Available profiles are `dev` (default, shortest execution time), `ci` and `ci_large` (longest
execution time).
The profiles are defined in `conftest.py`.

To disable tests for some of pyMOR's optional dependencies, set `PYMOR_CONFIG_DISABLE` to a
whitespace-separated list of {mod}`config <pymor.core.config>` items which should be prevented
from being loaded during the test session.
Conversely, to disable all tests that only use pyMOR's builtin discretization toolkit, set
`PYMOR_FIXTURES_DISABLE_BUILTIN` to `1` and pass `-m 'not builtin'` to the `pytest` command line.

### GitLab CI

We use GitLab deployed at https://zivgitlab.uni-muenster.de/pymor/pymor as our main CI
infrastructure.
All CI stages are run in docker containers that are built and pushed to the GitLab container
registry using

```
make ci_preflight_image      # used for preflight/docker images stages
make ci_preflight_image_push
make ci_images               # used for actual tests
make ci_images_push
```

The corresponding `Dockerfiles` are all contained in the `docker` directory.

The images are tagged with a `sha256sum` of the corresponding `requirements-ci-*.txt` file.
Images corresponding to the current state of `main` are tagged with `main`.
Images corresponding to tagged commits are tagged with the respective git tag
(see {ref}`ref_gitlab_ci_stage_deploy`).

#### Stage: preflight

The main responsible of the `preflight` job is to compute the `sha256sum` of all
`requirements-ci-*.txt` files in order to infer the tags of the CI images used for testing.
It also queries GitLab's container registry to check if the needed images are available.
The results are saved in a
[dotenv](https://docs.gitlab.com/ee/ci/yaml/artifacts_reports.html#artifactsreportsdotenv)
file to make them available as environment variables in later CI stages.

#### Stage: docker images

When need, updated CI images are built using a podman-in-docker setup.
The resulting images are then uploaded to the container registry.

(ref_gitlab_ci_stage_testbuild)=

#### Stage: test/build

This stage executes the `test_*.bash` scripts located in `./.ci/gitlab/`.
For each supported external PDE solver backend, an individual CI job is run.
Documentation is also built in this stage.

(ref_gitlab_ci_stage_deploy)=

#### Stage: deploy

docs
: Commits documentation built in {ref}`ref_gitlab_ci_stage_testbuild`to the
  [documentation repository](https://github.com/pymor/docs). This repository is the source for
  [https://docs.pymor.org/](https://docs.pymor.org/) served via GitHub Pages.
  A binder setup for the generated tutorials notebooks is added on a branch with a name
  matching the currently checked out git branch of pyMOR.

submit coverage
: This job accumulates all the coverage databases generated by previous stages and submits that
  to [codecov.io](https://codecov.io/github/pymor/pymor/).

coverage html
: Generates an html coverage report, which can be downloaded as a job artifact.

tag docker images
: If the running pipeline corresponds to a push on `main` or a tagged commit (release), the used
  CI images are tagged with either `main` or the corresponding git tag in GitLab's container
  registry.
  This will prevent these images from being removed during registry cleanup.
  In particular, this guarantees that the images are available for the binder setups linked in the
  online documentation.

### GitHub Actions

We use GitHub Actions to run some additional checks.
This includes Conda-based MacOS and Windows test suite runs and management of GitHub labels.
Further, for PRs from external repositories, a local mirror branch is created and updated.
This enables GitLab CI runs those PRs.

### pre-commit.ci

The main pyMOR repository has the `pre-commit.ci` GitHub App installed. This runs the pre-commit hooks
defined in `.pre-commit-config.yaml` on every pull request.
If the hooks change files, the changes are pushed back to the PR branch.
[Configuration](https://pre-commit.ci/#configuration) is done
via the `.pre-commit-config.yaml` file.
