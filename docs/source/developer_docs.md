# Developer Documentation

## Setting up an Environment for pyMOR Development

### Getting the source

Clone the pyMOR git repository using:

```
git clone https://github.com/pymor/pymor pymor_source
cd pymor_source
```

and, optionally, switch to the branch you are interested in, e.g.:

```
git checkout 2020.2.x
```

### Environment with Virtualenv

Create and activate a new virtualenv:

```
python3 -m virtualenv venv
source venv/bin/activate
```

Then, make an editable installation of pyMOR with:

```
pip install -e '.[full]'
```

### Environment with docker-compose

To get a shell in a preconfigured container run
(if required set `PYMOR_SUDO=1` in your environment to execute docker with elevated rights):

```
make docker_run
```

You can also use the setup in `.binder/docker-compose.yml` to easily
work on pyMOR with [pyCharm](https://www.jetbrains.com/help/pycharm/docker-compose.html).

## Coding guidelines and project management

### Code style

pyMOR follows the coding style of
[PEP8](https://www.python.org/dev/peps/pep-0008/) apart from a
few exceptions. Configurations for the [PEP8](https://pypi.python.org/pypi/pep8) and
[flake8](https://pypi.python.org/pypi/flake8) code checkers are contained in `setup.cfg`.

As an additional rule when calling functions, positional
arguments should generally be passed as positional arguments
whereas keyword arguments should be passed as keyword arguments.
This will make your code less likely to break, when the called
function is extended.

All functions and classes called or instantiated by users should
be sufficiently well documented.

#### How to check code style

Firstly, make sure that you installed the `requirements-ci.txt` with `pip install -r requirements-ci.txt`.
Afterwards use the Makefile to check for flake8 warnings with `make flake8`
or directly use `flake8` on the `src` folder.

### GitHub project

All new code enters pyMOR by means of a pull request. Pull requests (PR) trigger [automatic tests](#continuous-testing--integration-setup)
which are mandatory to pass before the PR can be merged. A PR must also be tagged with either one of the
`pr:new-feature`, `pr:fix`, `pr:removal`, `pr:deprecation` or `pr:change` labels to clearly identify the type of
changes it introduces. See the [labels' descriptions](https://github.com/pymor/pymor/labels?q=pr%3A) for more details.

### pre-commit hooks

pyMOR ships a config for the [pre-commit](https://pre-commit.com/) hook management system.
Using this setup can be a good way to find flake8 errors before pushing to GitHub, but using
it is not required. Once you have `pre-commit` installed in you environment, run

```
pre-commit install
```

Afterwards, the hooks configured in `.pre-commit-config.yaml` will run on all changed
files prior to committing changes. Errors will block the commit, some
checks will automatically fix the file.

## pyMOR's dependencies

The single source of truth for our dependency setup is `dependencies.py`.
From it the `requirements*txt` files  and `pyproject.toml` are generated (by calling `make`).
During `setup.py` execution `dependencies.py` is imported and the package lists passed to setuptools's
`setup_requires`, `install_requires`, `tests_require` and `extra_requires` accordingly
(see {ref}`this list <ref_gitlab_ci_stage_test>`).
The `extra_requires` dictionary here controls what 'extra' configurations are available for
`pip install pymor[extra1,extra2,extra3]`.
The requirements files are input for `.ci/create_conda_env.py` which created a Conda Environment
spec file that contains only those dependencies that are available on all OS-Python combinations
for which GitHub Action based CI is run in Conda envs.

A GitHub Action will update all "downstream" files of `dependencies.py` if it changes in a push.
The result will be pushed into the same branch, but due to a GitHub limitation no new workflows
will run as a result of that push.

When adding new package dependencies, or version restrictions, these need to be reflected into
a commit in our docker repository for the [constraints requirements](https://github.com/pymor/docker/tree/main/constraints)
so that updated images become available to CI after entering the new commit hash into `.env`.

(ref-makefile)=

## The Makefile

Via the `Makefile` it is possible to execute tests close to how they are run on CI with `make docker_test`.
All jobs described in {ref}`Gitlab CI Test Stage <ref_gitlab_ci_stage_test>` can be run this way by setting `PYMOR_TEST_SCRIPT`
accordingly. You can pass additional arguments to pytest by setting `PYMOR_PYTEST_EXTRA`.
To run the test suite without docker,
simply execute `make test` in the base directory of the pyMOR repository. This will
run the pytest suite with the default hypothesis profile "dev". For available profiles
see `src/pymortests/conftest.py`. A profile is selected by running `make PYMOR_HYPOTHESIS_PROFILE=PROFILE_NAME test`.
Run `make full-test` to also enable
[pyflakes](https://pypi.python.org/pypi/pyflakes) and [pep8](https://www.python.org/dev/peps/pep-0008/) checks.

### the `.env` file

This file records defaults used when executing CI scripts. These are loaded by make and can be
overridden like this: `make DOCKER_BASE_PYTHON=3.8 docker_test` (see also the top of the `Makefile`).


## Continuous Testing / Integration Setup

Our CI infrastructure is spread across two major platforms. These are Gitlab CI (Linux testsuite)
and GitHub Actions (Conda-based MacOS and Windows testsuite, misc. checks).

pyMOR uses [pytest](https://pytest.org/) for unit testing.
All tests are contained within the `src/pymortests` directory and can be run
individually by executing `python3 src/pymortests/the_module.py` or invoking
pytest directly. Please refer to the [pytest documentation](https://docs.pytest.org/en/latest/how-to/usage.html)
for detailed examples.

(ref_gitlab_ci)=

### Gitlab CI

:::{note}
Configured by `.ci/gitlab/ci.yml` which is generated from `.ci/gitlab/template.ci.py`
by the calling `make template` (needs appropriate Python environment) or `make docker_template`.
:::

All stages are run in docker containers ({ref}`more info  <ref_docker_images>`).
Jobs that potentially install packages get a frozen pypi mirror
as a "service" container. The mirror has a "oldest" variant in which all requirements are available
in the oldest versions that still satisfy all version restrictions (recursively checked).

(ref_gitlab_ci_stage_sanity)=

#### Stage: Sanity

A smoke test for the CI setup itself.
Checks if the `setup.py` can be processed and if all docker images needed by subsequent
stages are available in the `zivgitlab.wwu.io/pymor/docker/` registry.
Also ensures CI config and requirements generated from their templates match the committed files.

(ref_gitlab_ci_stage_test)=

#### Stage: Test

This stage executes `./.ci/gitlab/test_{{script}}.bash` for a list of different scripts:

vanilla

: This runs plain `pytest` with the common options defined in `./.ci/gitlab/common_test_setup.bash`.

cpp_demo

: Builds and executes the minimal cpp demo in `src/pymordemos/minimal_cpp_demo/`,
  see also {doc}`tutorial_external_solver`.

mpi

: Runs all demos with `mpirun -np 2` via `src/pymortests/mpi_run_demo_tests.py`, checks against recorded results.

numpy_git

: Same as vanilla, but against the unreleased numpy development branch. This makes sure we catch
  deprecation warnings or breaking changes early.

oldest

: Same as vanilla, but installs only packages from the "oldest" pypi mirror.

pip_installed

: First install pyMOR from git over https, uninstall and then install with `pip install .[full]`.
  Uninstall again and install from a generated (and checked) sdist tarball. Lastly run the pytest suite
  on the installed (!) pyMOR, not the git working tree.

tutorials

: Using docutils magic this extracts the Python code from all the tutorials in
  `docs/source/tutorials_*` (except tutorial_external_solver since that needs kernel switching)
  and runs it in parameterized pytest fixtures as imported modules.

All scripts are executed for all Python versions that pyMOR currently supports, with the exception
of `numpy_git` and `oldest`. These are only tested against the newest and oldest versions accordingly.

(ref_gitlab_ci_stage_build)=

#### Stage: Build

Builds documentation and manylinux wheels on all supported pythons. Also builds and pushes
a docker image that includes pyMOR installed from checkout. This is used as the base image for the binder-ready
deployment of the documentation in the last stage.

(ref_gitlab_ci_stage_install)=

#### Stage: Install_checks

from wheel

: Try to install wheels produced in previous stage on a few different Linuxs.

from source

: Try to install `pymor[full]` from git checkout. This checks that the extension module compile works,
  which is not covered by the "from wheel" step. Also install full optional requirements, which include
  packages omitted from `[full]`, after necessary additional system package install.

local docker

: Ensures minimal functionality for the local docker development setup .

(ref_gitlab_ci_stage_deploy)=

#### Stage: Deploy

docs

: Commits documentation built in {ref}`ref_gitlab_ci_stage_build` (from a single Python version, not all) to the
  [documentation repository](https://github.com/pymor/docs). This repository is the source for
  [https://docs.pymor.org/](https://docs.pymor.org/) served via GitHub Pages.
  A binder setup for the generated tutorials notebooks is added on a branch with a name
  matching the currently checked out git branch of pyMOR.

pypi

: Upload wheels to either the test or the real instance of the pypi repository, depending on whether
  the pipeline runs for a tagged commit.

coverage

: This job accumulates all the coverage databases generated by previous stages and submits that
  to [codecov.io](https://codecov.io/github/pymor/pymor/).

#### Github - Gitlab bridge

This a sanic based Python [application](https://github.com/pymor/ci_hooks_app) that receives webhook
events from GitHub for pull requests and pushes PR branches merged into main to Gitlab to run a
parallel CI pipeline to check whether the main branch will still pass tests after the PR is merged.
The bridge also does this for forks of pyMOR, but these have to be whitelisted in order to protect CI secrets.

### GitHub Actions

:::{note}
Configured by individual files in `.github/workflows/*`
:::

- Check all (external) links in changed Markdown documents are accessible.

- Make sure at least one `pr:*` label is set on the PR.

- Prohibit any commits with messages that indicate they can be auto-squashed

- Auto-assign the labels if certain files are changed by the PR.

- Update requirement files / conda env if `depependencies.py` changes.

- Runs pytest in conda-based environments on Windows/MacOS/Linux for oldest and newest supported Pythons
  - Entire conda envs are cached and only update if either manually invalidated by incrementing `CACHE_NUMBER`
    or if dependencies change.
  - All pytest jobs export a full environment lockfile, which can be downloaded on the summary tab for
    the "Conda Tests" action. Look for "Conda Env Exports".
  - Pytest XML reports can also be found there.

(ref_docker_images)=

### Docker images

The source for most of our docker images is this [repository](https://github.com/pymor/docker).
The images are build by a Makefile system that expresses dependencies, handles parameterization,
preloads caching and so forth. Builds are only permitted on a clean working tree, to increase reproducibility.
On each push [GitLab CI](https://zivgitlab.uni-muenster.de/pymor/docker/-/pipelines) builds the entire tree.
Great effort went into making incremental updates as fast as possible, but full rebuilds will take upwards of 3 hours.
There are two basic categories for images: those that get generated for each supported Python version and those that
are version independent.
For CI the main image, in which the pytest suite runs, is defined in `testing/`. The main workflow for this repository
is adding new packages to the appropriate requirements file in the `constraints/` subdir. From there
those packages will become available in the `pypi_mirror-*` images, but also pre-installed in the `testing` image.
