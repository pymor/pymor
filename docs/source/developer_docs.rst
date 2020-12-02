******************************************
Developer Documentation
******************************************


pyMOR's dependencies
######################################

The single source of truth for our dependency setup is ``dependencies.py``.
From it the `requirements*txt` files  and ``pyproject.toml`` are generated (by calling ``make``).
During ``setup.py`` execution ``dependencies.py`` is imported and the package lists passed to setuptools's
``setup_requires``, ``install_requires``,``tests_require`` and ``extra_requires`` accordingly.
The ``extra_requires`` dictionary here controls what 'extra' configurations are available for
``pip install pymor[extra1,extra2,extra3]``.

When adding new package dependencies, or version restrictions, these need to be reflected into
a commit in `constraints requirements <https://github.com/pymor/docker/tree/master/constraints>`_ so that these are
become available to CI after entering the new commit hash into ``.env``.



The Makefile
######################################

Via the ``Makefile`` it is possible to execute tests close to how they are run on CI with ``make docker_test``.
All jobs described in :ref:`Gitlab CI Test Stage <ref_gitlab_ci_stage_test>` can be run this way by setting ``PYMOR_TEST_SCRIPT``
accordingly.


the ``.env`` file
=================

This file records defaults used when executing CI scripts. These are loaded by make and can be
overridden like this: ``make DOCKER_BASE_PYTHON=3.8 docker_test`` (see also the top of the ``Makefile``).


Continuous Testing / Integration Setup
######################################

Our CI infrastructure is spread across three major platforms. These are Gitlab CI (Linux tests),
Azure Pipelines (MacOS and Windows tests) and GitHub Actions (misc. checks).



.. _ref_gitlab_ci:

Gitlab CI
=========

.. note:: Configured by ``.ci/gitlab/ci.yml`` which is generated from ``.ci/gitlab/template.ci.py``
  by the calling ``make template`` (needs appropriate Python environment) or ``make docker_template``.

All stages are run in docker containers started from images that are uploaded from the CI of
``https://github.com/pymor/docker/``. Jobs that potentially install packages get an frozen pypi mirror
as a "service" image. The mirror has a "oldest" variant in which all requirements are available
in the oldest versions that still satisfy all version restrictions (recursively checked).

Stage: Sanity
---------------------

Checks if the ``setup.py`` can be processed and if all docker images needed by subsequent
stages are available in the ``zivgitlab.wwu.io/pymor/docker/`` registry.

.. _ref_gitlab_ci_stage_test:

Stage: Test
---------------------

This stage executes ``./.ci/gitlab/test_{{script}}.bash`` for a list of different scripts:

vanilla
  This runs plain `pytest` with the common options defined in ``./.ci/gitlab/common_test_setup.bash``.

cpp_demo
  Builds and executes the minimal cpp demo in ``src/pymordemos/minimal_cpp_demo/``,
  see also :doc:`tutorial_external_solver`.

mpi
  Runs all demos with ``mpirun -np 2`` via ``src/pymortests/mpi_run_demo_tests.py``, checks against recorded results.

numpy_git
  Same as vanilla, but against the unreleased numpy development branch.

oldest
  Same as vanilla, but installs only packages from the "oldest" pypi mirror.

pip_installed
  First install pyMOR using from git over https, uninstall and then install with ``pip install .[full]``.
  Uninstall again and install from a generated (and checked) sdist tarball. Lastly run the pytest suite
  on the installed (!) pyMOR, not the git worktree.

tutorials
  By using docutils magic this extracts the Python code from all the tutorials in
   ``docs/source/tutorials_*`` and runs it in parameterized pytest fixtures
   as import modules.

All scripts are executed for all Python versions that pyMOR currently supports, with the exception
of ``numpy_git`` and ``oldest``. These are only tested against the newest and oldest versions accordingly.


Stage: Build
---------------------

Stage: Install_checks
---------------------

Stage: Deploy
---------------------


Github - Gitlab bridge
----------------------

Source repo link. Merged PR branch pushing. Fork handling. Status reporting.

GitHub Actions
==============

.. note:: Configured by individual files in ``.github/workflows/*``

Azure Pipelines
===============

.. note:: Configured by ``.ci/azure/pipeline-{osx,win}.yml`` respectively.
