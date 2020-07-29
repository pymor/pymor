******************************************
Developer Documentation
******************************************

pyMOR's dependencies
######################################

Explain dependencies.py, pip extras. Ref frozen pypi mirror for CI.

The Makefile
######################################

Test execution close to CI with ``make docker_test``. Configurable options.

Continuous Testing / Integration Setup
######################################

Our CI infrastructure is spread across three major platforms. These are Gitlab CI (Linux tests),
Azure Pipelines (MacOS and Windows tests) and GitHub Actions (misc. checks).

Gitlab CI
=========

.. note:: Configured by ``.ci/gitlab/ci.yml`` which is generated from ``.ci/gitlab/template.ci.py``
  by the calling ``make template``.

Github - Gitlab bridge
----------------------

Source repo link. Merged PR branch pushing. Fork handling. Status reporting.

GitHub Actions
==============

.. note:: Configured by individual files in ``.github/workflows/*``

Azure Pipelines
===============

.. note:: Configured by ``.ci/azure/pipeline-{osx,win}.yml`` respectively.

Local development with docker
######################################

pyMOR source mounted in .binder/ image. pycharm?
