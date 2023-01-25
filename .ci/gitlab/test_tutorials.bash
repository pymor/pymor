#!/bin/bash

COV_OPTION="--cov=pymor"

THIS_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ; pwd -P )"
source ${THIS_DIR}/common_test_setup.bash


# manually add plugins to load that are excluded for other runs
xvfb-run -a pytest ${COMMON_PYTEST_OPTS} --nb-coverage -s -p no:pycharm \
  -p nb_regression --cov=pymor -p notebook docs/test_tutorials.py

_coverage_xml
