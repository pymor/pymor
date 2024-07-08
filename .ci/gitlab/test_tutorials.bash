#!/bin/bash

THIS_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ; pwd -P )"
source ${THIS_DIR}/common_test_setup.bash

for fn in ${PYMOR_ROOT}/docs/source/tutorial*md ; do
  mystnb-to-jupyter -o "${fn}" "${fn/tutorial/..\/converted_tutorial}".ipynb
done

# manually add plugins to load that are excluded for other runs
xvfb-run -a pytest ${COMMON_PYTEST_OPTS} -s -p no:pycharm \
  -p nb_regression -p notebook docs/test_tutorials.py

# # manually add plugins to load that are excluded for other runs
# xvfb-run -a pytest ${COMMON_PYTEST_OPTS} --nb-coverage -s -p no:pycharm \
#   -p nb_regression --cov=pymor -p notebook docs/test_tutorials.py
#
# _coverage_xml
