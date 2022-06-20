#!/bin/bash

THIS_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ; pwd -P )"
source ${THIS_DIR}/common_test_setup.bash

for fn in ${PYMOR_ROOT}/docs/source/tutorial*md ; do
  mystnb-to-jupyter -o "${fn}" "${fn/tutorial/..\/converted_tutorial}".ipynb
done

xvfb-run -a py.test ${COMMON_PYTEST_OPTS} -s --cov= -p no:pycharm \
  docs/test_tutorials.py
coverage xml

