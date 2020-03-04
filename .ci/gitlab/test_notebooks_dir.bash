#!/bin/bash

THIS_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ; pwd -P )"
source ${THIS_DIR}/common_test_setup.bash

xvfb-run -a py.test ${COMMON_PYTEST_OPTS} --cov=src/pymor notebooks/test.py
