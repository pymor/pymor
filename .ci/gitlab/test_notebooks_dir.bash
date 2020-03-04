#!/bin/bash

THIS_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ; pwd -P )"
source ${THIS_DIR}/common_test_setup.bash

xvfb-run -a py.test --cov=src/pymor -r sxX --junitxml=test_results.xml notebooks/test.py
