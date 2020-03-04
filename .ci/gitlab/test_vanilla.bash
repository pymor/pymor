#!/bin/bash

THIS_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ; pwd -P )"
source ${THIS_DIR}/common_test_setup.bash
# this runs in pytest in a fake, auto numbered, X Server
xvfb-run -a py.test ${COVERAGE_OPTS} --cov-report=xml -r sxX --junitxml=test_results.xml
