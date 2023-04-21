#!/bin/bash

source /venv/bin/activate

# only run fenics-related tests
export PYMOR_PYTEST_EXTRA="-m 'not builtin'"
export PYMOR_CONFIG_DISABLE="ngsolve scikit_fem dealii dunegdt"
export PYMOR_FIXTURES_DISABLE_BUILTIN="1"

THIS_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ; pwd -P )"
source ${THIS_DIR}/common_test_setup.bash

# this runs in pytest in a fake, auto numbered, X Server
#xvfb-run -a pytest ${COMMON_PYTEST_OPTS} --hypothesis-show-statistics
echo ${PYMOR_PYTEST_EXTRA}
echo ${COMMON_PYTEST_OPTS}
eval xvfb-run -a pytest ${COMMON_PYTEST_OPTS}
_coverage_xml
