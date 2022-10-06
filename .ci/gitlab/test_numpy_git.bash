#!/bin/bash

THIS_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ; pwd -P )"
source ${THIS_DIR}/common_test_setup.bash

# allow numpy main to pull in requirements not yet in our docker images
# to be reverted after https://github.com/pymor/pymor/pull/1555
unset PIP_CONFIG_FILE
pip install git+https://github.com/numpy/numpy@main
# there seems to be no way of really overwriting -p no:warnings from setup.cfg
sed -i -e 's/\-p\ no\:warnings//g' setup.cfg
xvfb-run -a pytest ${COMMON_PYTEST_OPTS} -W once::DeprecationWarning -W once::PendingDeprecationWarning
_coverage_xml
