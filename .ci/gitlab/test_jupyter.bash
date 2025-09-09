#!/bin/bash

THIS_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ; pwd -P )"
source ${THIS_DIR}/common_test_setup.bash
echo ${PYMOR_PYTEST_EXTRA}
echo ${COMMON_PYTEST_OPTS}
eval pytest --nbmake ${COMMON_PYTEST_OPTS} ${PYMOR_ROOT}/src/pymortests/jupyter/*.ipynb
_coverage_xml
