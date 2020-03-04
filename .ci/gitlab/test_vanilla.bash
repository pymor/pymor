#!/bin/bash

THIS_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ; pwd -P )"
source ${THIS_DIR}/common_test_setup.bash
# this runs in pytest in a fake, auto numbered, X Server
xvfb-run -a py.test ${COMMON_PYTEST_OPTS}
