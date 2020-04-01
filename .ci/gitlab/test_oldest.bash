#!/bin/bash

THIS_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ; pwd -P )"
source ${THIS_DIR}/common_test_setup.bash

# we've changed numpy versions, recompile cyx
find src/pymor/ -name _*.c | xargs rm -f
find src/pymor/ -name _*.so | xargs rm -f
python setup.py build_ext -i

pip freeze
# this runs in pytest in a fake, auto numbered, X Server
xvfb-run -a py.test ${COMMON_PYTEST_OPTS}
