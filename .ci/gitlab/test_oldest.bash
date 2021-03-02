#!/bin/bash

THIS_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ; pwd -P )"
source ${THIS_DIR}/common_test_setup.bash

# _downgrade_ packages to what's avail in the oldest mirror
sudo pip uninstall -y -r requirements-optional.txt
sudo pip uninstall -y -r requirements-ci.txt
sudo pip uninstall -y -r requirements.txt
sudo pip install -U -r requirements.txt
sudo pip install -U -r requirements-ci.txt
sudo pip install -U -r requirements-optional.txt

# we've changed numpy versions, recompile cyx
find src/pymor/ -name _*.c | xargs rm -f
find src/pymor/ -name _*.so | xargs rm -f
python setup.py build_ext -i

pip freeze
# this runs in pytest in a fake, auto numbered, X Server
xvfb-run -a py.test ${COMMON_PYTEST_OPTS}
coverage xml
