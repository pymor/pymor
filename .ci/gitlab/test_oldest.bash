#!/bin/bash

THIS_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ; pwd -P )"
source ${THIS_DIR}/common_test_setup.bash

${SUDO} pip install pypi-oldest-requirements>=2020.2
# replaces any loose pin with a hard pin on the oldest version
pypi_minimal_requirements_pinned requirements.txt requirements.txt
pypi_minimal_requirements_pinned requirements-travis.txt requirements-travis.txt
pypi_minimal_requirements_pinned requirements-optional.txt requirements-optional.txt
${SUDO} pip install -r requirements.txt
${SUDO} pip install -r requirements-travis.txt
${SUDO} pip install -r requirements-optional.txt || echo "Some optional modules failed to install"
# we've changed numpy versions, recompile cyx
find src/pymor/ -name _*.c | xargs rm -f
find src/pymor/ -name _*.so | xargs rm -f
python setup.py build_ext -i

pip freeze
# this runs in pytest in a fake, auto numbered, X Server
xvfb-run -a py.test ${COMMON_PYTEST_OPTS}
