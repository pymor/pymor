#!/bin/bash

THIS_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ; pwd -P )"
source ${THIS_DIR}/common_test_setup.bash

export SDIST_DIR=/tmp/pymor_sdist/
PIP_CLONE_URL="git+${CI_PROJECT_URL}@${CI_COMMIT_SHA}"
${SUDO} pip uninstall -y -r requirements.txt
${SUDO} pip uninstall -y -r requirements-ci.txt
${SUDO} pip uninstall -y -r requirements-optional.txt || echo "Some optional modules failed to uninstall"
${SUDO} pip install ${PIP_CLONE_URL}
${SUDO} pip uninstall -y pymor
${SUDO} pip install ${PIP_CLONE_URL}#egg=pymor[full]
${SUDO} pip uninstall -y pymor
${SUDO} pip install -r requirements.txt
${SUDO} pip install -r requirements-ci.txt
${SUDO} pip install -r requirements-optional.txt || echo "Some optional modules failed to install"

python setup.py sdist -d ${SDIST_DIR}/ --format=gztar
twine check ${SDIST_DIR}/*
check-manifest -p python ${PWD}
pushd ${SDIST_DIR}
${SUDO} pip install $(ls ${SDIST_DIR})
popd
set -o pipefail
xvfb-run -a py.test ${COVERAGE_OPTS} -r sxX --pyargs pymortests -c .ci/installed_pytest.ini |& grep -v 'pymess/lrnm.py:82: PendingDeprecationWarning'
pymor-demo -h
