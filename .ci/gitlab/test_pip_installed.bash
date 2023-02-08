#!/bin/bash

THIS_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ; pwd -P )"
source ${THIS_DIR}/common_test_setup.bash

export SDIST_DIR=/tmp/pymor_sdist/
PIP_CLONE_URL="git+${CI_PROJECT_URL}@${CI_COMMIT_SHA}"

# Disabled, see https://github.com/pymor/pymor/issues/897
# pip install virtualenv
# virtualenv /tmp/venv
# source /tmp/venv/bin/activate
pip install ${PIP_CLONE_URL}
pip uninstall -y pymor

# this is currently disabled because it erroneously pulls in pyqt5
# pip install ${PIP_CLONE_URL}#egg=pymor[full]
# pip uninstall -y pymor

pip install .[full]
pip uninstall -y pymor
# other requirements are installed from pymor[full]
pip install -r requirements-ci.txt

python setup.py sdist -d ${SDIST_DIR}/ --format=gztar
twine check ${SDIST_DIR}/*
# silence 'detected dubious ownership in repository at '/builds/pymor/pymor''
# no idea where this comes from
git config --global --add safe.directory /builds/pymor/pymor
check-manifest -p python ${PWD}
pushd ${SDIST_DIR}
pip install $(ls ${SDIST_DIR})
popd

# make sure pytest does not pick up anything from the source tree
export PYTHONPATH=${PYTHONPATH_PRE}
tmpdir=$(mktemp -d)
pushd ${tmpdir}
# for some reason the installed conftest plugin is not picked up
cp ${PYMOR_ROOT}/conftest.py .
INI="${PYMOR_ROOT}/.ci/installed_pytest.ini"
xvfb-run -a pytest ${COMMON_PYTEST_OPTS} --cov-config=${INI} --pyargs pymortests -c ${INI}
# make sure the demo script was instaled and is usable
pymor-demo -h
popd
