#!/bin/bash

THIS_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ; pwd -P )"
source ${THIS_DIR}/common_test_setup.bash

export SDIST_DIR=/tmp/pymor_sdist/
PIP_CLONE_URL="git+${CI_PROJECT_URL}@${CI_COMMIT_SHA}"

# Disabled, see https://github.com/pymor/pymor/issues/897
# pip install virtualenv
# virtualenv /tmp/venv
# source /tmp/venv/bin/activate
pip install --use-feature=2020-resolver ${PIP_CLONE_URL}
pip uninstall -y pymor

# this is currently disabled because it erroneously pulls in pyqt5
# pip install ${PIP_CLONE_URL}#egg=pymor[full]
# pip uninstall -y pymor

pip install --use-feature=2020-resolver .[full]
pip uninstall -y pymor
# other requirements are installed from pymor[full]
pip install --use-feature=2020-resolver -r requirements-ci.txt

python setup.py sdist -d ${SDIST_DIR}/ --format=gztar
twine check ${SDIST_DIR}/*
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
xvfb-run -a pytest ${COMMON_PYTEST_OPTS} --pyargs pymortests -c ${PYMOR_ROOT}/.ci/installed_pytest.ini
# make sure the demo script was instaled and is usable
pymor-demo -h
popd
