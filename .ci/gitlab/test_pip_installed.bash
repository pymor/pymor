#!/bin/bash

THIS_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ; pwd -P )"
source "${THIS_DIR}/common_test_setup.bash"

export SDIST_DIR=/tmp/pymor_sdist/
PIP_CLONE_URL="git+${CI_PROJECT_URL}@${CI_COMMIT_SHA}"

# Disabled, see https://github.com/pymor/pymor/issues/897
# pip install virtualenv
# virtualenv /tmp/venv
# source /tmp/venv/bin/activate

function uninstall () {
  local pref=${1:-}
  for req in ${PYMOR_ROOT}/requirements*txt ; do
    ${pref} pip uninstall -yr "${req}"
    ${pref} pip uninstall -y pymor
  done
}

# first uninstall needs sudo for /usr/local
uninstall "sudo -H"
pip install ${PIP_CLONE_URL}

# this is currently disabled because it erroneously pulls in pyqt5
#uninstall
#pip install "${PIP_CLONE_URL}#egg=pymor[full]"

uninstall
pip install -e .[full]
pip install -r requirements-ci.txt
python setup.py sdist -d ${SDIST_DIR}/ --format=gztar
twine check ${SDIST_DIR}/*
# silence 'detected dubious ownership in repository at '/builds/pymor/pymor''
# no idea where this comes from
git config --global --add safe.directory /builds/pymor/pymor
check-manifest -p python ${PWD}
pushd ${SDIST_DIR}
uninstall
pip install $(ls ${SDIST_DIR})
popd

uninstall
pip install .[full]

# other requirements are installed from pymor[full]
pip install -r requirements-ci.txt
# make sure pytest does not pick up anything from the source tree
export PYTHONPATH=${PYTHONPATH_PRE}
tmpdir=$(mktemp -d)
pushd ${tmpdir}
# for some reason the installed conftest plugin is not picked up
cp ${PYMOR_ROOT}/conftest.py .
INI="${PYMOR_ROOT}/.ci/installed_pytest.ini"
xvfb-run -a python -m pytest ${COMMON_PYTEST_OPTS} --cov-config=${INI} --pyargs pymortests -c ${INI}
# make sure the demo script was instaled and is usable
pymor-demo -h
popd
