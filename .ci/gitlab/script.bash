#!/bin/bash

if [ "x${CI_MERGE_REQUEST_ID}" == "x" ] ; then
    export PULL_REQUEST=false
else
    export PULL_REQUEST=${CI_MERGE_REQUEST_ID}
fi

export PYTHONPATH=${CI_PROJECT_DIR}/src:${PYTHONPATH}
SUDO="sudo -E -H"
PYMOR_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})" ; cd ../../ ; pwd -P )"
cd "${PYMOR_ROOT}"

# any failure here should fail the whole test
set -eux
${SUDO} pip install -U pip

# check if requirements files are up-to-date
./dependencies.py && git diff --exit-code requirements*

# most of these should be baked into the docker image already
${SUDO} pip install -r requirements.txt
${SUDO} pip install -r requirements-ci.txt
${SUDO} pip install -r requirements-optional.txt || echo "Some optional modules failed to install"

#allow xdist to work by fixing parametrization order
export PYTHONHASHSEED=0

python setup.py build_ext -i
if [ "${PYMOR_PYTEST_MARKER}" == "PIP_ONLY" ] ; then
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
    xvfb-run -a py.test -r sxX --pyargs pymortests -c .ci/installed_pytest.ini |& grep -v 'pymess/lrnm.py:82: PendingDeprecationWarning'
    pymor-demo -h
elif [ "${PYMOR_PYTEST_MARKER}" == "MPI" ] ; then
    xvfb-run -a mpirun --allow-run-as-root -n 2 python src/pymortests/mpi_run_demo_tests.py
elif [ "${PYMOR_PYTEST_MARKER}" == "NUMPY" ] ; then
    ${SUDO} pip uninstall -y numpy
    ${SUDO} pip install git+https://github.com/numpy/numpy@master
    # there seems to be no way of really overwriting -p no:warnings from setup.cfg
    sed -i -e 's/\-p\ no\:warnings//g' setup.cfg
    xvfb-run -a py.test -W once::DeprecationWarning -W once::PendingDeprecationWarning -r sxX --junitxml=test_results_${PYMOR_VERSION}.xml
else
    # this runs in pytest in a fake, auto numbered, X Server
    xvfb-run -a py.test -r sxX --junitxml=test_results.xml
fi
