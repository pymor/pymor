#!/bin/bash

PYMOR_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})" ; cd ../../ ; pwd -P )"
cd "${PYMOR_ROOT}"

# any failure here should fail the whole test
set -e

# check if requirements files are up-to-date
./dependencies.py && git diff --exit-code

# most of these should be baked into the docker image already
sudo pip install -r requirements.txt
sudo pip install -r requirements-travis.txt
sudo pip install -r requirements-optional.txt || echo "Some optional modules failed to install"


python setup.py build_ext -i
if [ "${PYTEST_MARKER}" == "PIP_ONLY" ] ; then
    export SDIST_DIR=/tmp/pymor_sdist/
    # this fails on PRs, so skip it
    if [[ "${TRAVIS_PULL_REQUEST}" == "false" ]] ; then
      sudo pip install git+https://github.com/${TRAVIS_REPO_SLUG}.git@${TRAVIS_COMMIT}
      sudo pip uninstall -y pymor
    fi
    python setup.py sdist -d ${SDIST_DIR}/ --format=gztar
    check-manifest -p python ${PWD}
    pushd ${SDIST_DIR}
    sudo pip install $(ls ${SDIST_DIR})
    popd
    xvfb-run -a py.test -r sxX --pyargs pymortests -c .ci/installed_pytest.ini
    COVERALLS_REPO_TOKEN=${COVERALLS_TOKEN} coveralls
elif [ "${PYTEST_MARKER}" == "MPI" ] ; then
    xvfb-run -a mpirun --allow-run-as-root -n 2 python src/pymortests/mpi_run_demo_tests.py
elif [ "${PYTEST_MARKER}" == "NUMPY" ] ; then
    sudo pip uninstall -y numpy
    sudo pip install git+https://github.com/numpy/numpy@master
    # there seems to be no way of really overwriting -p no:warnings from setup.cfg
    sed -i -e 's/\-p\ no\:warnings//g' setup.cfg
    xvfb-run -a py.test -W once::DeprecationWarning -W once::PendingDeprecationWarning -r sxX --junitxml=test_results_${PYMOR_VERSION}.xml
else
    PYMOR_VERSION=$(python -c 'import pymor;print(pymor.__version__)')
    # this runs in pytest in a fake, auto numbered, X Server
    xvfb-run -a py.test -r sxX --junitxml=test_results_${PYMOR_VERSION}.xml
    COVERALLS_REPO_TOKEN=${COVERALLS_TOKEN} coveralls
fi

