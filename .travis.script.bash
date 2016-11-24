#!/bin/bash

# base image has no sudo installed as per official docker suggestions
alias sudo='gosu root'

# most of these should be baked into the docker image already
sudo pip install -r requirements.txt
sudo pip install -r requirements-travis.txt
sudo pip install -r requirements-optional.txt || echo "Some optional modules failed to install"


python setup.py build_ext -i
if [ "${PYTEST_MARKER}" == "PIP_ONLY" ] ; then
    export SDIST_DIR=/tmp/pymor_sdist/
    # this fails on PRs, so skip it
    [[ "${TRAVIS_PULL_REQUEST}" != "false" ]] || sudo pip install git+https://github.com/${TRAVIS_REPO_SLUG}.git@${TRAVIS_COMMIT}
    sudo pip uninstall  -y pymor
    python setup.py sdist -d ${SDIST_DIR}/ --format=gztar
    check-manifest -p python ${PWD}
    pushd ${SDIST_DIR}
    sudo pip install $(ls ${SDIST_DIR})
    popd
    xvfb-run -a py.test --pyargs pymortests -c .installed_pytest.ini -k "not slow"
elif [ "${PYTEST_MARKER}" == "MPI" ] ; then
    export PYTHONPATH=$(pwd)/src
    xvfb-run -a mpirun -n 2 python src/pymortests/mpi_run_demo_tests.py
else
    export PYTHONPATH=$(pwd)/src
    # this runs in pytest in a fake, auto numbered, X Server
    xvfb-run -a py.test -k "${PYTEST_MARKER}"
fi

# TODO: insert coveralls calling here
