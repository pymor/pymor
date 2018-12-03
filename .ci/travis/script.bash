#!/bin/bash

PYMOR_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})" ; cd ../../ ; pwd -P )"
cd "${PYMOR_ROOT}"

# any failure here should fail the whole test
set -e
sudo pip install -U pip

# check if requirements files are up-to-date
./dependencies.py && git diff --exit-code requirements*

# most of these should be baked into the docker image already
sudo pip install -r requirements.txt
sudo pip install -r requirements-travis.txt
sudo pip install -r requirements-optional.txt || echo "Some optional modules failed to install"

function coverage_submit {
    codecov
}

#allow xdist to work by fixing parametrization order
export PYTHONHASHSEED=0

python setup.py build_ext -i
if [ "${PYMOR_PYTEST_MARKER}" == "PIP_ONLY" ] ; then
    export SDIST_DIR=/tmp/pymor_sdist/
    # this fails on PRs, so skip it
    if [[ "${TRAVIS_PULL_REQUEST}" == "false" ]] ; then
      sudo pip uninstall -y -r requirements.txt
      sudo pip uninstall -y -r requirements-travis.txt
      sudo pip uninstall -y -r requirements-optional.txt || echo "Some optional modules failed to uninstall"
      sudo pip install git+https://github.com/${TRAVIS_REPO_SLUG}.git@${TRAVIS_COMMIT}
      sudo pip uninstall -y pymor
      sudo pip install git+https://github.com/${TRAVIS_REPO_SLUG}.git@${TRAVIS_COMMIT}#egg=pymor[full]
      sudo pip uninstall -y pymor
      sudo pip install -r requirements.txt
      sudo pip install -r requirements-travis.txt
      sudo pip install -r requirements-optional.txt || echo "Some optional modules failed to install"
    fi

    # README sanity
    python setup.py check -r -s
    rstcheck README.txt

    python setup.py sdist -d ${SDIST_DIR}/ --format=gztar
    sudo pip install check-manifest
    check-manifest -p python ${PWD}
    pushd ${SDIST_DIR}
    sudo pip install $(ls ${SDIST_DIR})
    popd
    xvfb-run -a py.test -r sxX --pyargs pymortests -c .ci/installed_pytest.ini

    coverage_submit
elif [ "${PYMOR_PYTEST_MARKER}" == "MPI" ] ; then
    xvfb-run -a mpirun --allow-run-as-root -n 2 python src/pymortests/mpi_run_demo_tests.py

elif [ "${PYMOR_PYTEST_MARKER}" == "NUMPY" ] ; then
    sudo pip uninstall -y numpy
    sudo pip install git+https://github.com/numpy/numpy@master
    # there seems to be no way of really overwriting -p no:warnings from setup.cfg
    sed -i -e 's/\-p\ no\:warnings//g' setup.cfg
    xvfb-run -a py.test -W once::DeprecationWarning -W once::PendingDeprecationWarning -r sxX --junitxml=test_results_${PYMOR_VERSION}.xml
    coverage_submit
else
    # this runs in pytest in a fake, auto numbered, X Server
    xvfb-run -a py.test -r sxX --junitxml=test_results.xml
    coverage_submit
fi

