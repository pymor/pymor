#!/bin/bash

set -e

mkdir ${CI_PROJECT_DIR}/${ARCHIVE_DIR} && mv ${CI_PROJECT_DIR}/shared/*whl ${CI_PROJECT_DIR}/${ARCHIVE_DIR}

cd ${CI_PROJECT_DIR}
python3 setup.py sdist -d ${ARCHIVE_DIR} --format=gztar

if [[ "x${CI_COMMIT_TAG}" == "x" ]] ; then
    export TWINE_REPOSITORY_URL=https://test.pypi.org/legacy/
    export TWINE_USERNAME=${TESTPYPI_USER}
    export TWINE_PASSWORD=${TESTPYPI_TOKEN}
else
    export TWINE_USERNAME=${PYPI_USER}
    export TWINE_PASSWORD=${PYPI_TOKEN}
fi

twine check ${CI_PROJECT_DIR}/${ARCHIVE_DIR}/pymor*.whl ${CI_PROJECT_DIR}/${ARCHIVE_DIR}/pymor*tar.gz
twine upload --verbose ${CI_PROJECT_DIR}/${ARCHIVE_DIR}/pymor*.whl ${CI_PROJECT_DIR}/${ARCHIVE_DIR}/pymor*tar.gz
