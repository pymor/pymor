#!/bin/bash

mkdir ${CI_PROJECT_DIR}/${ARCHIVE_DIR} && mv ${CI_PROJECT_DIR}/shared/*whl ${CI_PROJECT_DIR}/${ARCHIVE_DIR}

cd ${CI_PROJECT_DIR}
python setup.py sdist -d ${ARCHIVE_DIR} --format=gztar

if [[ "x${CI_COMMIT_TAG}" == "x" ]] ; then
    TWINE_REPOSITORY=testpypi
    TWINE_USER=${TESTPYPI_USER}
    TWINE_PASSWORD=${TESTPYPI_USER}
else
    TWINE_REPOSITORY=pypi
    TWINE_USER=${PYPI_USER}
    TWINE_PASSWORD=${PYPI_USER}
fi
export TWINE_NON_INTERACTIVE=1
#twine upload --verbose ${CI_PROJECT_DIR}/${ARCHIVE_DIR}/pymor*.whl ${CI_PROJECT_DIR}/${ARCHIVE_DIR}/pymor*tar.gz
