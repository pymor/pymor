#!/bin/bash

set -e

mkdir -p ${CI_PROJECT_DIR}/${ARCHIVE_DIR}
mv ${CI_PROJECT_DIR}/dist/pymor*.whl ${CI_PROJECT_DIR}/dist/pymor*tar.gz ${CI_PROJECT_DIR}/${ARCHIVE_DIR}

cd ${CI_PROJECT_DIR}

if [[ "x${CI_COMMIT_TAG}" == "x" ]] ; then
    export TWINE_REPOSITORY=testpypi
    export TWINE_USERNAME=${TESTPYPI_USER}
    export TWINE_PASSWORD=${TESTPYPI_TOKEN}
else
    export TWINE_USERNAME=${PYPI_USER}
    export TWINE_PASSWORD=${PYPI_TOKEN}
fi

twine check ${CI_PROJECT_DIR}/${ARCHIVE_DIR}/pymor*.whl ${CI_PROJECT_DIR}/${ARCHIVE_DIR}/pymor*tar.gz
twine upload --verbose ${CI_PROJECT_DIR}/${ARCHIVE_DIR}/pymor*.whl ${CI_PROJECT_DIR}/${ARCHIVE_DIR}/pymor*tar.gz
