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

# try to make env similar to what's on RTD
${SUDO} pip uninstall -y -r requirements.txt
${SUDO} pip uninstall -y -r requirements-travis.txt
${SUDO} pip uninstall -y -r requirements-optional.txt || echo "Some optional modules failed to uninstall"
${SUDO} pip install -r requirements-rtd.txt

${SUDO} python setup.py install --force

sphinx-build -T -b readthedocs -d _build/doctrees-readthedocs -D language=en . _build/html
