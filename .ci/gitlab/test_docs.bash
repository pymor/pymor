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
${SUDO} apt update -q && ${SUDO} apt install -qy pandoc

# we actually need to have most of our deps installed
# since the docker images contains all of the external PDE solvers
${SUDO} pip install -r requirements.txt
${SUDO} pip install -r requirements-ci.txt
${SUDO} pip install -r requirements-optional.txt || echo "Some optional modules failed to install"

${SUDO} pip install .[docs]
python setup.py build_ext -i

export READTHEDOCS='True'
make docs

mkdir -p public/${CI_COMMIT_REF_SLUG}/
mv docs/_build/html/* public/${CI_COMMIT_REF_SLUG}/
cp -r docs/public_root/* public/
