#!/bin/bash

CODECOV_TOKEN="${PYMOR_CODECOV_TOKEN}"
SUDO="sudo -E -H"
PYMOR_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})" ; cd ../../ ; pwd -P )"
cd "${PYMOR_ROOT}"

# any failure here should fail the whole test
set -eux
${SUDO} pip install -U pip
${SUDO} pip install -r requirements-travis.txt

ls -lh
git status
codecov
