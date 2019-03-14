#!/bin/bash

# any failure here should fail the whole test
set -eux

CODECOV_TOKEN="${PYMOR_CODECOV_TOKEN}"
PYMOR_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})" ; cd ../../ ; pwd -P )"
cd "${PYMOR_ROOT}"

pip install -U pip
pip install -r requirements-travis.txt

ls -lh
git status
codecov
