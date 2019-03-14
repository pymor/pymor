#!/bin/bash

# any failure here should fail the whole test
set -eu

PYMOR_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})" ; cd ../../ ; pwd -P )"
cd "${PYMOR_ROOT}"

pip install -U pip
pip install codecov

ls -lha
git status
codecov --required  --token "${PYMOR_CODECOV_TOKEN}"  --file .coverage -F ${PYMOR_PYTEST_MARKER}
