#!/bin/bash

# any failure here should fail the whole test
set -eu

PYMOR_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})" ; cd ../../ ; pwd -P )"
cd "${PYMOR_ROOT}"

pip install -U pip
pip install codecov

codecov -v --required \
  --token "${PYMOR_CODECOV_TOKEN}" \
  --file .coverage \
  --flags "${PYMOR_PYTEST_MARKER}" \
  --root "${PYMOR_ROOT}"
