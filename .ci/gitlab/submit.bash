#!/bin/bash

# any failure here should fail the whole test
set -eu

PYMOR_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})" ; cd ../../ ; pwd -P )"
cd "${PYMOR_ROOT}"

pip install -U pip
pip install codecov

codecov -v --required \
  --token "${PYMOR_CODECOV_TOKEN}" \
  --file coverage.xml \
  --flags "${COVERAGE_FLAG}" \
  --root "${PYMOR_ROOT}" \
  -X detect \
  --slug pymor/pymor \
  --commit ${CI_COMMIT_SHA} \
  --branch ${CI_COMMIT_REF_NAME}
