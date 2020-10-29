#!/bin/bash

# any failure here should fail the whole test
set -eu

PYMOR_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})" ; cd ../../ ; pwd -P )"
cd "${PYMOR_ROOT}"

python3 -m pip install -U pip codecov coverage
coverage combine coverage*
coverage xml

codecov  --required \
  --token ${PYMOR_CODECOV_TOKEN} \
  --file ${PYMOR_ROOT}/.coverage \
  --root ${PYMOR_ROOT} \
  -X detect \
  --slug pymor/pymor \
  --commit ${CI_COMMIT_SHA} \
  --branch ${CI_COMMIT_REF_NAME} 
