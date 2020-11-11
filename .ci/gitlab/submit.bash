#!/bin/bash

# any failure here should fail the whole test
set -eu

PYMOR_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})" ; cd ../../ ; pwd -P )"
cd "${PYMOR_ROOT}"

python3 -m pip install -U pip codecov coverage
coverage combine coverage*
# the mpi test_thermalblock_ipython results in '(builtin)' missing which we "--ignore-errors"
rm -f coverage.xml
coverage xml --ignore-errors

codecov  --required \
  --token ${PYMOR_CODECOV_TOKEN} \
  --file ${PYMOR_ROOT}/coverage.xml \
  --root ${PYMOR_ROOT} \
  -X detect \
  --slug pymor/pymor \
  --commit ${CI_COMMIT_SHA} \
  --branch ${CI_COMMIT_REF_NAME} 
