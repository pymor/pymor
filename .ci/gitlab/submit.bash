#!/bin/bash

# any failure here should fail the whole test
set -eu

PYMOR_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})" ; cd ../../ ; pwd -P )"
cd "${PYMOR_ROOT}"

wget https://uploader.codecov.io/latest/alpine/codecov
chmod +x codecov

rm -rf reports
mkdir reports
mv coverage*xml reports

./codecov  --required \
  -t ${PYMOR_CODECOV_TOKEN} \
  -s ./reports \
  --rootDir ${PYMOR_ROOT} \
  --flags gitlab_ci \
  --name gitlab_ci \
  -X detect \
  -v \
  -Z \
  --slug pymor/pymor \
  --sha ${CI_COMMIT_SHA} \
  --branch ${CI_COMMIT_REF_NAME}
