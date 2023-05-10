#!/bin/bash

# any failure here should fail the whole test
set -eu

PYMOR_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})" ; cd ../../ ; pwd -P )"
cd "${PYMOR_ROOT}"

wget https://uploader.codecov.io/latest/linux/codecov
chmod +x codecov

rm -rf reports
mkdir reports
mv coverage*xml reports

./codecov \
  --clean \
  --token ${PYMOR_CODECOV_TOKEN} \
  --dir ./reports \
  --rootDir ${PYMOR_ROOT} \
  --flags gitlab_ci \
  --name gitlab_ci \
  --feature detect \
  --verbose \
  --nonZero \
  --slug pymor/pymor \
  --sha ${CI_COMMIT_SHA} \
  --branch ${CI_COMMIT_REF_NAME}

# after successful upload these now are only the original sqlite files
mv coverage* reports
