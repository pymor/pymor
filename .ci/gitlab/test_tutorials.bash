#!/bin/bash

THIS_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ; pwd -P )"
source ${THIS_DIR}/common_test_setup.bash

cd ${PYMOR_ROOT}/docs/source

for fn in tutorial*.md ; do
  mystnb-to-jupyter -o "${fn}" "${fn}.ipynb"
done

pytest ${COMMON_PYTEST_OPTS} -s --nbmake *.ipynb
