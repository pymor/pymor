#!/bin/bash


set -e
export PYTHONPATH=${CI_PROJECT_DIR}/src:${PYTHONPATH}
SUDO="sudo -E -H"
PYMOR_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})" ; cd ../../ ; pwd -P )"
cd "${PYMOR_ROOT}"
COVERAGE_OPTS="--cov=src/pymor --cov-report=xml  --memprof-top-n 50 --memprof-csv-file=memory_usage.txt"
# any failure here should fail the whole test
set -eux
${SUDO} pip install -U pip

# most of these should be baked into the docker image already
${SUDO} pip install -r requirements.txt
${SUDO} pip install -r requirements-ci.txt

python setup.py build_ext -i
cd src/pymordemos/minimal_cpp_demo
cmake .
make
xvfb-run -a python -c 'import runpy; runpy.run_module("demo")'
