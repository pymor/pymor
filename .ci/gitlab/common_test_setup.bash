#!/bin/bash

if [ "x${CI_MERGE_REQUEST_ID}" == "x" ] ; then
    export PULL_REQUEST=false
else
    export PULL_REQUEST=${CI_MERGE_REQUEST_ID}
fi

COVERAGE_FILE=${COVERAGE_FILE:-coverage}
COV_OPTION=${COV_OPTION:---cov=src}
PYMOR_HYPOTHESIS_PROFILE=${PYMOR_HYPOTHESIS_PROFILE:-dev}
PYMOR_PYTEST_EXTRA=${PYMOR_PYTEST_EXTRA:-}

function _coverage_xml() {
  local extra=${1:-}
  # make sure to fail if there was an error collecting data
  coverage xml -o ${COVERAGE_FILE}.xml ${extra} --fail-under=10
}
export PYTHONPATH_PRE=${PYTHONPATH}
export PYTHONPATH=${CI_PROJECT_DIR}/src:${PYTHONPATH}
export PATH=~/.local/bin:${PATH}
export PYBIND11_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")

export PYMOR_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})" ; cd ../../ ; pwd -P )"
cd "${PYMOR_ROOT}"
# any failure here should fail the whole test
set -eux

# pymor checks if this file's owner uid matches with the interpreter executor's
ME=$(id -u)
DEFAULTS_OWNER="$(stat --format '%u' docs/source/pymor_defaults.py)"
if [ "x${ME}" != "x${DEFAULTS_OWNER}" ] ; then
  sudo chown ${ME} docs/source/pymor_defaults.py
fi

#allow xdist to work by fixing parametrization order
export PYTHONHASHSEED=0

# workaround import mpl with no ~/.cache/matplotlib/fontconfig*.json
# present segfaulting the interpreter
python -c "from matplotlib import pyplot" || true

PYMOR_VERSION=$(python -c 'import pymor;print(pymor.__version__)')

# `--cov-report=` suppresses terminal output
COMMON_PYTEST_OPTS="--junitxml=test_results_${PYMOR_VERSION}.xml \
  --cov-report= ${COV_OPTION} --cov-config=${PYMOR_ROOT}/setup.cfg --cov-context=test \
  --hypothesis-profile ${PYMOR_HYPOTHESIS_PROFILE} ${PYMOR_PYTEST_EXTRA}"

python -m pytest src/pymortests/docker_ci_smoketest.py
