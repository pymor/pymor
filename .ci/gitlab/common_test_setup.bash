#!/bin/bash

if [ "x${CI_MERGE_REQUEST_ID}" == "x" ] ; then
    export PULL_REQUEST=false
else
    export PULL_REQUEST=${CI_MERGE_REQUEST_ID}
fi

export PYTHONPATH_PRE=${PYTHONPATH}
export PYTHONPATH=${CI_PROJECT_DIR}/src:${PYTHONPATH}
export PATH=~/.local/bin:${PATH}
export PYBIND11_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")

export PYMOR_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})" ; cd ../../ ; pwd -P )"
cd "${PYMOR_ROOT}"
# any failure here should fail the whole test
set -eux

# switches default index to pypi-mirror service
export PIP_CONFIG_FILE=/usr/local/share/ci.pip.conf

# make sure image correct packages are baked into the image
python src/pymor/scripts/check_reqs.py requirements.txt
python src/pymor/scripts/check_reqs.py requirements-ci.txt
python src/pymor/scripts/check_reqs.py requirements-optional.txt

#allow xdist to work by fixing parametrization order
export PYTHONHASHSEED=0

# workaround import mpl with no ~/.cache/matplotlib/fontconfig*.json
# present segfaulting the interpreter
python -c "from matplotlib import pyplot" || true

PYMOR_VERSION=$(python -c 'import pymor;print(pymor.__version__)')
# `--cov-report=` suppresses terminal output
COMMON_PYTEST_OPTS="--junitxml=test_results_${PYMOR_VERSION}.xml \
  --cov-report= --cov --cov-config=setup.cfg --cov-context=test \
  --hypothesis-profile ${PYMOR_HYPOTHESIS_PROFILE} ${PYMOR_PYTEST_EXTRA}"
