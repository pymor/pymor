#!/bin/bash

if [ "x${CI_MERGE_REQUEST_ID}" == "x" ] ; then
    export PULL_REQUEST=false
else
    export PULL_REQUEST=${CI_MERGE_REQUEST_ID}
fi

export PYTHONPATH=${CI_PROJECT_DIR}/src:${PYTHONPATH}
export PATH=~/.local/bin:${PATH}

PYMOR_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})" ; cd ../../ ; pwd -P )"
cd "${PYMOR_ROOT}"
# any failure here should fail the whole test
set -eux

# switches default index to pypi-mirror service
([[ -d ~/.config/pip/ ]] || mkdir ~/.config/pip/) && cp /usr/local/share/ci.pip.conf ~/.config/pip/pip.conf

# most of these should be baked into the docker image already
pip install -r requirements.txt
pip install -r requirements-ci.txt
pip install -r requirements-optional.txt

#allow xdist to work by fixing parametrization order
export PYTHONHASHSEED=0

python setup.py build_ext -i

PYMOR_VERSION=$(python -c 'import pymor;print(pymor.__version__)')
COMMON_PYTEST_OPTS="--junitxml=test_results_${PYMOR_VERSION}.xml --cov=src/pymor --cov-report=xml  --memprof-top-n 50 --memprof-csv-file=memory_usage.txt"
