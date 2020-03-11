#!/bin/bash

set -eu

MANYLINUX=manylinux${1}
shift

if [[ "x${CI_COMMIT_TAG}" == "x" ]] ; then
    sed -i -e 's;style\ \=\ pep440;style\ \=\ ci_wheel_builder;g' setup.cfg
fi

set -u

# since we're in a d-in-d setup this needs to a be a path shared from the real host
BUILDER_WHEELHOUSE=${SHARED_PATH}
PYMOR_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})" ; cd ../../ ; pwd -P )"
cd "${PYMOR_ROOT}"

set -x
mkdir -p ${BUILDER_WHEELHOUSE}

BUILDER_IMAGE=pymor/wheelbuilder_${MANYLINUX}_py${PYVER}:${PYPI_MIRROR_TAG}
docker pull ${BUILDER_IMAGE} 1> /dev/null
docker run --rm  -t -e LOCAL_USER_ID=$(id -u)  \
    -v ${BUILDER_WHEELHOUSE}:/io/wheelhouse \
    -v ${PYMOR_ROOT}:/io/pymor ${BUILDER_IMAGE} /usr/local/bin/build-wheels.sh #1> /dev/null
