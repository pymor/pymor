#!/bin/bash

function docker_tag_exists() {
    curl --silent -f -lSL https://hub.docker.com/v2/repositories/$1/tags/$2 > /dev/null
}

PYMOR_ROOT="$(cd "$(dirname "$0")" && cd ../../ && pwd -P )"

cd "${PYMOR_ROOT}"

set -eux

PYTHONS="${1}"
# make sure binder setup is current
make docker_file && git diff --exit-code .binder/Dockerfile
# make sure CI setup is current
./.ci/gitlab/template.ci.py && git diff --exit-code .ci/gitlab/ci.yml
# check if requirements files are up-to-date
./dependencies.py && git diff --exit-code requirements* pyproject.toml

source ${PYMOR_ROOT}/.env
for py in ${PYTHONS} ; do
  docker_tag_exists pymor/testing_py${py} ${CI_IMAGE_TAG}
  docker_tag_exists pymor/pypi-mirror_stable_py${py} ${PYPI_MIRROR_TAG}
  docker_tag_exists pymor/pypi-mirror_oldest_py${py} ${PYPI_MIRROR_TAG}
  for ml in 1 2010 2014 ; do
    docker_tag_exists pymor/wheelbuilder_manylinux${ml}_py${py} ${PYPI_MIRROR_TAG}
  done
done

for script in ${PYMOR_ROOT}/.ci/gitlab/test* ; do
  [[ -x "${script}" ]] || exit 1
done
