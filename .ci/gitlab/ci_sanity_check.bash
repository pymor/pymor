#!/bin/bash

function docker_tag_exists() {
    curl --silent -f -lSL https://hub.docker.com/v2/repositories/$1/tags/$2 > /dev/null
}

PYMOR_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})" ; cd ../../ ; pwd -P )"
cd "${PYMOR_ROOT}"

set -eux

PYTHONS="${1}"
# make sure binder setup is current
make docker_file && git diff --exit-code .binder/Dockerfile
# make sure CI setup is current
./.ci/gitlab/template.ci.py && git diff --exit-code .ci/gitlab/ci.yml
# check if requirements files are up-to-date
./dependencies.py && git diff --exit-code requirements* pyproject.toml

for py in ${PYTHONS} ; do
  docker_tag_exists pymor/testing_py${py} $(cat .ci/CI_IMAGE_TAG)
  docker_tag_exists pymor/pypi-mirror_stable_py${py} $(cat .ci/PYPI_MIRROR_TAG)
  docker_tag_exists pymor/pypi-mirror_oldest_py${py} $(cat .ci/PYPI_MIRROR_TAG)
done

for script in ${PYMOR_ROOT}/.ci/gitlab/test* ; do
  [[ -x "${script}" ]] || exit 1
done
