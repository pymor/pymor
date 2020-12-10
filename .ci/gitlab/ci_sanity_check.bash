#!/bin/bash

function docker_tag_exists() {
  tag=$2
  image=$1
  private_token=${GITLAB_API_RO}
  project=2758
  baseurl="https://zivgitlab.uni-muenster.de/api/v4/projects/$project/registry"
  repo_id=$( curl --silent --header "PRIVATE-TOKEN: $private_token" "${baseurl}/repositories?per_page=100" | jq --arg image $image '.[] | select(.name==$image) | .id')
  name=$( curl --silent --header "PRIVATE-TOKEN: $private_token" "${baseurl}/repositories/$repo_id/tags/$tag" | jq -r '.name' )
  [[ "$name" == "$tag" ]] || (echo "Repoid ${repo_id} -- image ${image} -- name ${name} -- tag ${tag}"; exit -1)
}

PYMOR_ROOT="$(cd "$(dirname "$0")" && cd ../../ && pwd -P )"

cd "${PYMOR_ROOT}"

set -eux

PYTHONS="${1}"
MANYLINUXS="${2}"
# make sure CI setup is current
./.ci/gitlab/template.ci.py && git diff --exit-code .ci/gitlab/ci.yml
# check if requirements files are up-to-date
./dependencies.py && git diff --exit-code requirements* pyproject.toml

source ${PYMOR_ROOT}/.env
for py in ${PYTHONS} ; do
  docker_tag_exists pymor/testing_py${py} ${CI_IMAGE_TAG}
  docker_tag_exists pymor/pypi-mirror_stable_py${py} ${PYPI_MIRROR_TAG}
  docker_tag_exists pymor/pypi-mirror_oldest_py${py} ${PYPI_MIRROR_TAG}
  for ml in ${MANYLINUXS}; do
    docker_tag_exists pymor/wheelbuilder_manylinux${ml}_py${py} ${PYPI_MIRROR_TAG}
  done
done

for script in ${PYMOR_ROOT}/.ci/gitlab/test* ; do
  [[ -x "${script}" ]] || exit 1
done
