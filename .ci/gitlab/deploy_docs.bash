#!/bin/bash

if [ "x${CI_MERGE_REQUEST_ID}" == "x" ] ; then
    export PULL_REQUEST=false
else
    export PULL_REQUEST=${CI_MERGE_REQUEST_ID}
fi

export PYTHONPATH=${CI_PROJECT_DIR}/src:${PYTHONPATH}
SUDO="sudo -E -H"
PYMOR_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})" ; cd ../../ ; pwd -P )"
cd "${PYMOR_ROOT}"

# any failure here should fail the whole test
set -eux

export USER=pymor
make dockerdocs

docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
docker pull ${IMAGE}
container=$(docker create --entrypoint / ${IMAGE})

mkdir -p public/${CI_COMMIT_REF_SLUG}/
rm -rf public/${CI_COMMIT_REF_SLUG}/
docker cp ${container}:/public/ public/

rsync -a docs/_build/html/ public/${CI_COMMIT_REF_SLUG}/
cp -r docs/public_root/* public/
docker build -t ${IMAGE} -f .ci/docker/docs/Dockerfile .
docker push ${IMAGE}
