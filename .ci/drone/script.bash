#!/bin/bash

TRAVIS_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ; cd ../travis/ ; pwd -P )"

export CODECOV_TOKEN="${PYMOR_CODECOV_TOKEN}" \
    TRAVIS_REPO_SLUG="${DRONE_REPO_OWNER}/${DRONE_REPO_NAME}" \
    TRAVIS_COMMIT="${DRONE_COMMIT_SHA}"

if [ "${DRONE_BUILD_EVENT}" == "pull_request" ] ; then
    export TRAVIS_PULL_REQUEST=${DRONE_PULL_REQUEST}
else
    export TRAVIS_PULL_REQUEST=false
fi

"${TRAVIS_DIR}/script.bash"
