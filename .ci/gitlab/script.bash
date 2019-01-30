#!/bin/bash

TRAVIS_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ; cd ../travis/ ; pwd -P )"

export CODECOV_TOKEN="${PYMOR_CODECOV_TOKEN}" \
    TRAVIS_REPO_SLUG="${CI_PROJECT_PATH}" \
    TRAVIS_COMMIT="${CI_COMMIT_SHA}"

if [ "x${CI_MERGE_REQUEST_ID}" == "x" ] ; then
    export TRAVIS_PULL_REQUEST=false
else
    export TRAVIS_PULL_REQUEST=${CI_MERGE_REQUEST_ID}
fi
export PYTHONPATH=${CI_PROJECT_DIR}/src:${PYTHONPATH}

"${TRAVIS_DIR}/script.bash"

