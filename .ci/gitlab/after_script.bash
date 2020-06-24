#!/bin/bash

# exit early if we do not want to upload tests
if [ ${CI_MERGE_REQUEST_SOURCE_PROJECT_URL} != ${CI_MERGE_REQUEST_PROJECT_URL} ] ; then
    exit 0
fi

set -e
set -u

PYMOR_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})" ; cd ../../ ; pwd -P )"
cd "${PYMOR_ROOT}"

source ./.ci/gitlab/init_sshkey.bash
init_ssh

TESTLOGS_URL="git@github.com:pymor/pymor-testlogs.git"
LOGS_DIR="${HOME}/pymor_logs"
BRANCH=${CI_COMMIT_REF_NAME}
if [ "x${CI_MERGE_REQUEST_ID}" != "xfalse" ] ; then
    BRANCH=PR_${CI_MERGE_REQUEST_ID}_to_${BRANCH}
fi
PYMOR_VERSION=$(python -c 'import pymor;print(pymor.__version__)')
RESULT_FN=test_results.xml
PY_VER=$(python -c 'import platform;print(platform.python_version())')

if [ "${PYMOR_PYTEST_MARKER}" == "None" ] ; then

    git clone  ${TESTLOGS_URL}  ${LOGS_DIR}
    cd ${LOGS_DIR}
    # check if branch exists, see http://stackoverflow.com/questions/8223906/how-to-check-if-remote-branch-exists-on-a-given-remote-repository
    if [ `git ls-remote --heads ${TESTLOGS_URL} ${BRANCH} | wc -l` -ne 0 ] ; then
        git checkout ${BRANCH}
        else
        git checkout -b ${BRANCH}
    fi

    TARGET_DIR=${LOGS_DIR}/${BRANCH}/${PY_VER}/${PYMOR_VERSION}/
    [[ -d "${TARGET_DIR}" ]]  || mkdir -p ${TARGET_DIR}
    cp ${PYMOR_ROOT}/${RESULT_FN} ${TARGET_DIR}/
    printenv | \grep -v encrypted | \grep -v TOKEN | sort > ${TARGET_DIR}/env

    git add ${TARGET_DIR}/*
    git config user.name "pyMOR Bot"
    git config user.email "gitlab-ci@pymor.org"
    git commit -m "Testlogs for Job ${CI_JOB_ID} - ${CI_COMMIT_BEFORE_SHA} ... ${CI_COMMIT_SHA}"
    git push -q --set-upstream origin ${BRANCH}

fi
