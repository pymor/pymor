#!/bin/bash

# exit early if the var is not and should not be available
if [ x"${encrypted_aaee34775583_key}" == x ] ; then
    exit $([ "${TRAVIS_REPO_SLUG}" != "pymor/pymor" ])
fi

set -e
set -u

PYMOR_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})" ; cd ../../ ; pwd -P )"
cd "${PYMOR_ROOT}"

./.ci/travis/init_sshkey.bash "${encrypted_aaee34775583_key}" "${encrypted_aaee34775583_iv}" \
    ${PYMOR_ROOT}/.ci/travis/logs_deploy_key

TESTLOGS_URL="git@github.com:pymor/pymor-testlogs.git"
LOGS_DIR="${HOME}/pymor_logs"
BRANCH=${TRAVIS_BRANCH}
if [ "x${TRAVIS_PULL_REQUEST}" != "xfalse" ] ; then
    BRANCH=PR_${TRAVIS_PULL_REQUEST}_to_${BRANCH}
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
    cp ${PYMOR_ROOT}/${RESULT_FN} ${PYMOR_ROOT}/test_timings.csv ${TARGET_DIR}/
    printenv | \grep -v encrypted | \grep -v TOKEN | sort > ${TARGET_DIR}/env

    git add ${TARGET_DIR}/*
    git config user.name "pyMOR Bot"
    git config user.email "travis@pymor.org"
    git commit -m "Testlogs for Job ${TRAVIS_JOB_NUMBER} - ${TRAVIS_COMMIT_RANGE}"
    git push -q --set-upstream origin ${BRANCH}

fi
