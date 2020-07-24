#!/bin/bash

source ${CI_PROJECT_DIR}/.ci/gitlab/init_sshkey.bash
init_ssh

PYMOR_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})" ; cd ../../ ; pwd -P )"
cd "${PYMOR_ROOT}"

# any failure here should fail the whole test
set -eux

REPO=git@github.com:pymor/docs.git
REPO_DIR=${CI_PROJECT_DIR}/repo
TARGET_DIR=${REPO_DIR}/${CI_COMMIT_REF_SLUG/github\/PUSH_/from_fork__}

git clone --depth 2 ${REPO} ${REPO_DIR}

rm -rf ${TARGET_DIR}
mkdir -p ${TARGET_DIR}

# we get the already built html documentation as an artefact from an earlier build stage
rsync -a ${PYMOR_ROOT}/docs/_build/html/ ${TARGET_DIR}
cd ${REPO_DIR}
git config user.name "pyMOR Bot"
git config user.email "gitlab-ci@pymor.org"
git add ${TARGET_DIR}
git commit -m "Updated docs for ${CI_COMMIT_REF_NAME}"

${PYMOR_ROOT}/.ci/gitlab/docs_makeindex.py ${REPO_DIR}
git add list.html
git commit -m "Updated index for ${CI_COMMIT_REF_NAME}" || echo "nothing to add"


git push || (git pull --rebase && git push )


du -sch ${REPO_DIR}/*
du -sch ${TARGET_DIR}/*
