#!/bin/bash

source ${CI_PROJECT_DIR}/.ci/gitlab/init_sshkey.bash
init_ssh

PYMOR_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})" ; cd ../../ ; pwd -P )"
cd "${PYMOR_ROOT}"

# any failure here should fail the whole test
set -eux

REPO=git@github.com:pymor/docs.git
REPO_DIR=${CI_PROJECT_DIR}/repo
if [[ "x${CI_COMMIT_REF_SLUG}" == "x" ]] ; then
  CI_COMMIT_REF_SLUG=$(slugify ${CI_COMMIT_REF_NAME})
fi
# this must match PYMOR_ROOT/docs/source/conf.py:try_on_binder_branch
SLUG=${CI_COMMIT_REF_SLUG/github\/PUSH_/from_fork__}
TARGET_DIR=${REPO_DIR}/${SLUG}

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

[[ "${SLUG}" != "main" ]] && git checkout -b ${SLUG}
rm -rf ${REPO_DIR}/.binder
mkdir ${REPO_DIR}/.binder

# this needs to go into the repo root, not the subdir!
sed -e "s;BINDERIMAGE;${BINDERIMAGE};g" -e "s;SLUG;${SLUG};g" \
	${PYMOR_ROOT}/.ci/gitlab/Dockerfile.binder.tocopy > ${REPO_DIR}/.binder/Dockerfile

# for binder the notebooks need to exist alongside their .rst version
cd ${TARGET_DIR}
for nb in $(find ./_downloads/ -name "*.ipynb") ; do
  ln -s ${nb}
done

git add ${TARGET_DIR}/*ipynb
git add ${REPO_DIR}/.binder/

git commit -am "Binder setup for ${CI_COMMIT_REF_NAME}"

# makes sure the binder infra is usable
repo2docker --user-id 2000 --user-name juno --no-run --debug .

git push origin ${SLUG} -f
