#!/bin/bash

# any failure here should fail the whole test
set -eux

function init_ssh {
    which ssh-agent || ( apt-get update -y && apt-get install openssh-client git rsync -y ) || \
      apk --update add openssh-client git rsync

    eval $(ssh-agent -s)
    echo "$DOCS_DEPLOY_KEY" | tr -d '\r' | ssh-add - > /dev/null

    [[ -d ~/.ssh ]] || mkdir -p  ~/.ssh
    chmod 700 ~/.ssh
    ssh-keyscan -H github.com >> ~/.ssh/known_hosts
}
init_ssh

PYMOR_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})" ; cd ../../ ; pwd -P )"
cd "${PYMOR_ROOT}"


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
(git diff --quiet && git diff --staged --quiet) || \
  git commit -m "Updated docs for ${CI_COMMIT_REF_NAME}"

${PYMOR_ROOT}/.ci/gitlab/docs_makeindex.py ${REPO_DIR}
git add list.html
(git diff --quiet && git diff --staged --quiet) || \
  git commit -m "Updated index for ${CI_COMMIT_REF_NAME}"

git push || (git pull --rebase && git push )

[[ "${SLUG}" != "main" ]] && git checkout -b ${SLUG}
rm -rf ${REPO_DIR}/.binder
mkdir ${REPO_DIR}/.binder

if [ -v CI_COMMIT_TAG ] ; then
	IMAGE_TAG=$CI_COMMIT_TAG
elif [ -v CI_COMMIT_BRANCH ] ; then
	if [ "${CI_COMMIT_BRANCH}" = main ] ; then
		IMAGE_TAG=main
	else
		IMAGE_TAG="${CI_CURRENT_IMAGE_TAG}"
	fi
else
	IMAGE_TAG="${CI_CURRENT_IMAGE_TAG}"
fi
# cp ${PYMOR_ROOT}/requirements-ci.txt ${REPO_DIR}/.binder/requirements.txt
# echo "python-3.10" > ${REPO_DIR}/.binder/runtime.txt
# this needs to go into the repo root, not the subdir!
sed -e "s;BINDERIMAGE;zivgitlab.wwu.io/pymor/pymor/ci-current:${IMAGE_TAG};g" -e "s;SLUG;${SLUG};g" \
    -e "s;PYMOR_COMMIT;${CI_COMMIT_SHA};g" \
	${PYMOR_ROOT}/docker/Dockerfile.binder.tocopy > ${REPO_DIR}/.binder/Dockerfile

# for binder the notebooks need to exist alongside their .rst version
cd ${TARGET_DIR}
for nb in $(find ./_downloads/ -name "*.ipynb") ; do
  # cannot use symlinks here due to github pages limitation
  cp -a ${nb} .
done

git add ${TARGET_DIR}/*ipynb
git add ${REPO_DIR}/.binder/

(git diff --quiet && git diff --staged --quiet) || \
  git commit -am "Binder setup for ${CI_COMMIT_REF_NAME}"

git push origin ${SLUG} -f
