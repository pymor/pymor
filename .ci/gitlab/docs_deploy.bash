#!/bin/bash

# any failure here should fail the whole test
set -eux

function init_ssh {
    which ssh-agent || ( apt-get update -y && apt-get install openssh-client git rsync -y ) || \
      apk --update add openssh-client git rsync

    eval $(ssh-agent -s)

    chmod 600 "$DOCS_DEPLOY_KEY"
    ssh-add "$DOCS_DEPLOY_KEY"

    chmod 600 "$DOCS_DEPLOY_KEY_ZIV"
    ssh-add "$DOCS_DEPLOY_KEY_ZIV"

    chmod 600 "$BINDER_DEPLOY_KEY"

    [[ -d ~/.ssh ]] || mkdir -p  ~/.ssh
    chmod 700 ~/.ssh
    ssh-keyscan -H github.com >> ~/.ssh/known_hosts
    ssh-keyscan -H docs-ng.pymor.org >> ~/.ssh/known_hosts
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

ssh docs@docs-ng.pymor.org


## binder
cd "${PYMOR_ROOT}"
git clone --depth 2 git@github.com:pymor/binder.git binder_repo
cd binder_repo
git config user.name "pyMOR Bot"
git config user.email "gitlab-ci@pymor.org"

if [[ "${SLUG}" != "main" ]] ; then
	git checkout --orphan ${SLUG}
	git rm -rf .
else
	git rm -rf ./.binder
fi
mkdir .binder

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

NOTEBOOK_URLS=""
pushd ${PYMOR_ROOT}/docs/_build/html/
for nb in $(find _downloads/ -name "*.ipynb") ; do
	NOTEBOOK_URLS="${NOTEBOOK_URLS} https://docs.pymor.org/${SLUG}/${nb}"
done
popd

# this needs to go into the repo root, not the subdir!
sed -e "s;BINDERIMAGE;zivgitlab.wwu.io/pymor/pymor/ci-current:${IMAGE_TAG};g" -e "s;SLUG;${SLUG};g" \
    -e "s;PYMOR_COMMIT;${CI_COMMIT_SHA};g" \
    -e "s;NOTEBOOK_URLS;${NOTEBOOK_URLS};g" \
	${PYMOR_ROOT}/docker/Dockerfile.binder.tocopy > .binder/Dockerfile

# for binder the notebooks need to exist alongside their .rst version
# cd ${TARGET_DIR}

# echo "python-3.11" > .binder/runtime.txt
# cp ${PYMOR_ROOT}/conda-linux-64.lock .binder/environment.yml
# echo "pymor @ git+https://github.com/pymor/pymor@${CI_COMMIT_SHA}" > .binder/requirements.txt

# git add ${REPO_DIR}/*ipynb
git add .binder/

(git diff --quiet && git diff --staged --quiet) || \
  git commit -am "Binder setup for ${CI_COMMIT_REF_NAME}"

# ensure that git uses the right deploy key
ssh-add -D
ssh-add "$BINDER_DEPLOY_KEY"
git push origin ${SLUG} -f
