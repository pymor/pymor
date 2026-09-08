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

    [[ -d ~/.ssh ]] || mkdir -p  ~/.ssh
    chmod 700 ~/.ssh
    ssh-keyscan -H github.com >> ~/.ssh/known_hosts
    ssh-keyscan -H docs-ng.pymor.org >> ~/.ssh/known_hosts
}
init_ssh

PYMOR_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})" ; cd ../../ ; pwd -P )"
cd "${PYMOR_ROOT}"


if [[ "x${CI_COMMIT_REF_SLUG}" == "x" ]] ; then
  CI_COMMIT_REF_SLUG=$(slugify ${CI_COMMIT_REF_NAME})
fi
SLUG=${CI_COMMIT_REF_SLUG/github\/PUSH_/from_fork__}

# we get the already built html documentation as an artefact from an earlier build stage
touch ${PYMOR_ROOT}/docs/_build/html/
rsync -a --delete ${PYMOR_ROOT}/docs/_build/html/ docs@docs-ng.pymor.org:${SLUG}
