#!/bin/bash

set -e
set -u
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

function init_ssh {
    which ssh-agent || ( apt-get update -y && apt-get install openssh-client git rsync -y )

    eval $(ssh-agent -s)
    echo "$WHEELS_DEPLOY_KEY" | tr -d '\r' | ssh-add - > /dev/null
    echo "$TESTLOG_DEPLOY_KEY" | tr -d '\r' | ssh-add - > /dev/null

    [[ -d ~/.ssh ]] || mkdir -p  ~/.ssh
    chmod 700 ~/.ssh
    ssh-keyscan -H github.com >> ~/.ssh/known_hosts
}
