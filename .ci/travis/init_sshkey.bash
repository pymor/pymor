#!/bin/bash

set -e
set -u
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

KEY=${1}
IV=${2}
RSAKEY=${3}.rsa.enc
pushd ${DIR}
[[ -d ~/.ssh ]] || mkdir -p  ~/.ssh
ssh-keyscan -H github.com >> ~/.ssh/known_hosts
openssl aes-256-cbc -K ${KEY} -iv ${IV} -in ${RSAKEY} -out ~/.ssh/id_rsa -d
chmod 600 ~/.ssh/id_rsa
popd
