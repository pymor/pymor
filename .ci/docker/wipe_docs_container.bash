#!/usr/bin/env bash

PYMOR_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})" ; cd ../../ ; pwd -P )"
IMAGE=zivgitlab.wwu.io/pymor/pymor/docs:latest
tmpfile=$(mktemp /tmp/dockerfile.XXXXXX)
echo "FROM scratch" > ${tmpfile}
# Dockerfile cannot be empty
echo "COPY . /tmp" >> ${tmpfile}
docker build -t ${IMAGE} - < ${tmpfile}
docker push ${IMAGE}
rm ${tmpfile}
