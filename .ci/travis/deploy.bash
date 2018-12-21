#!/bin/bash

set -eu

BUILDER_WHEELHOUSE=/tmp/wheelhouse
REPODIR=${HOME}/wheels
PYMOR_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})" ; cd ../../ ; pwd -P )"
cd "${PYMOR_ROOT}"

sed -i -e 's;style\ \=\ pep440;style\ \=\ ci_wheel_builder;g' setup.cfg

rm -rf ~/.ssh

./.ci/travis/init_sshkey.bash "${encrypted_a599472c800f_key}" "${encrypted_a599472c800f_iv}" \
    ${PYMOR_ROOT}/.ci/travis/wheels.deploy.key

set -x
mkdir -p ${BUILDER_WHEELHOUSE}
git clone git@github.com:pymor/wheels.pymor.org ${REPODIR}
for py in 3.5 3.6 3.7 ; do
    BUILDER_IMAGE=pymor/wheelbuilder:py${py}
    git clean -xdf
    docker pull ${BUILDER_IMAGE} 1> /dev/null
    docker run --rm  -t -e LOCAL_USER_ID=$(id -u)  \
        -v ${BUILDER_WHEELHOUSE}:/io/wheelhouse \
        -v ${PYMOR_ROOT}:/io/pymor ${BUILDER_IMAGE} /usr/local/bin/build-wheels.sh
done

cp ${PYMOR_ROOT}/.ci/docker/deploy_checks/Dockerfile ${BUILDER_WHEELHOUSE}
for os in debian_stable debian_testing centos_7 ; do
    docker build --build-arg tag=${os} ${BUILDER_WHEELHOUSE}
done

for py in 3.5 3.6 3.7 ; do
    ${REPODIR}/add_wheels.py ${TRAVIS_BRANCH} ${BUILDER_WHEELHOUSE}/pymor*manylinux*.whl
done

set +u
pip install jinja2

cd ${REPODIR}
git config user.name "pyMOR Bot"
git config user.email "travis@pymor.org"
git commit -am "[deploy] wheels for ${TRAVIS_COMMIT}"
git pull --rebase
git push
