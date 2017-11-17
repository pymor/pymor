#!/bin/bash

set -eu

BUILDER_WHEELHOUSE=/tmp/wheelhouse
if [ ${TRAVIS_BRANCH} = master ] ; then
    REPODIR=${HOME}/wheels
else
    REPODIR=${HOME}/wheels/branches/${TRAVIS_BRANCH}/
fi

PYMOR_ROOT="$(cd "$(dirname ${BASH_SOURCE[0]})" ; cd ../../ ; pwd -P )"
cd "${PYMOR_ROOT}"

rm -rf ~/.ssh
ls -l ${PYMOR_ROOT}/.ci/travis/wheels.deploy.key.rsa.enc

./.ci/travis/init_sshkey.bash "${encrypted_a599472c800f_key}" "${encrypted_a599472c800f_iv}" \
    ${PYMOR_ROOT}/.ci/travis/wheels.deploy.key

mkdir -p ${BUILDER_WHEELHOUSE}
git clone git@github.com:pymor/wheels.pymor.org ${REPODIR}
for py in 2.7 3.5 3.6 ; do
    BUILDER_IMAGE=pymor/manylinux:py${py}
    git clean -xdf
    docker pull ${BUILDER_IMAGE} 1> /dev/null
    docker run --rm  -t -e LOCAL_USER_ID=$(id -u)  \
        -v ${BUILDER_WHEELHOUSE}:/io/wheelhouse \
        -v ${PYMOR_ROOT}:/io/pymor ${BUILDER_IMAGE} /usr/local/bin/build-wheels.sh 1> /dev/null
    rsync -a ${BUILDER_WHEELHOUSE}/pymor*manylinux*.whl ${REPODIR}/
done

set +u
pip install jinja2

cd ${REPODIR}
find . -name "*.whl" | xargs git add
make index
git config user.name "pyMOR Bot"
git config user.email "travis@pymor.org"
git commit -am "[deploy] wheels for ${TRAVIS_COMMIT}"
git pull --rebase
git push
