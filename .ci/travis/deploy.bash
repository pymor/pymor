#!/bin/bash

set -e
REPODIR=${HOME}/wheels
./.ci/travis/init_sshkey.bash "${encrypted_a599472c800f_key}" "${encrypted_a599472c800f_iv}" \
    ${PYMOR_ROOT}/.ci/travis/wheels.deploy

git clone git@github.com:pymor/wheels.pymor.org ${REPODIR}
for py in 2.7 3.5 3.6 ; do
    BUILDER_IMAGE=pymor/manylinux:py${REV}
    docker run --rm  -t -e LOCAL_USER_ID=$(id -u)  \
		-v ${PWD}:/io ${BUILDER_IMAGE} /usr/local/bin/build-wheels.sh
done

if [ ${TRAVIS_BRANCH} = master ] ; then
    cp ${PWD}/wheelhouse/pymor*manylinux*.whl ${REPODIR}/
else
    cp ${PWD}/wheelhouse/pymor*manylinux*.whl ${REPODIR}/${TRAVIS_BRANCH}
fi

cd ${REPODIR}
find . -name "*.whl" | xargs git add
git commit -m "[deploy] wheels for ${TRAVIS_COMMIT}"
git push
