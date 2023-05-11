#!/bin/bash

PYMOR_ROOT="$(cd "$(dirname "$0")" && cd ../../ && pwd -P )"
cd "${PYMOR_ROOT}"

set -eux

CI_CURRENT_IMAGE_TAG=$(sha256sum requirements-ci-current.txt | cut -d ' ' -f 1)
CI_OLDEST_IMAGE_TAG=$(sha256sum requirements-ci-oldest.txt | cut -d ' ' -f 1)
CI_FENICS_IMAGE_TAG=$(sha256sum requirements-ci-fenics.txt | cut -d ' ' -f 1)

echo "CI_CURRENT_IMAGE_TAG=${CI_CURRENT_IMAGE_TAG}" > out.env
echo "CI_OLDEST_IMAGE_TAG=${CI_OLDEST_IMAGE_TAG}" >> out.env
echo "CI_FENICS_IMAGE_TAG=${CI_FENICS_IMAGE_TAG}" >> out.env

if ./.ci/gitlab/check_image_in_registry.py ci-current $CI_CURRENT_IMAGE_TAG ; then
	echo "CI_BUILD_CURRENT_IMAGE=no" >> out.env
else
	echo "CI_BUILD_CURRENT_IMAGE=yes" >> out.env
fi

if ./.ci/gitlab/check_image_in_registry.py ci-oldest $CI_OLDEST_IMAGE_TAG ; then
	echo "CI_BUILD_OLDEST_IMAGE=no" >> out.env
else
	echo "CI_BUILD_OLDEST_IMAGE=yes" >> out.env
fi

if ./.ci/gitlab/check_image_in_registry.py ci-fenics $CI_FENICS_IMAGE_TAG ; then
	echo "CI_BUILD_FENICS_IMAGE=no" >> out.env
else
	echo "CI_BUILD_FENICS_IMAGE=yes" >> out.env
fi

cat out.env

# makes sure mailmap is up-to-date
./.ci/gitlab/check_mailmap.py ./.mailmap
