#!/bin/bash

THIS_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ; pwd -P )"
export PYMOR_HYPOTHESIS_PROFILE=dev
source ${THIS_DIR}/common_test_setup.bash

make docs
