#!/bin/bash

THIS_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ; pwd -P )"
source ${THIS_DIR}/common_test_setup.bash

# pymor checks if this file's owner uid matches with the interpreter executor's
ME=$(id -u)
chown ${ME} docs/source/pymor_defaults.py

make docs
