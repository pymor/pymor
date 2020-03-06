#!/bin/bash

THIS_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ; pwd -P )"
source ${THIS_DIR}/common_test_setup.bash

cd src/pymordemos/minimal_cpp_demo
cmake .
make
xvfb-run -a python -c 'import runpy; runpy.run_module("demo")'
