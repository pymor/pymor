#!/bin/bash

THIS_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ; pwd -P )"
source ${THIS_DIR}/common_test_setup.bash

xvfb-run -a mpirun -n 2 coverage run --rcfile=setup.cfg --parallel-mode src/pymortests/mpi_run_demo_tests.py
coverage combine
