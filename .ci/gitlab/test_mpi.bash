#!/bin/bash

THIS_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ; pwd -P )"
source ${THIS_DIR}/common_test_setup.bash

xvfb-run -a mpirun --allow-run-as-root -n 2 python src/pymortests/mpi_run_demo_tests.py
