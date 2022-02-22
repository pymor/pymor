#!/bin/bash

THIS_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ; pwd -P )"
source ${THIS_DIR}/common_test_setup.bash

# as a workaround intermittent MPI finalize errors which we
# cannot seem to directly affect, we save the intermediate
# pytest exit code in a file and check that afterwards
# while ignoring the mpirun result itself
xvfb-run -a mpirun --timeout 1200 --mca btl self,vader -n 2 python -u -m coverage run --rcfile=setup.cfg \
  --parallel-mode src/pymortests/mpi_run_demo_tests.py || true
[[ "$(cat pytest.mpirun.success)" == "True" ]] || exit 127

coverage combine
# the test_thermalblock_ipython results in '(builtin)' missing which we "--ignore-errors"
coverage xml --ignore-errors
