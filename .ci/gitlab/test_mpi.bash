#!/bin/bash

THIS_DIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ; pwd -P )"
source ${THIS_DIR}/common_test_setup.bash

# this allows us to run in containers that cannot use ptrace
export OMPI_MCA_btl_vader_single_copy_mechanism=none

# as a workaround intermittent MPI finalize errors which we
# cannot seem to directly affect, we save the intermediate
# pytest exit code in a file and check that afterwards
# while ignoring the mpirun result itself
xvfb-run -a mpirun --allow-run-as-root --timeout 1200 --mca btl self,vader -n 2 \
	coverage run --source=src --parallel-mode \
	src/pymortests/mpi_run_demo_tests.py || true

for fn in ./.mpirun_*/pytest.mpirun.success ; do
  [[ "$(cat ${fn})" == "True" ]] || exit 127
done

coverage combine
_coverage_xml
