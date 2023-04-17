#!/bin/bash
set -eu

# this sets pip to install from the mirror container to reduce internet download time
\cp -f ${CI_PROJECT_DIR}/.ci/gitlab/install_checks/ci.pip.conf /etc/pip.conf

# first section should only need minimal setup
python3 -m pip install ${CI_PROJECT_DIR}[full]

# second section gets additional setup for slycot, mpi4py etc
export CC=/usr/lib64/openmpi/bin/mpicc
yum install -y python3-devel openmpi-devel openblas-devel cmake make gcc-gfortran gcc-c++
python3 -m pip install -r ${CI_PROJECT_DIR}[all]
