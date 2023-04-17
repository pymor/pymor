#!/bin/bash
set -eu

# this sets pip to install from the mirror container to reduce internet download time
cp ${CI_PROJECT_DIR}/.ci/gitlab/install_checks/ci.pip.conf /etc/pip.conf

# first section should ideally only need minimal setup
python3 -m pip install ${CI_PROJECT_DIR}[full]

# second section gets additional setup for slycot, mpi4py etc
apt update
apt install -y python3-dev libopenblas-dev gfortran libopenmpi-dev gcc g++ make cmake
python3 -m pip install -r ${CI_PROJECT_DIR}[all]
