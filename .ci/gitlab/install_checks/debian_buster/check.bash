#!/bin/bash
set -eu

cp ${CI_PROJECT_DIR}/.ci/gitlab/install_checks/ci.pip.conf /etc/pip.conf

# first section should ideally only needs minimal setup for out extension modules to compile
apt update
apt install -y gcc python3-dev
pip3 install -U pip
pip install ${CI_PROJECT_DIR}[full]

# second section gets additional setup for slycot, mpi4py etc
apt install -y libopenblas-dev gfortran libopenmpi-dev gcc cmake make
pip install -r ${CI_PROJECT_DIR}/requirements-optional.txt
