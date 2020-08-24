#!/bin/bash
set -eu

cp ${CI_PROJECT_DIR}/.ci/gitlab/install_checks/ci.pip.conf /etc/pip.conf

# first section should ideally only need minimal setup for out extension modules to compile
yum install -y gcc python3-devel
pip3 install -U pip
pip install ${CI_PROJECT_DIR}[full]

# second section gets additional setup for slycot, mpi4py etc
export CC=/usr/lib64/openmpi/bin/mpicc
yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
yum install -y openmpi-devel openblas-devel cmake make gcc-gfortran gcc-c++
pip3 install -r ${CI_PROJECT_DIR}/requirements-optional.txt
