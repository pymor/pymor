FROM pymor/deploy_checks:centos_7

ADD . /src
RUN yum install -y gcc && source scl_source enable rh-python36  && pip3 install -U pip && pip install /src[full]

ENV CC=/usr/lib64/openmpi/bin/mpicc
RUN yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm && \
    yum install -y openmpi-devel openblas-devel
RUN source scl_source enable rh-python36  && pip3 install -r /src/requirements-optional.txt
