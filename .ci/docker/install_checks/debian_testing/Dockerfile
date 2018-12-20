FROM pymor/deploy_checks:debian_testing

ADD . /src
RUN pip3 install /src[full]

RUN apt install -y libopenblas-dev gfortran libopenmpi-dev gcc
RUN pip3 install -r /src/requirements-optional.txt
