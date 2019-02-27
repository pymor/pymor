FROM pymor/deploy_checks:debian_buster

ADD . /src
RUN apt update && apt install -y gcc python3-dev && pip3 install -U pip && pip install /src[full]

RUN apt install -y libopenblas-dev gfortran libopenmpi-dev gcc
RUN pip install -r /src/requirements-optional.txt
