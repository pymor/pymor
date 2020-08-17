FROM pymor/deploy_checks:debian_bullseye

ADD . /src
RUN apt update && apt install -y gcc g++ python3-dev && pip3 install -U pip && pip install /src[full]

RUN apt install -y libopenblas-dev gfortran libopenmpi-dev gcc g++
RUN pip install -r /src/requirements-optional.txt
