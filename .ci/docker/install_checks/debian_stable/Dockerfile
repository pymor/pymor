FROM debian:stable

RUN apt update -y && apt install -y python3-pip
ADD . /src
RUN pip3 install /src[full]

RUN apt install -y libopenblas-dev gfortran libopenmpi-dev gcc
RUN pip3 install -r /src/requirements-optional.txt
