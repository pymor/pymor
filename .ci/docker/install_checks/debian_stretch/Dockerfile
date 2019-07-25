FROM pymor/deploy_checks:debian_stretch

ADD . /src
# stretch ships python 3.5 therefore this check must fail
RUN /bin/bash -c "apt update && apt install -y gcc python3-dev && pip3 install -U pip && \
    pip install /src[full] |& grep -qF 'Package '\''pymor'\'' requires a different Python:'" 

RUN apt install -y libopenblas-dev gfortran libopenmpi-dev gcc
RUN pip install -r /src/requirements-optional.txt
