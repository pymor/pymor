ARG PYVER=3.7
FROM pymor/testing:$PYVER

MAINTAINER rene.fritze@wwu.de

USER root
RUN useradd --shell /bin/bash -u 1000 -o -c "" -m pymor && \
    chown -R pymor /home/pymor
ARG PYMOR_JUPYTER_TOKEN
ENV JUPYTER_TOKEN=${PYMOR_JUPYTER_TOKEN} \
    PYTHONPATH=/pymor/src:${PYTHONPATH}
ADD requirements*.txt /tmp/
RUN pip install jupyterlab && \
    pip install -r /tmp/requirements-optional.txt
USER pymor
ENTRYPOINT []
WORKDIR /pymor/notebooks
