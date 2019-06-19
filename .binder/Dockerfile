# the docker container for binder needs pymor installed entirely inside
# the container, local dev needs it in path from mounted src
# we trick docker into fulfilling both roles via a conditional ONBUILD
# if you want to use the local dev setup, see docker/docker-compose.yml
ARG PYVER=3.7
ARG BUILD_ENV=binder

FROM pymor/testing:$PYVER as image_binder
ONBUILD ADD . /pymor

FROM pymor/testing:$PYVER as image_dev
ONBUILD RUN echo "dev image uses mounted pymor" && mkdir /pymor
ONBUILD ENV PYTHONPATH=/pymor/src:${PYTHONPATH}

# select "base" image according to build arg
FROM image_${BUILD_ENV}
MAINTAINER rene.fritze@wwu.de

# binder wants to set the NB_ vars anyways, so we use it to service both setups
ARG NB_USER
ARG NB_UID
ARG PYMOR_JUPYTER_TOKEN

USER root
RUN useradd -d /home/pymor --shell /bin/bash -u ${NB_UID} -o -c "" -m ${NB_USER} && \
    chown -R ${NB_USER} /home/pymor /pymor/

# the vispy needs npm/nodejs installed for the jupter extensions
RUN curl -sL https://deb.nodesource.com/setup_10.x | bash - \
    &&  apt update && apt install nodejs
# at this point setup.py exists only in image_binder
# for the dev image it's only available via the mount at runtime
ADD requirements*.txt /tmp/
ADD .binder/ipython_kernel_config.py /etc/ipython/
ADD .binder/dev-entrypoint.bash /usr/local/bin/dev-entrypoint.bash
RUN /bin/bash -c "pip install jupyterlab && \
    pip install -r /tmp/requirements-optional.txt && \
    [[ -e /pymor/setup.py ]] && pip install /pymor || echo 'no install needed' && \
    jupyter nbextension enable --py widgetsnbextension && \
    jupyter nbextension enable --py vispy && \
    jupyter labextension install @jupyter-widgets/jupyterlab-manager && \
    jupyter labextension install vispy "
USER ${NB_USER}

ENV JUPYTER_TOKEN=${PYMOR_JUPYTER_TOKEN} \
    USER=${NB_USER} \
    HOME=/home/pymor

ENTRYPOINT []
WORKDIR /pymor/notebooks
