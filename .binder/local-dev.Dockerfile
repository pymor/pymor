# the docker container for local dev just needs to pypi-mirror dependencies baked in
ARG PYPI_MIRROR_TAG=latest
ARG BASE=pymor/testing_py3.7:latest

FROM pymor/pypi-mirror_stable_py3.7:${PYPI_MIRROR_TAG} as dependencies
FROM ${BASE}
MAINTAINER rene.fritze@wwu.de

COPY --from=dependencies /pymor/downloads/* /tmp/dependencies/
RUN pip install /tmp/dependencies/*
