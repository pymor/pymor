ARG tag
FROM pymor/deploy_checks:${tag}

ADD . /wheelhouse
RUN pip3 install -f /wheelhouse pymor[full]
