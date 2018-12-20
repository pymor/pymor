ARG tag
FROM pymor/deploy_check:${tag}

ADD . /wheelhouse
RUN pip3 install -f /wheelhouse pymor[full]
