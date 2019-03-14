#!/usr/bin/env python3

tpl = '''# THIS FILE IS AUTOGENERATED -- DO NOT EDIT #
#   Edit and Re-run .ci/gitlab/template.ci.py instead       #

stages:
  - test
  - check install
  - deploy

.pytest:
    retry:
        max: 2
        when:
            - runner_system_failure
            - stuck_or_timeout_failure
            - api_failure
    script: .ci/gitlab/script.bash
    environment:
        name: unsafe
    after_script:
      - .ci/gitlab/after_script.bash
    only: ['branches', 'tags', 'triggers', 'merge-requests']
    artifacts:
        name: "$CI_JOB_STAGE-$CI_COMMIT_REF_SLUG"
        expire_in: 3 months
        paths:
            - src/pymortests/testdata/check_results/*/*_changed
            - .coverage
        reports:
            junit: test_results.xml

3.6_numpy:
    extends: .pytest
    image: pymor/testing:3.6
    stage: test
    variables:
        PYMOR_PYTEST_MARKER: "numpy"
        DOCKER_TAG: "3.6"

{%- for py, m in matrix %}
{{py}}_{{m}}:
    extends: .pytest
    image: pymor/testing:{{py}}
    stage: test
    variables:
        PYMOR_PYTEST_MARKER: "{{m}}"
        DOCKER_TAG: "{{py}}"
{%- endfor %}

{%- for py, m in matrix if m != 'MPI' %}
{{py}}_{{m}}_submit:
    extends: .pytest
    image: pymor/testing:{{py}}
    stage: deploy
    dependencies:
        - {{py}}_{{m}}
    environment:
        name: safe
    except:
        - github/PR_.*
    variables:
        PYMOR_PYTEST_MARKER: "{{m}}"
        DOCKER_TAG: "{{py}}"
    script: .ci/gitlab/submit.bash
{%- endfor %}

.docker-in-docker:
    retry:
        max: 2
        when:
            - always
    image: docker:stable
    variables:
        DOCKER_HOST: tcp://docker:2375/
        DOCKER_DRIVER: overlay2
    before_script:
        - apk --update add openssh-client rsync git file bash python3
        - pip3 install jinja2
        - 'export SHARED_PATH="${CI_PROJECT_DIR}/shared"'
        - mkdir -p ${SHARED_PATH}
    services:
        - docker:dind
    environment:
        name: unsafe

{%- for OS in testos %}
{{OS}}_pip:
    extends: .docker-in-docker
    stage: deploy
    script: docker build -f .ci/docker/install_checks/{{OS}}/Dockerfile .
{% endfor %}

{%- for PY in pythons %}
{{PY}}_wheel:
    extends: .docker-in-docker
    stage: deploy
    only: ['branches', 'tags', 'triggers']
    variables:
        PYVER: "{{PY}}"
        TEST_OS: "{{ ' '.join(testos) }}"
    script: bash .ci/gitlab/wheels.bash
    artifacts:
        paths:
        # cannot use exported var from env here
        - ${CI_PROJECT_DIR}/shared/pymor*manylinux*whl
        expire_in: 1 week
{% endfor %}

# THIS FILE IS AUTOGENERATED -- DO NOT EDIT #
#   Edit and Re-run .ci/gitlab/template.ci.py instead       #

'''


import os
import jinja2
import sys
from itertools import product
tpl = jinja2.Template(tpl)
pythons = ['3.6', '3.7']
marker = [None, "PIP_ONLY", "MPI"]
with open(os.path.join(os.path.dirname(__file__), 'ci.yml'), 'wt') as yml:
    matrix = list(product(pythons, marker))
    yml.write(tpl.render(matrix=matrix,testos=['debian_stretch', 'debian_buster', 'debian_testing', 'centos_7'], pythons=pythons))
