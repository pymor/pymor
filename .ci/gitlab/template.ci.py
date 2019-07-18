#!/usr/bin/env python3

tpl = '''# THIS FILE IS AUTOGENERATED -- DO NOT EDIT #
#   Edit and Re-run .ci/gitlab/template.ci.py instead       #

stages:
  - test
  - check install
  - deploy

.test_base:
    retry:
        max: 2
        when:
            - runner_system_failure
            - stuck_or_timeout_failure
            - api_failure
    only: ['branches', 'tags', 'triggers', 'merge-requests']

.pytest:
    extends: .test_base
    script: .ci/gitlab/script.bash
    tags:
      - long execution time
    environment:
        name: unsafe
    after_script:
      - .ci/gitlab/after_script.bash
    artifacts:
        name: "$CI_JOB_STAGE-$CI_COMMIT_REF_SLUG"
        expire_in: 3 months
        paths:
            - src/pymortests/testdata/check_results/*/*_changed
            - coverage.xml
        reports:
            junit: test_results.xml

numpy 3.6:
    extends: .pytest
    image: pymor/testing:3.6
    stage: test
    variables:
        PYMOR_PYTEST_MARKER: "numpy"

docs:
    extends: .test_base
    image: pymor/testing:3.6
    stage: test
    script: .ci/gitlab/test_docs.bash
    artifacts:
        name: "$CI_JOB_STAGE-$CI_COMMIT_REF_SLUG"
        expire_in: 3 months
        paths:
            - docs/_build/html

{%- for py, m in matrix %}
{{m}} {{py}}:
    extends: .pytest
    image: pymor/testing:{{py}}
    stage: test
    variables:
        PYMOR_PYTEST_MARKER: "{{m}}"
{%- endfor %}

{# note: only Vanilla and numpy runs generate coverage or test_results so we can skip others entirely here #}
.submit:
    extends: .test_base
    environment:
        name: safe
    except:
        - /^github\/PR_.*$/
    stage: deploy
    script: .ci/gitlab/submit.bash

{%- for py, m in matrix if m == 'Vanilla' %}
submit {{m}} {{py}}:
    extends: .submit
    image: pymor/python:{{py}}
    dependencies:
        - {{m}} {{py}}
    variables:
        PYMOR_PYTEST_MARKER: "{{m}}"
{%- endfor %}

submit numpy 3.6:
    extends: .submit
    image: pymor/python:3.6
    dependencies:
        - numpy 3.6
    variables:
        PYMOR_PYTEST_MARKER: "numpy"

# this step makes sure that on older python our install fails with
# a nice message ala "python too old" instead of "SyntaxError"
verify setup.py:
    extends: .test_base
    image: python:3.5-alpine
    stage: deploy
    script:
        - python setup.py egg_info

.docker-in-docker:
    tags:
      - docker-in-docker
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
        - pip3 install jinja2 jupyter-repo2docker docker-compose
        - 'export SHARED_PATH="${CI_PROJECT_DIR}/shared"'
        - mkdir -p ${SHARED_PATH}
    services:
        - docker:dind
    environment:
        name: unsafe

{%- for OS in testos %}
pip {{OS.replace('_', ' ')}}:
    extends: .docker-in-docker
    stage: deploy
    script: docker build -f .ci/docker/install_checks/{{OS}}/Dockerfile .
{% endfor %}

# this should ensure binderhubs can still build a runnable image from our repo
repo2docker:
    extends: .docker-in-docker
    stage: deploy
    variables:
        IMAGE: ${CI_REGISTRY_IMAGE}/binder:${CI_COMMIT_REF_SLUG}
        CMD: "jupyter nbconvert --to notebook --execute /pymor/.ci/ci_dummy.ipynb"
        USER: juno
    script:
        - repo2docker --user-id 2000 --user-name ${USER} --no-run --debug --image-name ${IMAGE} .
        - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
        - docker run ${IMAGE} ${CMD}
        - docker push ${IMAGE}
        - cd .binder
        - docker-compose build
        - docker-compose run jupyter ${CMD}

.wheel:
    extends: .docker-in-docker
    stage: deploy
    only: ['branches', 'tags', 'triggers']
    variables:
        TEST_OS: "{{ ' '.join(testos) }}"
    artifacts:
        paths:
        # cannot use exported var from env here
        - ${CI_PROJECT_DIR}/shared/pymor*manylinux*whl
        expire_in: 1 week

{%- for PY in pythons %}
{%- for ML in [1, 2010] %}
wheel {{ML}} {{PY}}:
    extends: .wheel
    variables:
        PYVER: "{{PY}}"
    script: bash .ci/gitlab/wheels.bash {{ML}}
{% endfor %}
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
marker = ["Vanilla", "PIP_ONLY", "NOTEBOOKS"]
with open(os.path.join(os.path.dirname(__file__), 'ci.yml'), 'wt') as yml:
    matrix = list(product(pythons, marker))
    yml.write(tpl.render(matrix=matrix,testos=['debian_stretch', 'debian_buster', 'debian_testing', 'centos_7'], pythons=pythons))
