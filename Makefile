#!/usr/bin/env make

# customization points via makefile key-value arguments
#
# interpreter in images: 3.{6,7,8} currently available
# DOCKER_BASE_PYTHON=3.7
# test script executed with `docker_test`: mpi, notebooks_dir, oldest, vanilla, mpi, numpy_git, pip_installed
# PYMOR_TEST_SCRIPT=vanilla
# version pinned mirror to be used: stable or oldest
# PYPI_MIRROR=stable
# wheel check OS: debian_buster centos_8 debian_testing
# PYMOR_TEST_OS=debian_buster
# hypothesis profiles: dev, debug, ci, ci-pr, ci-large
# PYMOR_HYPOTHESIS_PROFILE=dev
#

THIS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
DOCKER_RUN=docker run -v $(THIS_DIR):/pymor --env-file  $(THIS_DIR)/.env
DOCKER_COMPOSE=CI_COMMIT_SHA=$(shell git log -1 --pretty=format:"%H") \
	docker-compose -f .binder/docker-compose.yml -p pymor
NB_DIR=notebooks
PANDOC_MAJOR=$(shell ( which pandoc && pandoc --version | head  -n1 | cut -d ' ' -f 2 | cut -d '.' -f 1)) || echo "pandoc missing")
ifeq ($(PANDOC_MAJOR),1)
	PANDOC_FORMAT=-f markdown_github
endif
# this loads $(ENV_FILE) as both makefile variables and into shell env
ENV_FILE?=.env
include $(ENV_FILE)
export $(shell sed 's/=.*//' $(ENV_FILE))

.PHONY: docker README.html pylint test docs

all:
	./dependencies.py

# I want HTML (to preview the formatting :))
README.html: README.md
	pandoc $(PANDOC_FORMAT) -t html $< > $@

README: README.html

pep8:
	pep8 ./src

flake8:
	flake8 ./src

test:
	python setup.py pytest

jupyter:
	jupyter notebook --notebook-dir=$(NB_DIR) --NotebookApp.disable_check_xsrf=True

tutorials: NB_DIR="docs/_build/html"
tutorials: docs jupyter

full-test:
	@echo
	@echo "Ensuring that all required pytest plugins are installed ..."
	@echo "--------------------------------------------------------------------------------"
	@echo
	pip install pytest-flakes
	pip install pytest-pep8
	pip install pytest-cov
	@echo
	@echo "--------------------------------------------------------------------------------"
	@echo
	py.test --flakes --pep8 --cov=src/pymor --cov-report=html --cov-report=xml src/pymortests

docs:
	PYTHONPATH=${PWD}/src/:${PYTHONPATH} make -C docs html

template: docker_file
	./dependencies.py
	./.ci/gitlab/template.ci.py

# docker targets
docker_template: docker_file
	$(DOCKER_RUN) pymor/ci_sanity:$(CI_IMAGE_TAG) /pymor/dependencies.py
	$(DOCKER_RUN) pymor/ci_sanity:$(CI_IMAGE_TAG) /pymor/.ci/gitlab/template.ci.py

docker_file:
	 sed -e "s;CI_IMAGE_TAG;$(CI_IMAGE_TAG);g" -e "s;DOCKER_BASE_PYTHON;$(DOCKER_BASE_PYTHON);g" \
		 .binder/Dockerfile.in > .binder/Dockerfile

docker_image: docker_file
	$(DOCKER_COMPOSE) build

docker_docs: docker_image
	NB_DIR=notebooks $(DOCKER_COMPOSE) run docs ./.ci/gitlab/test_docs.bash

docker_run: docker_image
	$(DOCKER_COMPOSE) run --service-ports jupyter bash

docker_exec: docker_image
	$(DOCKER_COMPOSE) run --service-ports jupyter bash -l -c "${DOCKER_CMD}"

docker_tutorials: NB_DIR=docs/_build/html
docker_tutorials: docker_docs docker_jupyter

docker_test: docker_image
	PYMOR_PYTEST_MARKER=$(PYMOR_PYTEST_MARKER) $(DOCKER_COMPOSE) up pytest

docker_jupyter: docker_image
	NB_DIR=$(NB_DIR) $(DOCKER_COMPOSE) up jupyter
docker_wheel_check: docker_image
	PYMOR_TEST_OS=$(PYMOR_TEST_OS) $(DOCKER_COMPOSE) run --service-ports wheel_check bash
