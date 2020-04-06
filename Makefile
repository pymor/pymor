
# customization points via environment variables
# 3.{6,7,8} supported currently
# DOCKER_BASE_PYTHON=3.7
# one of mpi, notebooks_dir, oldest, vanilla, mpi, numpy_git, pip_installed
# PYMOR_TEST_SCRIPT=vanilla
# stable or oldest
# PYPI_MIRROR=stable
# debian_buster centos_8 debian_testing
# PYMOR_TEST_OS=debian_buster
# end: customization points via environment variables

DOCKER_COMPOSE=CI_COMMIT_SHA=$(shell git log -1 --pretty=format:"%H") \
	docker-compose -f .binder/docker-compose.yml -p pymor
NB_DIR=notebooks
PANDOC_MAJOR=$(shell pandoc --version | head  -n1 | cut -d ' ' -f 2 | cut -d '.' -f 1)
ifeq ($(PANDOC_MAJOR),1)
	PANDOC_FORMAT=-f markdown_github
endif
# load env file, but do not overwrite pre-existing values
ENV_FILE?=.env
ENV_KEYS:=$(shell env | sort | grep -v -P "^_.*" | grep -v \] 	| sed 's/=.*//'g | tr '\n' '|')DUMMY

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
	python setup.py test

jupyter:
	jupyter notebook --notebook-dir=$(NB_DIR) --NotebookApp.disable_check_xsrf=True

tutorials: NB_DIR="docs/_build/html"
tutorials: docs jupyter

fasttest:
	PYMOR_PYTEST_MARKER="not slow" python setup.py test

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
docker_file:
	 @export $$( cat $(ENV_FILE) | grep -v -E "$(ENV_KEYS)" ) && \
		sed -e "s;CI_IMAGE_TAG;$${CI_IMAGE_TAG};g" -e "s;DOCKER_BASE_PYTHON;$${DOCKER_BASE_PYTHON};g" \
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
