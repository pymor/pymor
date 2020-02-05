DOCKER_COMPOSE=docker-compose -f .binder/docker-compose.yml -p pymor
PYMOR_PYTEST_MARKER?=None
NB_DIR=notebooks
PANDOC_MAJOR=$(shell pandoc --version | head  -n1 | cut -d ' ' -f 2 | cut -d '.' -f 1)
ifeq ($(PANDOC_MAJOR),1)
	PANDOC_FORMAT=-f markdown_github
endif

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

# Docker targets
docker_image:
	$(DOCKER_COMPOSE) build

docker_docs:
	NB_DIR=notebooks $(DOCKER_COMPOSE) run docs ./.ci/gitlab/test_docs.bash

docker_run: docker_image
	$(DOCKER_COMPOSE) run --service-ports jupyter bash

docker_tutorials: NB_DIR=docs/_build/html
docker_tutorials: docker_docs docker_jupyter

docker_test: docker_image
	PYMOR_PYTEST_MARKER=$(PYMOR_PYTEST_MARKER) $(DOCKER_COMPOSE) up pytest

docker_jupyter: docker_image
	NB_DIR=$(NB_DIR) $(DOCKER_COMPOSE) up jupyter
