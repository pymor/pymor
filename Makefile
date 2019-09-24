DOCKER_COMPOSE=docker-compose -f .binder/docker-compose.yml -p pymor
PYMOR_PYTEST_MARKER?=None
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

image:
	$(DOCKER_COMPOSE) build
dockerdocs: image
	$(DOCKER_COMPOSE) run pytest ./.ci/gitlab/test_docs.bash
dockerrun: image
	$(DOCKER_COMPOSE) run jupyter bash
dockertest: image
	PYMOR_PYTEST_MARKER=$(PYMOR_PYTEST_MARKER) $(DOCKER_COMPOSE) up pytest
jupyter_server: image
	$(DOCKER_COMPOSE) up jupyter

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
