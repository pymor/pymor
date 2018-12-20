PYMOR_DOCKER_TAG?=3.6
PYMOR_PYTEST_MARKER?=None
PANDOC_MAJOR=$(shell pandoc --version | head  -n1 | cut -d ' ' -f 2 | cut -d '.' -f 1)
ifeq ($(PANDOC_MAJOR),1)
	PANDOC_FORMAT=-f markdown_github
endif

.PHONY: README.html pylint test

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

dockerrun:
	docker run --rm -it -v $(shell pwd):/src -e PYMOR_PYTEST_MARKER=$(PYMOR_PYTEST_MARKER) pymor/testing:$(PYMOR_DOCKER_TAG) bash

dockertest:
	PYMOR_DOCKER_TAG=$(PYMOR_DOCKER_TAG) PYMOR_PYTEST_MARKER=$(PYMOR_PYTEST_MARKER) ./.ci/travis/run_travis_builders.py

dockertestfull:
	./.ci/travis/run_travis_builders.py

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
	py.test --flakes --pep8 --cov=pymor --cov-report=html --cov-report=xml src/pymortests

doc:
	PYTHONPATH=${PWD}/src/:${PYTHONPATH} make -C docs html

