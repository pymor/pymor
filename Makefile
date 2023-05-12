#!/usr/bin/env make

DOCKER ?= docker
THIS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

CI_CURRENT_IMAGE_TAG := $(shell sha256sum $(THIS_DIR)/requirements-ci-current.txt | cut -d " " -f 1)
CI_OLDEST_IMAGE_TAG  := $(shell sha256sum $(THIS_DIR)/requirements-ci-oldest.txt  | cut -d " " -f 1)
CI_FENICS_IMAGE_TAG  := $(shell sha256sum $(THIS_DIR)/requirements-ci-fenics.txt  | cut -d " " -f 1)

CI_CURRENT_IMAGE_TARGET_TAG := $(or $(TARGET_TAG),$(CI_CURRENT_IMAGE_TAG))
CI_OLDEST_IMAGE_TARGET_TAG  := $(or $(TARGET_TAG),$(CI_OLDEST_IMAGE_TAG))
CI_FENICS_IMAGE_TARGET_TAG  := $(or $(TARGET_TAG),$(CI_FENICS_IMAGE_TAG))

CI_PREFLIGHT_IMAGE_TARGET_TAG  := $(or $(TARGET_TAG),latest)


PANDOC_MAJOR=$(shell ( which pandoc && pandoc --version | head  -n1 | cut -d ' ' -f 2 | cut -d '.' -f 1)) || echo "pandoc missing")
ifeq ($(PANDOC_MAJOR),1)
	PANDOC_FORMAT=-f markdown_github
endif

.PHONY: README.html README test docs

# I want HTML (to preview the formatting :))
README.html: README.md
	pandoc $(PANDOC_FORMAT) -t html $< > $@

README: README.html

test:
	xvfb-run pytest

docs:
	PYTHONPATH=${PWD}/src/:${PYTHONPATH} make -C docs html

ci_preflight_image:
	$(DOCKER) build -t pymor/ci-preflight -f $(THIS_DIR)/docker/Dockerfile.ci-preflight $(THIS_DIR)

CI_EXTRAS= \
	--extra docs-additional \
	--extra tests \
	--extra ann \
	--extra slycot \
	--extra pymess \
	--extra ipyparallel \
	--extra mpi \
	--extra gui \
	--extra jupyter \
	--extra vtk \
	--extra gmsh \
	--extra dune \
	--extra ngsolve \
	--extra scikit-fem

ci_current_requirements:
	# we run pip-compile in a container to ensure that the right Python version is used
	$(DOCKER) run --rm -it -v=$(THIS_DIR):/src python:3.10-bullseye /bin/bash -c "\
		cd /src && \
		pip install pip-tools==6.13.0 && \
		pip-compile --resolver backtracking \
			$(CI_EXTRAS) \
			--extra-index-url https://download.pytorch.org/whl/cpu \
			-o requirements-ci-current.txt \
		"

ci_oldest_requirements:
	# we run pip-compile in a container to ensure that the right Python version is used
	$(DOCKER) run --rm -it -v=$(THIS_DIR):/src python:3.8-bullseye /bin/bash -c "\
		cd /src && \
		pip install pip-tools==6.13.0 && \
		pip-compile --resolver backtracking \
			$(CI_EXTRAS) \
			--extra-index-url https://download.pytorch.org/whl/cpu \
			-o requirements-ci-oldest.txt \
			pyproject.toml requirements-ci-oldest-pins.in \
		"


ci_fenics_requirements:
	$(DOCKER) run --rm -it -v=$(THIS_DIR):/src python:3.11-bullseye /bin/bash -c "\
		cd /src && \
		pip install pip-tools==6.13.0 && \
		pip-compile --resolver backtracking \
			--extra docs_additional \
			--extra tests \
			--extra ci \
			--extra ann \
			--extra ipyparallel \
			--extra mpi \
			--extra-index-url https://download.pytorch.org/whl/cpu \
			-o requirements-ci-fenics.txt \
		"

CONDA_EXTRAS = \
	--extras tests \
	--extras ci \
	--extras slycot \
	--extras ipyparallel \
	--extras mpi \
	--extras gui \
	--extras jupyter \
	--extras vtk \
	--extras gmsh
	# pymess, dune, ngsolve, scikit-fem (no recent version) not available as conda-forge packages
	# pytorch not available for win64
	# docs-additional not needed

ci_conda_requirements:
	conda-lock --micromamba -c conda-forge --filter-extras --no-dev-dependencies $(CONDA_EXTRAS) -f pyproject.toml
	conda-lock render $(CONDA_EXTRAS)

ci_requirements: ci_current_requirements ci_oldest_requirements ci_fenics_requirements ci_conda_requirements

ci_current_image:
	$(DOCKER) build -t pymor/ci-current:$(CI_CURRENT_IMAGE_TAG) -f $(THIS_DIR)/docker/Dockerfile.ci-current $(THIS_DIR)

ci_oldest_image:
	$(DOCKER) build -t pymor/ci-oldest:$(CI_OLDEST_IMAGE_TAG) -f $(THIS_DIR)/docker/Dockerfile.ci-oldest $(THIS_DIR)

ci_fenics_image:
	$(DOCKER) build -t pymor/ci-fenics:$(CI_FENICS_IMAGE_TAG) -f $(THIS_DIR)/docker/Dockerfile.ci-fenics $(THIS_DIR)

ci_images: ci_current_image ci_oldest_image ci_fenics_image


ci_current_image_pull:
	$(DOCKER) pull zivgitlab.wwu.io/pymor/pymor/ci-current:$(CI_CURRENT_IMAGE_TAG)

ci_oldest_image_pull:
	$(DOCKER) pull zivgitlab.wwu.io/pymor/pymor/ci-oldest:$(CI_OLDEST_IMAGE_TAG)

ci_fenics_image_pull:
	$(DOCKER) pull zivgitlab.wwu.io/pymor/pymor/ci-fenics:$(CI_FENICS_IMAGE_TAG)

ci_images_pull: ci_current_image_pull ci_oldest_image_pull ci_fenics_image_pull


ci_current_image_push:
	$(DOCKER) login $(DOCKER_LOGIN_ARGS) zivgitlab.wwu.io
	$(DOCKER) push pymor/ci-current:$(CI_CURRENT_IMAGE_TAG) \
		zivgitlab.wwu.io/pymor/pymor/ci-current:$(CI_CURRENT_IMAGE_TARGET_TAG)

ci_oldest_image_push:
	$(DOCKER) login $(DOCKER_LOGIN_ARGS) zivgitlab.wwu.io
	$(DOCKER) push pymor/ci-oldest:$(CI_OLDEST_IMAGE_TAG) \
		zivgitlab.wwu.io/pymor/pymor/ci-oldest:$(CI_OLDEST_IMAGE_TARGET_TAG)

ci_fenics_image_push:
	$(DOCKER) login $(DOCKER_LOGIN_ARGS) zivgitlab.wwu.io
	$(DOCKER) push pymor/ci-fenics:$(CI_FENICS_IMAGE_TAG) \
		zivgitlab.wwu.io/pymor/pymor/ci-fenics:$(CI_FENICS_IMAGE_TARGET_TAG)

ci_preflight_image_push:
	$(DOCKER) login $(DOCKER_LOGIN_ARGS) zivgitlab.wwu.io
	$(DOCKER) push pymor/ci-preflight \
		zivgitlab.wwu.io/pymor/pymor/ci-preflight

ci_images_push: ci_current_image_push ci_oldest_image_push ci_fenics_image_push


ci_current_image_run:
	$(DOCKER) run --rm -it -v=$(THIS_DIR):/src pymor/ci-current:$(CI_CURRENT_IMAGE_TAG)

ci_oldest_image_run:
	$(DOCKER) run --rm -it -v=$(THIS_DIR):/src pymor/ci-oldest:$(CI_OLDEST_IMAGE_TAG)

ci_fenics_image_run:
	$(DOCKER) run --rm -it -v=$(THIS_DIR):/src pymor/ci-fenics:$(CI_FENICS_IMAGE_TAG)


ci_current_image_run_notebook:
	$(DOCKER) run --rm -it -p 8888:8888 -v=$(THIS_DIR):/src pymor/ci-current:$(CI_CURRENT_IMAGE_TAG) \
		bash -c "pip install -e . && jupyter notebook --allow-root --ip=0.0.0.0"

ci_oldest_image_run_notebook:
	$(DOCKER) run --rm -it -p 8888:8888 -v=$(THIS_DIR):/src pymor/ci-oldest:$(CI_OLDEST_IMAGE_TAG) \
		bash -c "pip install -e . && jupyter notebook --allow-root --ip=0.0.0.0"
