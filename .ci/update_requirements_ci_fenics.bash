#!/bin/bash

echo "Updating requirements-ci-fenics.txt ..."

source /venv/bin/activate
pip-compile --resolver backtracking --extra ipyparallel --extra ann --extra ci \
	--extra-index-url https://download.pytorch.org/whl/cpu \
	-o requirements-ci-fenics.txt
