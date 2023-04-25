#!/bin/bash

echo "Updating requirements-ci.txt ..."

pip-compile --resolver backtracking --extra ci --extra docs --extra io \
	--extra ipyparallel --extra ann --extra optional --extra compiled \
	--extra-index-url https://download.pytorch.org/whl/cpu \
	-o requirements-ci.txt
