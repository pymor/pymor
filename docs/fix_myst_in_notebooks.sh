#!/bin/sh

BASEDIR=$(dirname "$0")

for path in $(find . -name '*.ipynb') ; do
	echo "Fixing $path"
	"${BASEDIR}/fix_myst_in_notebook.py" $path
done

