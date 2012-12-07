
.PHONY: all pylint test

all: README.txt README.html

# PyPI wants ReStructured text
README.txt: README.markdown
	pandoc -f markdown -t rst $< > $@

# I want HTML (to preview the formatting :))
README.html: README.markdown
	pandoc -f markdown -t html $< > $@

pylint:
	cd src ; pylint --rcfile pylint.cfg pymor

test:
	/usr/bin/env python ./run_tests.py

doc:
	sphinx-apidoc -o docs -f -F -H pyMor -A AUTHORS -V 0.0.1 -R 0.0.1 src/
	PYTHONPATH=${PWD}/src/:${PYTHONPATH} make -C docs html
