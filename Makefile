
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
	PYTHONPATH=${PWD}/src/:${PYTHONPATH} make -C docs html
