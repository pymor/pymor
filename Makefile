
.PHONY: README.txt README.html pylint test

all: 

# PyPI wants ReStructured text
README.txt: README.markdown
	pandoc -f markdown -t plain $< > $@

# I want HTML (to preview the formatting :))
README.html: README.markdown
	pandoc -f markdown -t html $< > $@

README: README.txt README.html

pylint:
	cd src ; pylint --rcfile pylint.cfg pymor

pep8:
	pep8 ./src

flake8:
	flake8 ./src

test:
	python setup.py test

doc:
	PYTHONPATH=${PWD}/src/:${PYTHONPATH} make -C docs html
