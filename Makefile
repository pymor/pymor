
.PHONY: README.txt README.html pylint test

all: 

# PyPI wants ReStructured text
README.txt: README.markdown
	pandoc -f markdown -t plain $< > $@

# I want HTML (to preview the formatting :))
README.html: README.markdown
	pandoc -f markdown -t html $< > $@

README: README.txt README.html

pep8:
	pep8 ./src

flake8:
	flake8 ./src

test:
	python setup.py test

full-test:
	python setup.py test --flakes --pep8
	
doc:
	PYTHONPATH=${PWD}/src/:${PYTHONPATH} make -C docs html
