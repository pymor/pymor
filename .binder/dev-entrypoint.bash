#!/bin/bash

cd /pymor
python setup.py build_ext -i

exec "${@}"
