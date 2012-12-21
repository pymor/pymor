#!/bin/bash
cython -a relations.pyx
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -I$HOME/.virtualenvs/pyMor27/lib/python2.7/site-packages/numpy/core/include -o relations.so relations.c
