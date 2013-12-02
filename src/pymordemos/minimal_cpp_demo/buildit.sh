#!/bin/bash

python bindings.py
g++ -shared -fPIC -o discretization.so -I/usr/include/python2.7 discretization.cc bindings_generated.cpp
