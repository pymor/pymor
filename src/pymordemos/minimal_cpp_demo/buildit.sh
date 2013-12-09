#!/bin/bash

python bindings.py
g++ -shared -fPIC -o discretization.so $(pkg-config python-2.7 --cflags) discretization.cc bindings_generated.cpp
