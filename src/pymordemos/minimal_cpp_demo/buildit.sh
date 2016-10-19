#!/bin/bash

python bindings.py
g++ -shared -fPIC -o discretization.so $(pkg-config python3 --cflags --libs) discretization.cc bindings_generated.cpp
