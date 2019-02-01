#!/bin/bash

python bindings.py
g++ -shared -fPIC -o model.so $(pkg-config python3 --cflags --libs) model.cc bindings_generated.cpp
