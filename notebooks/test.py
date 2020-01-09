# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

""" Since we do not currently install notebooks via setup.py
    This file has to live outside the pymortests dir and be handled
    separately in the CI script too.
"""
import testipynb
import os

from pymortests.base import runmodule

NBDIR = os.path.abspath(os.path.dirname(__file__))

Test = testipynb.TestNotebooks(directory=NBDIR, timeout=600)
TestNotebooks = Test.get_tests()

if __name__ == "__main__":
    runmodule(filename=__file__)
