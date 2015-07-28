#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#
# Contributors: Falk Meyer <falk.meyer@wwu.de>

from __future__ import absolute_import, division, print_function

from pymor.discretizers.disk import discretize_instationary_from_disk


# Test a simple stationary discretization for pymor.discretizers.disk
def discretize_stationary_test(parameter_file="parameter_instationary.ini"):

    # load discretization
    dis = discretize_instationary_from_disk(parameter_file)

    # compute solutions
    for parameter in dis.parameter_space.sample_uniformly(4):
        print(dis.solve(parameter))

if __name__ == '__main__':
    discretize_stationary_test()
