# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function


class Counter(object):

    def __init__(self, start=0):
        self.value = start

    def inc(self):
        self.value += 1
        return self.value
