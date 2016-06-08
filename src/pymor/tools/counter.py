# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)


class Counter(object):

    def __init__(self, start=0):
        self.value = start

    def inc(self):
        self.value += 1
        return self.value
