#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import pprint
import sys
if sys.version_info >= (3, 0, 0):
    import dbm
else:
    import anydbm as dbm


def output(filename):
    db = dbm.open(filename, 'r')
    pprint.pprint(db.items())
    db.close()


if __name__ == '__main__':
    for fn in sys.argv[1:]:
        output(fn)
