#!/usr/bin/env python

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
