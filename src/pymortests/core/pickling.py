# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import tempfile
import os
# import pytest

from pymor import core
from pymor.core.logger import getLogger
from pymortests.base import runmodule
from pymortests.fixtures import basicinterface_subclass    # NOQA


# @pytest.mark.skipif('__name__ != "__main__"')
def testDump(basicinterface_subclass):
    try:
        obj = basicinterface_subclass()
        assert isinstance(obj, basicinterface_subclass)
        if issubclass(basicinterface_subclass, core.Unpicklable):
            return
    except TypeError as e:
        logger = getLogger('pymortests.core.pickling')
        logger.debug('PicklingError: Not testing {} because its init failed: {}'.format(basicinterface_subclass,
                                                                                        str(e)))
        return

    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as dump_file:
        core.dump(obj, dump_file)
        dump_file.close()
        f = open(dump_file.name, 'rb')
        unpickled = core.load(f)
        assert obj.__class__ == unpickled.__class__
        os.unlink(dump_file.name)

if __name__ == "__main__":
    runmodule(filename=__file__)
