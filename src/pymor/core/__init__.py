# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.interfaces import (BasicInterface, ImmutableInterface, abstractmethod, abstractclassmethod,
                                   abstractstaticmethod, abstractproperty, inject_sid, disable_sid_generation,
                                   enable_sid_generation)
from pymor.core.logger import getLogger, set_log_levels



class Unpicklable(object):
    """Mix me into classes you know cannot be pickled.
    Our test system won't try to pickle me then
    """
    pass


try:
    import numpy as np
    A = np.zeros((0, 1))
    _ = A[[]]
    NUMPY_INDEX_QUIRK = False
except IndexError:
    NUMPY_INDEX_QUIRK = True
