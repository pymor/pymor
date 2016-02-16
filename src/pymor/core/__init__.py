# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

try:
    import numpy as np
    A = np.zeros((0, 1))
    _ = A[[]]
    NUMPY_INDEX_QUIRK = False
except IndexError:
    NUMPY_INDEX_QUIRK = True
