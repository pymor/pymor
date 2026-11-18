# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

# don't use type Real = ... to allow using Real/Complex with isinstance
Real    = int | float | np.floating
Complex = int | float | complex | np.number
