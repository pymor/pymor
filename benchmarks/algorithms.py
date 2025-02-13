# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.svd_va import method_of_snapshots


class VectorArrayAlgorithm:
    version = 1
    timeout = 600

    param_names = ('va_impl', 'dim', 'count')
    params = (['numpy', 'list'], [10**k for k in range(1, 7)], [10, 100])

    def setup(self, va_impl, dim, count):
        from pymor.vectorarrays.list import NumpyListVectorSpace
        from pymor.vectorarrays.numpy import NumpyVectorSpace

        rng = np.random.default_rng(42)
        data = rng.random((count, dim))
        space = NumpyVectorSpace if va_impl == 'numpy' else NumpyListVectorSpace
        self.array = space.from_numpy(data)

    def time_gram_schmidt(self, va_impl, dim, count):
        gram_schmidt(self.array, copy=False)

    def time_method_of_snapshots(self, va_impl, dim, count):
        method_of_snapshots(self.array)
