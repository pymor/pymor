# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.sparse as sps

from pymor.models.iosys import LTIModel
from pymor.reductors.bt import BTReductor
from pymor.reductors.h2 import IRKAReductor


class LTIMORBenchmark:
    version = 1
    timeout = 3600

    param_names = ('size',)
    params = (['small', 'large'],)

    def setup(self, size):
        rng = np.random.default_rng(0)
        m = 2
        p = 3
        if size == 'small':
            n = 100
            J = rng.standard_normal((n, n))
            J = J - J.T
            R = rng.standard_normal((n, n))
            R = R @ R.T
            Q = rng.standard_normal((n, n))
            Q = Q @ Q.T
        else:
            n = 1000
            def random_sparse_matrix(m, n, rng):
                return sps.random(m, n, density=5/n, format='csc', rng=rng,
                                  data_rvs=lambda size=None, rng=rng: rng.standard_normal(size))
            J = random_sparse_matrix(n, n, rng)
            J = J - J.T
            R = random_sparse_matrix(n, n, rng)
            R = R @ R.T + 0.1 * sps.eye(n)
            Q = random_sparse_matrix(n, n, rng)
            Q = Q @ Q.T + 0.1 * sps.eye(n)
        A = (J - R) @ Q
        B = rng.standard_normal((n, m))
        C = rng.standard_normal((p, n))
        self.fom = LTIModel.from_matrices(A, B, C)

    def time_bt(self, size):
        BTReductor(self.fom).reduce(10)

    def time_irka(self, size):
        IRKAReductor(self.fom).reduce(10)
