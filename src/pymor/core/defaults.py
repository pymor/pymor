from __future__ import absolute_import, division, print_function

import numpy as np
from .interfaces import BasicInterface
from .logger import getLogger


class Defaults(BasicInterface):
    '''
    float_cmp_tol:                  tolerance for pymor.tools.float_cmp

    gram_schmidt_tol:               tolerance for pymor.la.algroithms.gram_schmidt
    gram_schmidt_check:             check orthogonality of result
    gram_schmidt_check_tol:         tolerance for orthogonality check

    bicg_tol:                       tolerance for scipy.sparse.linalg.bicg
    bicg_maxiter:                   maximal number of iterations

    induced_norm_raise_negative:    raise error in la.induced_norm if the squared norm is negative
    induced_norm_tol:               tolerance for clipping negative norm squares to zero

    random_seed:                    seed for numpy's random generator; if None, use /dev/urandom as source for seed
    '''

    float_cmp_tol               = 2**4 * np.finfo(np.zeros(1).dtype).eps

    gram_schmidt_tol            = 1e-10
    # gram_schmidt_tol          = 1e-7  # according to comments in the rbmatlab source, such a high tolerance is
    #                                   # needed for treating nonlinear problems
    gram_schmidt_check          = True
    gram_schmidt_check_tol      = 1e-3

    bicg_tol                    = 1e-10
    bicg_maxiter                = None

    induced_norm_raise_negative = True
    induced_norm_tol            = 10e-10

    _random_seed                = 123456
    @property
    def random_seed(self):
        return self._random_seed

    @random_seed.setter
    def random_seed(self, s):
        self._random_seed = s
        np.random.seed(s)

    def __init__(self):
        np.random.seed(self.random_seed)

    def __str__(self):
        return '''
            float_cmp_tol                 = {0.float_cmp_tol}

            gram_schmidt_tol              = {0.gram_schmidt_tol}
            gram_schmidt_check            = {0.gram_schmidt_check}
            gram_schmidt_check_tol        = {0.gram_schmidt_check_tol}

            bicg_tol                      = {0.bicg_tol}
            bicg_maxiter                  = {0.bicg_maxiter}

            induced_norm_raise_negative   = {0.induced_norm_raise_negative}
            induced_norm_tol              = {0.induced_norm_tol}

            random_seed                   = {0.random_seed}
            '''.format(self)


defaults = Defaults()
defaults.lock()


# Set default log levels
# Log levels propagte downwards, i.e. if the level of "getLogger('a.b.c')" is not set
# the log level of "getLogger('a.b')" is assumed

getLogger('pymor').setLevel('WARN')
getLogger('pymor.core').setLevel('WARN')
