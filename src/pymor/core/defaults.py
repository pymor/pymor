from __future__ import absolute_import, division, print_function

import numpy as np
from .interfaces import BasicInterface
from .logger import getLogger


class Defaults(BasicInterface):
    '''
    float_cmp_tol: default tolerance for pymor.la.float_cmp
    '''

    float_cmp_tol    = 2**2 * np.finfo(np.zeros(1).dtype).eps

    gram_schmidt_tol = 1e-7  # according to comments in the rbmatlab source, such a high tolerance is
                             # needed for treating nonlinear problems

    def __str__(self):
        return 'float_cmp_tol = {}'.format(self.float_cmp_tol)


defaults = Defaults()
defaults.lock()


# Set default log levels
# Log levels propagte downwards, i.e. if the level of "getLogger('a.b.c')" is not set
# the log level of "getLogger('a.b')" is assumed

getLogger('pymor').setLevel('WARN')
getLogger('pymor.core').setLevel('WARN')
