# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import hashlib
import random

import numpy as np

from pymor.core import dumps
from pymor.core.logger import getLogger


_file_sha = hashlib.sha1(open(__file__).read()).digest()


class Defaults(object):
    '''Class defining application-wide defaults. Do not instantiate but use
    `pymor.defaults`.

      :float_cmp_tol:                      tolerance for :func:`~pymor.tools.floatcmp.float_cmp`

      :gram_schmidt_tol:                   tolerance for :func:`~pymor.la.gram_schmidt.gram_schmidt`
      :gram_schmidt_find_duplicates:       remove duplicate vectors before orthonormalizing
      :gram_schmidt_reiterate:             orthonormalize again if norm of vector decreases strongly during
                                           orthogonalization
      :gram_schmidt_reiteration_threshold: reorthonormalize if newnorm/oldnorm is smaller than this value
      :gram_schmidt_check:                 check orthogonality of result
      :gram_schmidt_check_tol:             tolerance for orthogonality check

      :pod_tol:                            tolerance below which eigenvalues are treated as zero
      :pod_symmetrize:                     symmetrize the Gram matrix
      :pod_orthonormalize:                 orthonormalize the result again
      :pod_check:                          check orthogonality of result
      :pod_check_tol:                      tolerance for orthogonality check

      :default_sparse_solver:              default sparse solver to use (bicgstab, bicgstab-spilu, spsolve)
      :bicgstab_tol:                       see :func:`scipy.sparse.linalg.bicgstab`
      :bicgstab_maxiter:                   see :func:`scipy.sparse.linalg.bicgstab`
      :spilu_drop_tol:                     see :func:`scipy.sparse.linalg.spilu`
      :spilu_fill_factor:                  see :func:`scipy.sparse.linalg.spilu`
      :spilu_drop_rule:                    see :func:`scipy.sparse.linalg.spilu`
      :spilu_permc_spec:                   see :func:`scipy.sparse.linalg.spilu`
      :spsolve_permc_spec:                 see :func:`scipy.sparse.linalg.spsolve`

      :induced_norm_raise_negative:        raise error in la.induced_norm if the squared norm is negative
      :induced_norm_tol:                   tolerance for clipping negative norm squares to zero

      :random_seed:                        seed for NumPy's random generator; if None, use /dev/urandom as
                                           source for seed

      :compact_print:                      print (arrays) in a compact but possibly not accurate way
      :qt_visualize_patch_backend:         backend to use for plotting in :func:`pymor.gui.qt.visualize_patch`
                                             ('gl' or 'matplotlib')
    '''

    float_cmp_tol                       = 2**4 * np.finfo(np.zeros(1).dtype).eps

    gram_schmidt_tol                    = 1e-14
    # gram_schmidt_tol                  = 1e-7  # according to comments in the rbmatlab source, such a high tolerance is
    #                                           # needed for treating nonlinear problems
    gram_schmidt_find_duplicates        = True
    gram_schmidt_reiterate              = True
    gram_schmidt_reiteration_threshold  = 1e-1
    gram_schmidt_check                  = True
    gram_schmidt_check_tol              = 1e-3

    pod_tol                             = 1e-15
    pod_symmetrize                      = False
    pod_orthonormalize                  = True
    pod_check                           = True
    pod_check_tol                       = 1e-10

    default_sparse_solver               = 'bicgstab-spilu'
    bicgstab_tol                        = 1e-15
    bicgstab_maxiter                    = None
    spilu_drop_tol                      = 1e-4
    spilu_fill_factor                   = 10
    spilu_drop_rule                     = 'basic,area'
    spilu_permc_spec                    = 'COLAMD'
    spsolve_permc_spec                  = 'COLAMD'

    induced_norm_raise_negative         = True
    induced_norm_tol                    = 10e-10

    _random_seed                        = 123456

    compact_print                       = False
    qt_visualize_patch_backend          = 'gl'

    @property
    def random_seed(self):
        return self._random_seed

    @random_seed.setter
    def random_seed(self, s):
        self._random_seed = s
        random.seed(s)
        np.random.seed(s)

    def __init__(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        self._calc_sid()

    def __str__(self):
        return '''
            float_cmp_tol                       = {0.float_cmp_tol}

            gram_schmidt_tol                    = {0.gram_schmidt_tol}
            gram_schmidt_find_duplicates        = {0.gram_schmidt_find_duplicates}
            gram_schmidt_reiterate              = {0.gram_schmidt_reiterate}
            gram_schmidt_reiteration_threshold  = {0.gram_schmidt_reiteration_threshold}
            gram_schmidt_check                  = {0.gram_schmidt_check}
            gram_schmidt_check_tol              = {0.gram_schmidt_check_tol}

            pod_tol                             = {0.pod_tol}
            pod_symmetrize                      = {0.pod_symmetrize}
            pod_orthonormalize                  = {0.pod_orthonormalize}
            pod_check                           = {0.pod_check}
            pod_check_tol                       = {0.pod_check_tol}

            default_sparse_solver               = {0.default_sparse_solver}
            bicgstab_tol                        = {0.bicgstab_tol}
            bicgstab_maxiter                    = {0.bicgstab_maxiter}
            spilu_drop_tol                      = {0.spilu_drop_tol}
            spilu_fill_factor                   = {0.spilu_fill_factor}
            spilu_drop_rule                     = {0.spilu_drop_rule}
            spilu_permc_spec                    = {0.spilu_permc_spec}
            spsolve_permc_spec                  = {0.spsolve_permc_spec}

            induced_norm_raise_negative         = {0.induced_norm_raise_negative}
            induced_norm_tol                    = {0.induced_norm_tol}

            random_seed                         = {0.random_seed}

            compact_print                       = {0.compact_print}
            qt_visualize_patch_backend          = {0.qt_visualize_patch_backend}
            '''.format(self)

    def _calc_sid(self):
        object.__setattr__(self, 'sid', dumps((_file_sha, tuple((k, v) for k, v in sorted(self.__dict__.iteritems())))))

    def _state_changed(self):
        self._calc_sid()
        import pymor.core.interfaces
        if pymor.core.interfaces.ImmutableMeta.sids_created:
            logger = getLogger('pymor')
            logger.warn('Changing state of pymor.defaults after sids have been calcuated. This might break caching!')

    def __setattr__(self, k, v):
        object.__setattr__(self, k, v)
        self._state_changed()

    def __delattr__(self, k):
        object.__delattr__(self, k)
        self._state_changed()


defaults = Defaults()


# Set default log levels
# Log levels propagate downwards, i.e. if the level of "getLogger('a.b.c')" is not set
# the log level of "getLogger('a.b')" is assumed

getLogger('pymor').setLevel('WARN')
getLogger('pymor.core').setLevel('WARN')
