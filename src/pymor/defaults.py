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
    """Class defining application-wide defaults. Do not instantiate but use
    `pymor.defaults`.

      :float_cmp_tol:                      tolerance for :func:`~pymor.tools.floatcmp.float_cmp`

      :gram_schmidt_tol:                   tolerance for :func:`~pymor.la.gram_schmidt.gram_schmidt`
      :gram_schmidt_find_duplicates:       remove duplicate vectors before orthonormalizing
      :gram_schmidt_reiterate:             orthonormalize again if norm of vector decreases strongly during
                                           orthogonalization
      :gram_schmidt_reiteration_threshold: reorthonormalize if newnorm/oldnorm is smaller than this value
      :gram_schmidt_check:                 check orthogonality of result
      :gram_schmidt_check_tol:             tolerance for orthogonality check

      :pod_tol:                            relative tolerance below which singular values are treated as zero
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

      :pyamg_tol:                          tolerance for `pyamg <http://pyamg.github.io/>` blackbox solver
      :pyamg_maxiter:                      maximum iterations for `pyamg <http://pyamg.github.io/>` blackbox solver
      :pyamg_verb:                         verbosity flag for `pyamg <http://pyamg.github.io/>` blackbox solver

      :pyamg_rs_strength:                  parameter for `pyamg <http://pyamg.github.io/>` Ruge-Stuben solver
      :pyamg_rs_CF:                        parameter for `pyamg <http://pyamg.github.io/>` Ruge-Stuben solver
      :pyamg_rs_presmoother:               parameter for `pyamg <http://pyamg.github.io/>` Ruge-Stuben solver
      :pyamg_rs_postsmoother:              parameter for `pyamg <http://pyamg.github.io/>` Ruge-Stuben solver
      :pyamg_rs_max_levels:                parameter for `pyamg <http://pyamg.github.io/>` Ruge-Stuben solver
      :pyamg_rs_max_coarse:                parameter for `pyamg <http://pyamg.github.io/>` Ruge-Stuben solver
      :pyamg_rs_coarse_solver:             parameter for `pyamg <http://pyamg.github.io/>` Ruge-Stuben solver
      :pyamg_rs_cycle:                     parameter for `pyamg <http://pyamg.github.io/>` Ruge-Stuben solver
      :pyamg_rs_accel:                     parameter for `pyamg <http://pyamg.github.io/>` Ruge-Stuben solver
      :pyamg_rs_tol:                       parameter for `pyamg <http://pyamg.github.io/>` Ruge-Stuben solver
      :pyamg_rs_maxiter:                   parameter for `pyamg <http://pyamg.github.io/>` Ruge-Stuben solver

      :pyamg_sa_symmetry:                  parameter for `pyamg <http://pyamg.github.io/>` Smoothed-Aggregation solver
      :pyamg_sa_strength:                  parameter for `pyamg <http://pyamg.github.io/>` Smoothed-Aggregation solver
      :pyamg_sa_aggregate:                 parameter for `pyamg <http://pyamg.github.io/>` Smoothed-Aggregation solver
      :pyamg_sa_smooth:                    parameter for `pyamg <http://pyamg.github.io/>` Smoothed-Aggregation solver
      :pyamg_sa_presmoother:               parameter for `pyamg <http://pyamg.github.io/>` Smoothed-Aggregation solver
      :pyamg_sa_postsmoother:              parameter for `pyamg <http://pyamg.github.io/>` Smoothed-Aggregation solver
      :pyamg_sa_improve_candidates:        parameter for `pyamg <http://pyamg.github.io/>` Smoothed-Aggregation solver
      :pyamg_sa_max_levels:                parameter for `pyamg <http://pyamg.github.io/>` Smoothed-Aggregation solver
      :pyamg_sa_max_coarse:                parameter for `pyamg <http://pyamg.github.io/>` Smoothed-Aggregation solver
      :pyamg_sa_diagonal_dominance:        parameter for `pyamg <http://pyamg.github.io/>` Smoothed-Aggregation solver
      :pyamg_sa_coarse_solver:             parameter for `pyamg <http://pyamg.github.io/>` Smoothed-Aggregation solver
      :pyamg_sa_cycle:                     parameter for `pyamg <http://pyamg.github.io/>` Smoothed-Aggregation solver
      :pyamg_sa_accel:                     parameter for `pyamg <http://pyamg.github.io/>` Smoothed-Aggregation solver
      :pyamg_sa_tol:                       parameter for `pyamg <http://pyamg.github.io/>` Smoothed-Aggregation solver
      :pyamg_sa_maxiter:                   parameter for `pyamg <http://pyamg.github.io/>` Smoothed-Aggregation solver

      :newton_miniter:                     minimum number of iterations
      :newton_maxiter:                     maximum number of iterations
      :newton_reduction:                   reduction of initial residual to achieve
      :newton_abs_limit:                   stop if absolute norm of residual falls below this limit
      :newton_stagnation_window:           see `newton_stagnation_threshold`
      :newton_stagnation_threshold:        stop if norm of residual is not reduced by this factor during the last
                                           `newton_stagnation_window` iterations
      :newton_abs_limit:                   stop if absolute norm of residual falls below this limit

      :induced_norm_raise_negative:        raise error in la.induced_norm if the squared norm is negative
      :induced_norm_tol:                   tolerance for clipping negative norm squares to zero

      :random_seed:                        seed for NumPy's random generator; if None, use /dev/urandom as
                                           source for seed

      :compact_print:                      print (arrays) in a compact but possibly not accurate way
      :qt_visualize_patch_backend:         backend to use for plotting in :func:`pymor.gui.qt.visualize_patch`
                                             ('gl' or 'matplotlib')
    """

    float_cmp_tol                       = 2**4 * np.finfo(np.zeros(1).dtype).eps

    gram_schmidt_tol                    = 1e-14
    # gram_schmidt_tol                  = 1e-7  # according to comments in the rbmatlab source, such a high tolerance is
    #                                           # needed for treating nonlinear problems
    gram_schmidt_find_duplicates        = True
    gram_schmidt_reiterate              = True
    gram_schmidt_reiteration_threshold  = 1e-1
    gram_schmidt_check                  = True
    gram_schmidt_check_tol              = 1e-3

    pod_tol                             = 4e-8
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

    pyamg_tol                           = 1e-5
    pyamg_maxiter                       = 400
    pyamg_verb                          = False

    pyamg_rs_strength                   = ('classical', {'theta': 0.25})
    pyamg_rs_CF                         = 'RS'
    pyamg_rs_presmoother                = ('gauss_seidel', {'sweep': 'symmetric'})
    pyamg_rs_postsmoother               = ('gauss_seidel', {'sweep': 'symmetric'})
    pyamg_rs_max_levels                 = 10
    pyamg_rs_max_coarse                 = 500
    pyamg_rs_coarse_solver              = 'pinv2'
    pyamg_rs_cycle                      = 'V'
    pyamg_rs_accel                      = None
    pyamg_rs_tol                        = 1e-5
    pyamg_rs_maxiter                    = 100

    pyamg_sa_symmetry                   = 'hermitian'
    pyamg_sa_strength                   = 'symmetric'
    pyamg_sa_aggregate                  = 'standard'
    pyamg_sa_smooth                     = ('jacobi', {'omega': 4.0/3.0})
    pyamg_sa_presmoother                = ('block_gauss_seidel', {'sweep': 'symmetric'})
    pyamg_sa_postsmoother               = ('block_gauss_seidel', {'sweep': 'symmetric'})
    pyamg_sa_improve_candidates         = [('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 4}), None]
    pyamg_sa_max_levels                 = 10
    pyamg_sa_max_coarse                 = 500
    pyamg_sa_diagonal_dominance         = False
    pyamg_sa_coarse_solver              = 'pinv2'
    pyamg_sa_cycle                      = 'V'
    pyamg_sa_accel                      = None
    pyamg_sa_tol                        = 1e-5
    pyamg_sa_maxiter                    = 100

    newton_miniter                      = 0
    newton_maxiter                      = 10
    newton_reduction                    = 1e-10
    newton_abs_limit                    = 1e-15
    newton_stagnation_window            = 0
    newton_stagnation_threshold         = np.inf

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

            pyamg_tol                           = {0.pyamg_tol}
            pyamg_maxiter                       = {0.pyamg_maxiter}
            pyamg_verb                          = {0.pyamg_verb}

            pyamg_rs_strength                   = {0.pyamg_rs_strength}
            pyamg_rs_CF                         = {0.pyamg_rs_CF}
            pyamg_rs_presmoother                = {0.pyamg_rs_presmoother}
            pyamg_rs_postsmoother               = {0.pyamg_rs_postsmoother}
            pyamg_rs_max_levels                 = {0.pyamg_rs_max_levels}
            pyamg_rs_max_coarse                 = {0.pyamg_rs_max_coarse}
            pyamg_rs_coarse_solver              = {0.pyamg_rs_coarse_solver}
            pyamg_rs_cycle                      = {0.pyamg_rs_cycle}
            pyamg_rs_accel                      = {0.pyamg_rs_accel}
            pyamg_rs_tol                        = {0.pyamg_rs_tol}
            pyamg_rs_maxiter                    = {0.pyamg_rs_maxiter}

            pyamg_sa_symmetry                   = {0.pyamg_sa_symmetry}
            pyamg_sa_strength                   = {0.pyamg_sa_strength}
            pyamg_sa_aggregate                  = {0.pyamg_sa_aggregate}
            pyamg_sa_smooth                     = {0.pyamg_sa_smooth}
            pyamg_sa_presmoother                = {0.pyamg_sa_presmoother}
            pyamg_sa_postsmoother               = {0.pyamg_sa_postsmoother}
            pyamg_sa_improve_candidates         = {0.pyamg_sa_improve_candidates}
            pyamg_sa_max_levels                 = {0.pyamg_sa_max_levels}
            pyamg_sa_max_coarse                 = {0.pyamg_sa_max_coarse}
            pyamg_sa_diagonal_dominance         = {0.pyamg_sa_diagonal_dominance}
            pyamg_sa_coarse_solver              = {0.pyamg_sa_coarse_solver}
            pyamg_sa_cycle                      = {0.pyamg_sa_cycle}
            pyamg_sa_accel                      = {0.pyamg_sa_accel}
            pyamg_sa_tol                        = {0.pyamg_sa_tol}
            pyamg_sa_maxiter                    = {0.pyamg_sa_maxiter}

            induced_norm_raise_negative         = {0.induced_norm_raise_negative}
            induced_norm_tol                    = {0.induced_norm_tol}

            newton_miniter                      = {0.newton_miniter}
            newton_maxiter                      = {0.newton_maxiter}
            newton_reduction                    = {0.newton_reduction}
            newton_abs_limit                    = {0.newton_abs_limit}
            newton_stagnation_window            = {0.newton_stagnation_window}

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
