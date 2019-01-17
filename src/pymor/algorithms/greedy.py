# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import time

import numpy as np

from pymor.core.exceptions import ExtensionError
from pymor.core.logger import getLogger
from pymor.parallel.dummy import dummy_pool
from pymor.parallel.manager import RemoteObjectManager


def greedy(d, reductor, samples, use_estimator=True, error_norm=None,
           atol=None, rtol=None, max_extensions=None, extension_params=None, pool=None):
    """Greedy basis generation algorithm.

    This algorithm generates a reduced basis by iteratively adding the
    worst approximated solution snapshot for a given training set to the
    reduced basis. The approximation error is computed either by directly
    comparing the reduced solution to the detailed solution or by using
    an error estimator (`use_estimator == True`). The reduction and basis
    extension steps are performed by calling the methods provided by the
    `reductor` and `extension_algorithm` arguments.

    Parameters
    ----------
    d
        The |Discretization| to reduce.
    reductor
        Reductor for reducing the given |Discretization|. This has to be
        an object with a `reduce` method, such that `reductor.reduce()`
        yields the reduced discretization, and an `exted_basis` method,
        such that `reductor.extend_basis(U, copy_U=False, **extension_params)`
        extends the current reduced basis by the vectors contained in `U`.
        For an example see :class:`~pymor.reductors.coercive.CoerciveRBReductor`.
    samples
        The set of |Parameter| samples on which to perform the greedy search.
    use_estimator
        If `True`, use `rd.estimate()` to estimate the errors on the
        sample set. Otherwise `d.solve()` is called to compute the exact
        model reduction error.
    error_norm
        If `use_estimator == False`, use this function to calculate the
        norm of the error. If `None`, the Euclidean norm is used.
    atol
        If not `None`, stop the algorithm if the maximum (estimated) error
        on the sample set drops below this value.
    rtol
        If not `None`, stop the algorithm if the maximum (estimated)
        relative error on the sample set drops below this value.
    max_extensions
        If not `None`, stop the algorithm after `max_extensions` extension
        steps.
    extension_params
        `dict` of parameters passed to the `reductor.extend_basis` method.
    pool
        If not `None`, the |WorkerPool| to use for parallelization.

    Returns
    -------
    Dict with the following fields:

        :rd:                     The reduced |Discretization| obtained for the
                                 computed basis.
        :max_errs:               Sequence of maximum errors during the greedy run.
        :max_err_mus:            The parameters corresponding to `max_errs`.
        :extensions:             Number of performed basis extensions.
        :time:                   Total runtime of the algorithm.
    """

    logger = getLogger('pymor.algorithms.greedy.greedy')
    samples = list(samples)
    sample_count = len(samples)
    extension_params = extension_params or {}
    logger.info('Started greedy search on {} samples'.format(sample_count))
    if pool is None or pool is dummy_pool:
        pool = dummy_pool
    else:
        logger.info('Using pool of {} workers for parallel greedy search'.format(len(pool)))

    with RemoteObjectManager() as rom:
        # Push everything we need during the greedy search to the workers.
        # Distribute the training set evenly among the workes.
        if not use_estimator:
            rom.manage(pool.push(d))
            if error_norm:
                rom.manage(pool.push(error_norm))
        samples = rom.manage(pool.scatter_list(samples))

        tic = time.time()
        extensions = 0
        max_errs = []
        max_err_mus = []

        while True:
            with logger.block('Reducing ...'):
                rd = reductor.reduce()

            if sample_count == 0:
                logger.info('There is nothing else to do for empty samples.')
                return {'rd': rd,
                        'max_errs': [], 'max_err_mus': [], 'extensions': 0,
                        'time': time.time() - tic}

            with logger.block('Estimating errors ...'):
                if use_estimator:
                    errors, mus = list(zip(*pool.apply(_estimate, rd=rd, d=None, reductor=None,
                                                       samples=samples, error_norm=None)))
                else:
                    errors, mus = list(zip(*pool.apply(_estimate, rd=rd, d=d, reductor=reductor,
                                                       samples=samples, error_norm=error_norm)))
            max_err_ind = np.argmax(errors)
            max_err, max_err_mu = errors[max_err_ind], mus[max_err_ind]

            max_errs.append(max_err)
            max_err_mus.append(max_err_mu)
            logger.info('Maximum error after {} extensions: {} (mu = {})'.format(extensions, max_err, max_err_mu))

            if atol is not None and max_err <= atol:
                logger.info('Absolute error tolerance ({}) reached! Stoping extension loop.'.format(atol))
                break

            if rtol is not None and max_err / max_errs[0] <= rtol:
                logger.info('Relative error tolerance ({}) reached! Stoping extension loop.'.format(rtol))
                break

            with logger.block('Computing solution snapshot for mu = {} ...'.format(max_err_mu)):
                U = d.solve(max_err_mu)
            with logger.block('Extending basis with solution snapshot ...'):
                try:
                    reductor.extend_basis(U, copy_U=False, **extension_params)
                except ExtensionError:
                    logger.info('Extension failed. Stopping now.')
                    break
            extensions += 1

            logger.info('')

            if max_extensions is not None and extensions >= max_extensions:
                logger.info('Maximum number of {} extensions reached.'.format(max_extensions))
                with logger.block('Reducing once more ...'):
                    rd = reductor.reduce()
                break

        tictoc = time.time() - tic
        logger.info('Greedy search took {} seconds'.format(tictoc))
        return {'rd': rd,
                'max_errs': max_errs, 'max_err_mus': max_err_mus, 'extensions': extensions,
                'time': tictoc}


def _estimate(rd=None, d=None, reductor=None, samples=None, error_norm=None):
    if not samples:
        return -1., None

    if d is None:
        errors = [rd.estimate(rd.solve(mu), mu) for mu in samples]
    elif error_norm is not None:
        errors = [error_norm(d.solve(mu) - reductor.reconstruct(rd.solve(mu))) for mu in samples]
    else:
        errors = [(d.solve(mu) - reductor.reconstruct(rd.solve(mu))).l2_norm() for mu in samples]
    # most error_norms will return an array of length 1 instead of a number, so we extract the numbers
    # if necessary
    errors = [x[0] if hasattr(x, '__len__') else x for x in errors]
    max_err_ind = np.argmax(errors)

    return errors[max_err_ind], samples[max_err_ind]
