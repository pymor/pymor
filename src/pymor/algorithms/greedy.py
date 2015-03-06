# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import time
from itertools import izip

from pymor.algorithms.basisextension import gram_schmidt_basis_extension
from pymor.core.exceptions import ExtensionError
from pymor.core.logger import getLogger


def greedy(discretization, reductor, samples, initial_basis=None, use_estimator=True, error_norm=None,
           extension_algorithm=gram_schmidt_basis_extension, target_error=None, max_extensions=None):
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
    discretization
        The |Discretization| to reduce.
    reductor
        Reductor for reducing the given |Discretization|. This has to be a
        function of the form `reductor(discretization, basis, extends=None)`.
        If your reductor takes more arguments, use, e.g., :func:`functools.partial`.
        The method has to return a tuple
        `(reduced_discretization, reconstructor, reduction_data)`.
        In case the last basis extension was `hierarchic` (see
        `extension_algorithm`), the extends argument is set to
        `(last_reduced_discretization, last_reconstructor, last_reduction_data)`
        which can be used by the reductor to speed up the reduction
        process. For an example see
        :func:`~pymor.reductors.linear.reduce_stationary_affine_linear`.
    samples
        The set of |Parameter| samples on which to perform the greedy search.
    initial_basis
        The initial reduced basis with which the algorithm starts. If `None`,
        an empty basis is used as initial basis.
    use_estimator
        If `True`, use `reduced_discretization.estimate()` to estimate the
        errors on the sample set. Otherwise a detailed simulation is
        performed to calculate the error.
    error_norm
        If `use_estimator == False`, use this function to calculate the
        norm of the error. If `None`, the Euclidean norm is used.
    extension_algorithm
        The extension algorithm to be used to extend the current reduced
        basis with the maximum error snapshot. This has to be a function
        of the form `extension_algorithm(old_basis, new_vector)`, which
        returns a tuple `(new_basis, extension_data)`, where
        `extension_data` is a dict at least containing the key
        `hierarchic`. `hierarchic` should be set to `True` if `new_basis`
        contains `old_basis` as its first vectors.
    target_error
        If not `None`, stop the algorithm if the maximum (estimated) error
        on the sample set drops below this value.
    max_extensions
        If not `None`, stop the algorithm after `max_extensions` extension
        steps.

    Returns
    -------
    Dict with the following fields:

        :basis:                  The reduced basis.
        :reduced_discretization: The reduced |Discretization| obtained for the
                                 computed basis.
        :reconstructor:          Reconstructor for `reduced_discretization`.
        :max_errs:               Sequence of maximum errors during the greedy run.
        :max_err_mus:            The parameters corresponding to `max_errs`.
    """

    logger = getLogger('pymor.algorithms.greedy.greedy')
    samples = list(samples)
    logger.info('Started greedy search on {} samples'.format(len(samples)))
    basis = initial_basis

    tic = time.time()
    extensions = 0
    max_errs = []
    max_err_mus = []
    hierarchic = False

    rd, rc, reduction_data = None, None, None
    while True:
        logger.info('Reducing ...')
        rd, rc, reduction_data = reductor(discretization, basis) if not hierarchic \
            else reductor(discretization, basis, extends=(rd, rc, reduction_data))

        if len(samples) == 0:
            logger.info('There is nothing else to do for empty samples.')
            return {'basis': basis, 'reduced_discretization': rd, 'reconstructor': rc,
                    'max_errs': [], 'max_err_mus': [], 'extensions': 0,
                    'time': time.time() - tic, 'reduction_data': reduction_data}

        logger.info('Estimating errors ...')
        if use_estimator:
            errors = [rd.estimate(rd.solve(mu), mu) for mu in samples]
        elif error_norm is not None:
            errors = [error_norm(discretization.solve(mu) - rc.reconstruct(rd.solve(mu))) for mu in samples]
        else:
            errors = [(discretization.solve(mu) - rc.reconstruct(rd.solve(mu))).l2_norm() for mu in samples]

        # most error_norms will return an array of length 1 instead of a number, so we extract the numbers
        # if necessary
        errors = map(lambda x: x[0] if hasattr(x, '__len__') else x, errors)

        max_err, max_err_mu = max(((err, mu) for err, mu in izip(errors, samples)), key=lambda t: t[0])
        max_errs.append(max_err)
        max_err_mus.append(max_err_mu)
        logger.info('Maximum error after {} extensions: {} (mu = {})'.format(extensions, max_err, max_err_mu))

        if target_error is not None and max_err <= target_error:
            logger.info('Reached maximal error on snapshots of {} <= {}'.format(max_err, target_error))
            break

        logger.info('Extending with snapshot for mu = {}'.format(max_err_mu))
        U = discretization.solve(max_err_mu)
        try:
            basis, extension_data = extension_algorithm(basis, U)
        except ExtensionError:
            logger.info('Extension failed. Stopping now.')
            break
        extensions += 1
        if 'hierarchic' not in extension_data:
            logger.warn('Extension algorithm does not report if extension was hierarchic. Assuming it was\'nt ..')
            hierarchic = False
        else:
            hierarchic = extension_data['hierarchic']

        logger.info('')

        if max_extensions is not None and extensions >= max_extensions:
            logger.info('Maximum number of {} extensions reached.'.format(max_extensions))
            logger.info('Reducing once more ...')
            rd, rc, reduction_data = reductor(discretization, basis) if not hierarchic \
                else reductor(discretization, basis, extends=(rd, rc, reduction_data))
            break

    tictoc = time.time() - tic
    logger.info('Greedy search took {} seconds'.format(tictoc))
    return {'basis': basis, 'reduced_discretization': rd, 'reconstructor': rc,
            'max_errs': max_errs, 'max_err_mus': max_err_mus, 'extensions': extensions,
            'time': tictoc, 'reduction_data': reduction_data}
