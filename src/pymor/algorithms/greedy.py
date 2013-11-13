# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import time
from itertools import izip

from pymor.algorithms.basisextension import trivial_basis_extension
from pymor.core import getLogger
from pymor.core.exceptions import ExtensionError


def greedy(discretization, reductor, samples, initial_data=None, use_estimator=True, error_norm=None,
           extension_algorithm=trivial_basis_extension, target_error=None, max_extensions=None):
    '''Greedy extension algorithm.

    Parameters
    ----------
    discretization
        The discretization to reduce.
    reductor
        Reductor for reducing the given discretization. This has to be a
        function of the form `reduce(discretization, data)` where data is
        the detailed data required by the reductor. If your reductor takes
        more arguments, use functools.partial.
    samples
        The set of parameter samples on which to perform the greedy search.
        Currently this set is fixed for the whole process.
    initial_data
        This is fed into reductor.reduce() for the initial projection.
        Typically this will be the reduced basis with which the algorithm
        starts.
    use_estimator
        If True, use reduced_discretization.estimate() to estimate the errors
        on the sample set. Otherwise a detailed simulation is used to calculate
        the error.
    error_norm
        If use_estimator == Flase, use this function to calculate the norm of
        the error. [Default l2_norm]
    extension_algorithm
        The extension algorithm to use to extend the current reduced basis with
        the maximum error snapshot.
    target_error
        If not None, stop the search if the maximum error on the sample set
        drops below this value.
    max_extensions
        If not None, stop algorithm after `max_extensions` extension steps.

    Returns
    -------
    Dict with the following fields:
        'data'
            The reduced basis. (More generally the data which needs to be
            fed into reduced_discretization.reduce().
        'reduced_discretization'
            The last reduced discretization which has been computed.
        'reconstructor'
            Reconstructor for `reduced_discretization`.
        'max_err'
            Last estimated maximum error on the sample set.
        'max_err_mu'
            The parameter that corresponds to `max_err`.
        'max_errs'
            Sequence of maximum errors during the greedy run.
        'max_errs_mu'
            The parameters corresponding to `max_errs`.
    '''

    logger = getLogger('pymor.algorithms.greedy.greedy')
    samples = list(samples)
    logger.info('Started greedy search on {} samples'.format(len(samples)))
    data = initial_data

    tic = time.time()
    extensions = 0
    max_errs = []
    max_err_mus = []
    hierarchic = False

    while True:
        logger.info('Reducing ...')
        rd, rc, reduction_data = reductor(discretization, data) if not hierarchic \
            else reductor(discretization, data, extends=(rd, rc, reduction_data))

        logger.info('Estimating errors ...')
        if use_estimator:
            errors = [rd.estimate(rd.solve(mu), mu) for mu in samples]
        elif error_norm is not None:
            errors = [error_norm(discretization.solve(mu) - rc.reconstruct(rd.solve(mu)))[0] for mu in samples]
        else:
            errors = [(discretization.solve(mu) - rc.reconstruct(rd.solve(mu))).l2_norm()[0] for mu in samples]

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
            data, extension_data = extension_algorithm(data, U)
        except ExtensionError:
            logger.info('Extension failed. Stopping now.')
            break
        extensions += 1
        if not 'hierarchic' in extension_data:
            logger.warn('Extension algorithm does not report if extension was hierarchic. Assuming it was\'nt ..')
            hierarchic = False
        else:
            hierarchic = extension_data['hierarchic']

        logger.info('')

        if max_extensions is not None and extensions >= max_extensions:
            logger.info('Maximal number of {} extensions reached.'.format(max_extensions))
            logger.info('Reducing once more ...')
            rd, rc, reduction_data = reductor(discretization, data) if not hierarchic \
                else reductor(discretization, data, extends=(rd, rc, reduction_data))
            break

    tictoc = time.time() - tic
    logger.info('Greedy search took {} seconds'.format(tictoc))
    return {'data': data, 'reduced_discretization': rd, 'reconstructor': rc, 'max_err': max_err,
            'max_err_mu': max_err_mu, 'max_errs': max_errs, 'max_err_mus': max_err_mus, 'extensions': extensions,
            'time': tictoc, 'reduction_data': reduction_data}
