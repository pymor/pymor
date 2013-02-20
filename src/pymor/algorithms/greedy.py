from __future__ import absolute_import, division, print_function

import time
from itertools import izip

import numpy as np

from pymor.core import getLogger
from pymor.core.exceptions import ExtensionError
from pymor.algorithms.basisextension import trivial_basis_extension
from pymor.la import l2_norm


def greedy(discretization, reductor, samples, initial_data=None, use_estimator=True, error_norm=l2_norm,
           extension_algorithm=trivial_basis_extension, target_error=None, max_extensions=None):

    logger = getLogger('pymor.algorithms.greedy.greedy')
    samples = list(samples)
    logger.info('Started greedy search on {} samples'.format(len(samples)))
    data = initial_data

    tic = time.time()
    extensions = 0
    max_errs = []
    max_err_mus = []

    while True:
        logger.info('Reducing ...')
        rd, rc = reductor.reduce(data)

        logger.info('Estimating errors ...')
        if use_estimator:
            errors = [rd.estimate(rd.solve(mu), mu) for mu in samples]
        else:
            errors = [error_norm(discretization.solve(mu) - rc.reconstruct(rd.solve(mu))) for mu in samples]

        max_err, max_err_mu = max(((err, mu) for err, mu in izip(errors, samples)), key=lambda t:t[0])
        max_errs.append(max_err)
        max_err_mus.append(max_err_mu)
        logger.info('Maximum error after {} extensions: {} (mu = {})'.format(extensions, max_err, max_err_mu))

        if target_error is not None and max_err <= target_err:
            logger.info('Reached maximal error on snapshots of {} <= {}'.format(max_err, target_error))
            break

        logger.info('Extending with snapshot for mu = {}'.format(max_err_mu))
        U = discretization.solve(max_err_mu)
        try:
            data = extension_algorithm(data, U)
        except ExtensionError:
            logger.info('Extension failed. Stopping now.')
            break
        extensions += 1

        logger.info('')

        if max_extensions is not None and extensions >= max_extensions:
            logger.info('Maximal number of {} extensions reached.'.format(max_extensions))
            break

    tictoc = time.time() - tic
    logger.info('Greedy search took {} seconds'.format(tictoc))
    return {'data':data, 'reduced_discretization':rd, 'reconstructor':rc, 'max_err':max_err, 'max_err_mu':max_err_mu,
            'max_errs':max_errs, 'max_err_mus':max_err_mus, 'extensions':extensions, 'time':tictoc}
