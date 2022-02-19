# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import time

import numpy as np

from pymor.core.base import BasicObject, abstractmethod
from pymor.core.exceptions import ExtensionError
from pymor.core.logger import getLogger
from pymor.parallel.dummy import dummy_pool
from pymor.parallel.interface import RemoteObject


def weak_greedy(surrogate, training_set, atol=None, rtol=None, max_extensions=None, pool=None):
    """Weak greedy basis generation algorithm :cite:`BCDDPW11`.

    This algorithm generates an approximation basis for a given set of vectors
    associated with a training set of parameters by iteratively evaluating a
    :class:`surrogate <WeakGreedySurrogate>` for the approximation error on
    the training set and adding the worst approximated vector (according to
    the surrogate) to the basis.

    The constructed basis is extracted from the surrogate after termination
    of the algorithm.

    Parameters
    ----------
    surrogate
        An instance of :class:`WeakGreedySurrogate` representing the surrogate
        for the approximation error.
    training_set
        The set of parameter samples on which to perform the greedy search.
    atol
        If not `None`, stop the algorithm if the maximum (estimated) error
        on the training set drops below this value.
    rtol
        If not `None`, stop the algorithm if the maximum (estimated)
        relative error on the training set drops below this value.
    max_extensions
        If not `None`, stop the algorithm after `max_extensions` extension
        steps.
    pool
        If not `None`, a |WorkerPool| to use for parallelization. Parallelization
        needs to be supported by `surrogate`.

    Returns
    -------
    Dict with the following fields:

        :max_errs:               Sequence of maximum estimated errors during the greedy run.
        :max_err_mus:            The parameters corresponding to `max_errs`.
        :extensions:             Number of performed basis extensions.
        :time:                   Total runtime of the algorithm.
    """
    logger = getLogger('pymor.algorithms.greedy.weak_greedy')
    training_set = list(training_set)
    logger.info(f'Started greedy search on training set of size {len(training_set)}.')

    tic = time.perf_counter()
    if not training_set:
        logger.info('There is nothing else to do for an empty training set.')
        return {'max_errs': [], 'max_err_mus': [], 'extensions': 0,
                'time': time.perf_counter() - tic}

    if pool is None:
        pool = dummy_pool
    elif pool is not dummy_pool:
        logger.info(f'Using pool of {len(pool)} workers for parallel greedy search.')

    # Distribute the training set evenly among the workers.
    if pool:
        training_set = pool.scatter_list(training_set)

    extensions = 0
    max_errs = []
    max_err_mus = []

    while True:
        with logger.block('Estimating errors ...'):
            max_err, max_err_mu = surrogate.evaluate(training_set)
            max_errs.append(max_err)
            max_err_mus.append(max_err_mu)

        logger.info(f'Maximum error after {extensions} extensions: {max_err} (mu = {max_err_mu})')

        if atol is not None and max_err <= atol:
            logger.info(f'Absolute error tolerance ({atol}) reached! Stopping extension loop.')
            break

        if rtol is not None and max_err / max_errs[0] <= rtol:
            logger.info(f'Relative error tolerance ({rtol}) reached! Stopping extension loop.')
            break

        with logger.block(f'Extending surrogate for mu = {max_err_mu} ...'):
            try:
                surrogate.extend(max_err_mu)
            except ExtensionError:
                logger.info('Extension failed. Stopping now.')
                break
            extensions += 1

        logger.info('')

        if max_extensions is not None and extensions >= max_extensions:
            logger.info(f'Maximum number of {max_extensions} extensions reached.')
            break

    tictoc = time.perf_counter() - tic
    logger.info(f'Greedy search took {tictoc} seconds')
    return {'max_errs': max_errs, 'max_err_mus': max_err_mus, 'extensions': extensions,
            'time': tictoc}


class WeakGreedySurrogate(BasicObject):
    """Surrogate for the approximation error in :func:`weak_greedy`."""

    @abstractmethod
    def evaluate(self, mus, return_all_values=False):
        """Evaluate the surrogate for given parameters.

        Parameters
        ----------
        mus
            List of parameters for which to estimate the approximation
            error. When parallelization is used, `mus` can be a |RemoteObject|.
        return_all_values
            See below.

        Returns
        -------
        If `return_all_values` is `True`, an |array| of the estimated errors.
        If `return_all_values` is `False`, the maximum estimated error as first
        return value and the corresponding parameter as second return value.
        """
        pass

    @abstractmethod
    def extend(self, mu):
        pass


def rb_greedy(fom, reductor, training_set, use_error_estimator=True, error_norm=None,
              atol=None, rtol=None, max_extensions=None, extension_params=None, pool=None):
    """Weak Greedy basis generation using the RB approximation error as surrogate.

    This algorithm generates a reduced basis using the :func:`weak greedy <weak_greedy>`
    algorithm :cite:`BCDDPW11`, where the approximation error is estimated from computing
    solutions of the reduced order model for the current reduced basis and then estimating
    the model reduction error.

    Parameters
    ----------
    fom
        The |Model| to reduce.
    reductor
        Reductor for reducing the given |Model|. This has to be
        an object with a `reduce` method, such that `reductor.reduce()`
        yields the reduced model, and an `exted_basis` method,
        such that `reductor.extend_basis(U, copy_U=False, **extension_params)`
        extends the current reduced basis by the vectors contained in `U`.
        For an example see :class:`~pymor.reductors.coercive.CoerciveRBReductor`.
    training_set
        The training set of |Parameters| on which to perform the greedy search.
    use_error_estimator
        If `False`, exactly compute the model reduction error by also computing
        the solution of `fom` for all |parameter values| of the training set.
        This is mainly useful when no estimator for the model reduction error
        is available.
    error_norm
        If `use_error_estimator` is `False`, use this function to calculate the
        norm of the error. If `None`, the Euclidean norm is used.
    atol
        See :func:`weak_greedy`.
    rtol
        See :func:`weak_greedy`.
    max_extensions
        See :func:`weak_greedy`.
    extension_params
        `dict` of parameters passed to the `reductor.extend_basis` method.
        If `None`, `'gram_schmidt'` basis extension will be used as a default
        for stationary problems (`fom.solve` returns `VectorArrays` of length 1)
        and `'pod'` basis extension (adding a single POD mode) for instationary
        problems.
    pool
        See :func:`weak_greedy`.

    Returns
    -------
    Dict with the following fields:

        :rom:                    The reduced |Model| obtained for the
                                 computed basis.
        :max_errs:               Sequence of maximum errors during the greedy run.
        :max_err_mus:            The parameters corresponding to `max_errs`.
        :extensions:             Number of performed basis extensions.
        :time:                   Total runtime of the algorithm.
    """
    surrogate = RBSurrogate(fom, reductor, use_error_estimator, error_norm, extension_params, pool or dummy_pool)

    result = weak_greedy(surrogate, training_set, atol=atol, rtol=rtol, max_extensions=max_extensions, pool=pool)
    result['rom'] = surrogate.rom

    return result


class RBSurrogate(WeakGreedySurrogate):
    """Surrogate for the :func:`weak_greedy` error used in :func:`rb_greedy`.

    Not intended to be used directly.
    """

    def __init__(self, fom, reductor, use_error_estimator, error_norm, extension_params, pool):
        self.__auto_init(locals())
        if use_error_estimator:
            self.remote_fom, self.remote_error_norm, self.remote_reductor = None, None, None
        else:
            self.remote_fom, self.remote_error_norm, self.remote_reductor = \
                pool.push(fom), pool.push(error_norm), pool.push(reductor)
        self.rom = None

    def evaluate(self, mus, return_all_values=False):
        if self.rom is None:
            with self.logger.block('Reducing ...'):
                self.rom = self.reductor.reduce()

        if not isinstance(mus, RemoteObject):
            mus = self.pool.scatter_list(mus)

        result = self.pool.apply(_rb_surrogate_evaluate,
                                 rom=self.rom,
                                 fom=self.remote_fom,
                                 reductor=self.remote_reductor,
                                 mus=mus,
                                 error_norm=self.remote_error_norm,
                                 return_all_values=return_all_values)
        if return_all_values:
            return np.hstack(result)
        else:
            errs, max_err_mus = list(zip(*result))
            max_err_ind = np.argmax(errs)
            return errs[max_err_ind], max_err_mus[max_err_ind]

    def extend(self, mu):
        with self.logger.block(f'Computing solution snapshot for mu = {mu} ...'):
            U = self.fom.solve(mu)
        with self.logger.block('Extending basis with solution snapshot ...'):
            extension_params = self.extension_params
            if len(U) > 1:
                if extension_params is None:
                    extension_params = {'method': 'pod'}
                else:
                    extension_params.setdefault('method', 'pod')
            self.reductor.extend_basis(U, copy_U=False, **(extension_params or {}))
            if not self.use_error_estimator:
                self.remote_reductor = self.pool.push(self.reductor)
        with self.logger.block('Reducing ...'):
            self.rom = self.reductor.reduce()


def _rb_surrogate_evaluate(rom=None, fom=None, reductor=None, mus=None, error_norm=None, return_all_values=False):
    if not mus:
        if return_all_values:
            return []
        else:
            return -1., None

    if fom is None:
        errors = [rom.estimate_error(mu) for mu in mus]
    elif error_norm is not None:
        errors = [error_norm(fom.solve(mu) - reductor.reconstruct(rom.solve(mu))) for mu in mus]
    else:
        errors = [(fom.solve(mu) - reductor.reconstruct(rom.solve(mu))).norm() for mu in mus]
    # most error_norms will return an array of length 1 instead of a number,
    # so we extract the numbers if necessary
    errors = [x[0] if hasattr(x, '__len__') else x for x in errors]
    if return_all_values:
        return errors
    else:
        max_err_ind = np.argmax(errors)
        return errors[max_err_ind], mus[max_err_ind]
