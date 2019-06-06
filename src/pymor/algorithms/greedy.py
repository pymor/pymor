# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import time

import numpy as np

from pymor.core.exceptions import ExtensionError
from pymor.core.logger import getLogger
from pymor.parallel.dummy import dummy_pool
from pymor.parallel.manager import RemoteObjectManager


def greedy(
    fom,
    reductor,
    samples,
    use_estimator=True,
    error_norm=None,
    atol=None,
    rtol=None,
    max_extensions=None,
    extension_params=None,
    pool=None,
):
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
    fom
        The |Model| to reduce.
    reductor
        Reductor for reducing the given |Model|. This has to be
        an object with a `reduce` method, such that `reductor.reduce()`
        yields the reduced model, and an `exted_basis` method,
        such that `reductor.extend_basis(U, copy_U=False, **extension_params)`
        extends the current reduced basis by the vectors contained in `U`.
        For an example see :class:`~pymor.reductors.coercive.CoerciveRBReductor`.
    samples
        The set of |Parameter| samples on which to perform the greedy search.
    use_estimator
        If `True`, use `rom.estimate()` to estimate the errors on the
        sample set. Otherwise `fom.solve()` is called to compute the exact
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

        :rom:                    The reduced |Model| obtained for the
                                 computed basis.
        :max_errs:               Sequence of maximum errors during the greedy run.
        :max_err_mus:            The parameters corresponding to `max_errs`.
        :extensions:             Number of performed basis extensions.
        :time:                   Total runtime of the algorithm.
    """

    logger = getLogger("pymor.algorithms.greedy.greedy")
    samples = list(samples)
    sample_count = len(samples)
    extension_params = extension_params or {}
    logger.info(f"Started greedy search on {sample_count} samples")
    if pool is None or pool is dummy_pool:
        pool = dummy_pool
    else:
        logger.info(f"Using pool of {len(pool)} workers for parallel greedy search")

    with RemoteObjectManager() as rom:
        # Push everything we need during the greedy search to the workers.
        # Distribute the training set evenly among the workes.
        if not use_estimator:
            rom.manage(pool.push(fom))
            if error_norm:
                rom.manage(pool.push(error_norm))
        samples = rom.manage(pool.scatter_list(samples))

        tic = time.time()
        extensions = 0
        max_errs = []
        max_err_mus = []

        while True:
            with logger.block("Reducing ..."):
                rom = reductor.reduce()

            if sample_count == 0:
                logger.info("There is nothing else to do for empty samples.")
                return {
                    "rom": rom,
                    "max_errs": [],
                    "max_err_mus": [],
                    "extensions": 0,
                    "time": time.time() - tic,
                }

            with logger.block("Estimating errors ..."):
                if use_estimator:
                    errors, mus = list(
                        zip(
                            *pool.apply(
                                _estimate,
                                rom=rom,
                                fom=None,
                                reductor=None,
                                samples=samples,
                                error_norm=None,
                            )
                        )
                    )
                else:
                    errors, mus = list(
                        zip(
                            *pool.apply(
                                _estimate,
                                rom=rom,
                                fom=fom,
                                reductor=reductor,
                                samples=samples,
                                error_norm=error_norm,
                            )
                        )
                    )
            max_err_ind = np.argmax(errors)
            max_err, max_err_mu = errors[max_err_ind], mus[max_err_ind]

            max_errs.append(max_err)
            max_err_mus.append(max_err_mu)
            logger.info(
                f"Maximum error after {extensions} extensions: {max_err} (mu = {max_err_mu})"
            )

            if atol is not None and max_err <= atol:
                logger.info(
                    f"Absolute error tolerance ({atol}) reached! Stoping extension loop."
                )
                break

            if rtol is not None and max_err / max_errs[0] <= rtol:
                logger.info(
                    f"Relative error tolerance ({rtol}) reached! Stoping extension loop."
                )
                break

            with logger.block(f"Computing solution snapshot for mu = {max_err_mu} ..."):
                U = fom.solve(max_err_mu)
            with logger.block("Extending basis with solution snapshot ..."):
                try:
                    reductor.extend_basis(U, copy_U=False, **extension_params)
                except ExtensionError:
                    logger.info("Extension failed. Stopping now.")
                    break
            extensions += 1

            logger.info("")

            if max_extensions is not None and extensions >= max_extensions:
                logger.info(f"Maximum number of {max_extensions} extensions reached.")
                with logger.block("Reducing once more ..."):
                    rom = reductor.reduce()
                break

        tictoc = time.time() - tic
        logger.info(f"Greedy search took {tictoc} seconds")
        return {
            "rom": rom,
            "max_errs": max_errs,
            "max_err_mus": max_err_mus,
            "extensions": extensions,
            "time": tictoc,
        }


def _estimate(rom=None, fom=None, reductor=None, samples=None, error_norm=None):
    if not samples:
        return -1.0, None

    if fom is None:
        errors = [rom.estimate(rom.solve(mu), mu) for mu in samples]
    elif error_norm is not None:
        errors = [
            error_norm(fom.solve(mu) - reductor.reconstruct(rom.solve(mu)))
            for mu in samples
        ]
    else:
        errors = [
            (fom.solve(mu) - reductor.reconstruct(rom.solve(mu))).l2_norm()
            for mu in samples
        ]
    # most error_norms will return an array of length 1 instead of a number, so we extract the numbers
    # if necessary
    errors = [x[0] if hasattr(x, "__len__") else x for x in errors]
    max_err_ind = np.argmax(errors)

    return errors[max_err_ind], samples[max_err_ind]
