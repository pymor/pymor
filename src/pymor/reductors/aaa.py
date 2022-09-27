# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import itertools

from pymor.core.base import BasicObject
from pymor.core.logger import getLogger
from pymor.models.transfer_function import TransferFunction


class pAAAReductor(BasicObject):
    """Reductor implementing the parametric AAA algorithm.

    The reductor implements the parametric AAA algorithm and can be used either with
    data or a given full-order model which has a `transfer_function` attribute. MIMO
    data is accepted.

    Parameters
    ----------
    data
        If `fom` is `None` a tuple where the first value corresponds to sampling values
        and the second value contains samples. Sampling values are represented as a
        nested list `svs` such that `svs[i]` corresponds to sampling values of the `i-th`
        variable. Samples are represented as a tensor `S`. E.g., for 3 inputs `S[i,j,k]`
        corresponds to the sampled value at `(svs[0][i],svs[1][j],svs[2][k])`. In the
        MIMO case `S[i,j,k]` represents a matrix of dimension `dim_input` times `dim_output`.
        If `fom` is not `None` data only contains a list of sampling values.
    fom
        |TransferFunction| or |Model| with a `transfer_function` attribute.
    conjugate
        Wether to compute complex conjugates of first sampling variables and enforce
        interpolation in complex conjugate pairs (allows for constructing real system matrices).
    nsp_tol
        Tolerance for null space of higher-dimensional Loewner matrix to check for
        interpolation or convergence.
    post_process
        Wether to do post-processing or not.
    L_rk_tol
        Tolerance for ranks of 1-D Loewner matrices computed in post-processing.
    """
    def __init__(self, data, fom=None, conjugate=True, nsp_tol=1e-16, post_process=True, L_rk_tol=1e-8):
        if fom is not None:
            assert isinstance(fom, TransferFunction) or hasattr(fom, 'transfer_function')
            if not isinstance(fom, TransferFunction):
                fom = fom.transfer_function
            self.num_vars = 1 + len(fom.parameters)

            assert len(data) == self.num_vars
            self.sampling_values = data
            self.parameters = fom.parameters
            sampling_grid = np.meshgrid(*(sv for sv in data), indexing='ij')
            if fom.dim_input == 1 and fom.dim_output == 1:
                self.samples = np.empty([*(len(sv) for sv in data)], dtype=self.sampling_values[0].dtype)
                for idc in itertools.product(*(range(ss) for ss in self.samples.shape)):
                    params = {}
                    for i, p in enumerate(fom.parameters.keys()):
                        params[p] = sampling_grid[i+1][idc]
                    self.samples[idc] = fom.eval_tf(sampling_grid[0][idc], mu=params)
            else:
                sample_shape = [*(len(sv) for sv in data)]
                sample_shape.append(fom.dim_output)
                sample_shape.append(fom.dim_input)
                self.samples = np.empty(sample_shape, dtype=self.sampling_values[0].dtype)
                for idc in itertools.product(*(range(ss) for ss in self.samples.shape[:-2])):
                    params = {}
                    for i, p in enumerate(fom.parameters.keys()):
                        params[p] = sampling_grid[i+1][idc]
                    self.samples[idc] = fom.eval_tf(sampling_grid[0][idc], mu=params)
        else:
            assert len(data) == 2
            self.sampling_values = data[0]
            self.samples = data[1]
            self.num_vars = len(data[0])
            self.parameters = {}
            for i in range(self.num_vars-1):
                self.parameters['p' + str(i)] = 1

        self.__auto_init(locals())

    def reduce(self, tol=1e-3, max_iters=None):
        logger = getLogger('pymor.reductors.AAA.reduce')

        svs = self.sampling_values
        samples = self.samples

        # add complex conjugate samples
        if self.conjugate:
            for i in range(len(svs[0])):
                s = svs[0][i]
                if np.conj(s) not in svs[0]:
                    svs[0] = np.append(svs[0], np.conj(s))
                    samples = np.vstack((samples, np.conj(samples[i, None])))

        num_vars = len(svs)
        max_samples = np.max(np.abs(samples))
        rel_tol = tol * max_samples

        # Transform samples for MIMO case
        if len(samples.shape) != len(svs):
            assert len(samples.shape) == len(svs) + 2
            dim_input = samples.shape[-1]
            dim_output = samples.shape[-2]
            samples_T = np.empty(samples.shape[:-2], dtype=samples.dtype)
            w = np.random.uniform(size=(1, dim_output))
            v = np.random.uniform(size=(dim_input, 1))
            w = w / np.linalg.norm(w)
            v = v / np.linalg.norm(v)
            for li in list(itertools.product(*(range(s) for s in samples.shape[:-2]))):
                samples_T[li] = w @ samples[li] @ v
            samples_orig = samples
            samples = samples_T
        else:
            dim_input = 1
            dim_output = 1

        # initilize data partitions, error, max iterations
        err = np.inf
        itpl_part = [*([] for _ in range(num_vars))]
        if max_iters is None:
            max_iters = [*(len(s)-1 for s in svs)]

        assert len(max_iters) == len(svs)

        # start iteration with constant function
        bary_func = np.vectorize(lambda *args: np.mean(samples))

        # iteration counter
        j = 0

        while np.any([*(len(i) < mi for (i, mi) in zip(itpl_part, max_iters))]):

            # compute approximation error over entire sampled data set
            grid = np.meshgrid(*(sv for sv in svs), indexing='ij')
            err_mat = np.abs(bary_func(*(g for g in grid))-samples)

            # set errors to zero such that new interpolation points are consistent with max_iters
            zero_idx = []
            for i in range(num_vars):
                if len(itpl_part[i]) >= max_iters[i]:
                    zero_idx.append(list(range(samples.shape[i])))
                else:
                    zero_idx.append(itpl_part[i])
            err_mat[np.ix_(*(zi for zi in zero_idx))] = 0
            err = np.max(err_mat)

            j += 1
            logger.info(f'Relative error at step {j}: {err/max_samples:.5e}, \
                number of interpolation points {[*(len(ip) for ip in itpl_part)]}')

            # stopping criterion based on relative approximation error
            if err <= rel_tol:
                break

            greedy_idx = np.unravel_index(err_mat.argmax(), err_mat.shape)
            for i in range(num_vars):
                if greedy_idx[i] not in itpl_part[i] and len(itpl_part[i]) < max_iters[i]:
                    itpl_part[i].append(greedy_idx[i])

                    # perform double interpolation step to enforce real state-space representation
                    if i == 0 and self.conjugate and np.imag(svs[i][greedy_idx[i]]) != 0:
                        conj_sample = np.conj(svs[i][greedy_idx[i]])
                        conj_idx = np.where(svs[0] == conj_sample)[0]
                        itpl_part[i].append(conj_idx[0])

            # solve LS problem
            L = full_nd_loewner(samples, svs, itpl_part)

            _, S, V = np.linalg.svd(L)
            VH = np.conj(V.T)
            coefs = VH[:, -1:]

            # post-processing for non-minimal interpolants
            d_nsp = np.sum(S/S[0] < self.nsp_tol)
            if d_nsp > 1:
                if self.post_process:
                    logger.info('Non-minimal order interpolant computed. Starting post-processing.')
                    pp_coefs, pp_itpl_part = _post_processing(samples, svs, itpl_part, d_nsp, self.L_rk_tol)
                    if pp_coefs is not None:
                        coefs, itpl_part = pp_coefs, pp_itpl_part
                    else:
                        logger.warning('Post-processing failed. Consider reducing "L_rk_tol".')
                else:
                    logger.warning('Non-minimal order interpolant computed.')

            # update barycentric form
            itpl_samples = samples[np.ix_(*(ip for ip in itpl_part))]
            itpl_samples = np.reshape(itpl_samples, -1)
            itpl_nodes = [*(sv[lp] for sv, lp in zip(svs, itpl_part))]
            bary_func = np.vectorize(make_bary_func(itpl_nodes, itpl_samples, coefs))

            if self.post_process and d_nsp >= 1:
                logger.info('Converged due to non-trivial null space of Loewner matrix after post-processing.')
                break

        # in MIMO case construct barycentric form based on matrix/vector samples
        if dim_input != 1 or dim_output != 1:
            itpl_samples = samples_orig[np.ix_(*(ip for ip in itpl_part))]
            itpl_samples = np.reshape(itpl_samples, (-1, dim_output, dim_input))

        bary_func = make_bary_func(itpl_nodes, itpl_samples, coefs)

        if self.num_vars > 1:
            return TransferFunction(dim_input, dim_output,
                                    lambda s, mu: bary_func(s, *(mu[p] for p in self.parameters)),
                                    parameters=self.parameters)
        else:
            return TransferFunction(dim_input, dim_output, lambda s: bary_func(s))


def nd_loewner(samples, svs, itpl_part):
    """Compute higher-dimensional Loewner matrix using only LS partition."""
    d = len(samples.shape)
    ls_part = _ls_part(itpl_part, svs)

    s0 = svs[0]
    s = s0[itpl_part[0]]
    sh = s0[ls_part[0]]
    sd = sh[:, np.newaxis] - s[np.newaxis]
    sdpd = sd
    for i in range(1, d):
        p0 = svs[i]
        p = p0[itpl_part[i]]
        ph = p0[ls_part[i]]
        pd = ph[:, np.newaxis] - p[np.newaxis]
        sdpd = np.kron(sdpd, pd)
    samples0 = samples[np.ix_(*(p for p in itpl_part))]
    samples0 = np.reshape(samples0, (-1, np.prod(samples0.shape)))
    samples1 = samples[np.ix_(*(p for p in ls_part))]
    samples1 = np.reshape(samples1, (-1, np.prod(samples1.shape)))
    samplesd = samples1.T - samples0

    return samplesd / sdpd


def full_nd_loewner(samples, svs, itpl_part):
    """Compute higher-dimensional Loewner matrix taking all errors into account."""
    L = nd_loewner(samples, svs, itpl_part)
    range_S = range(len(svs))

    # consider all combinations of variables coming from interpolation vs LS partition
    for i in itertools.product(*([0, 1] for _ in range_S)):

        # skip cases corresponding to all interpolated or all LS fit
        if i == tuple(len(svs)*[0]) or i == tuple(len(svs)*[1]):
            continue

        for j in itertools.product(*(itpl_part[k] for k in range_S if i[k] == 0)):
            l_j = list(j)
            for ii in range(len(i)):
                if i[ii] == 1:
                    l_j.insert(ii, slice(None))
            samples0 = samples[tuple(l_j)]
            svs0 = [svs[k] for k in range_S if i[k] == 1]
            itpl_part0 = [itpl_part[k] for k in range_S if i[k] == 1]
            T_mat = 1
            for k in range_S:
                if i[k] == 1:
                    T_new = np.eye(len(itpl_part[k]))
                else:
                    idx = np.where(np.array(itpl_part[k]) == l_j[k])[0][0]
                    T_new = np.eye(1, len(itpl_part[k]), idx)
                T_mat = np.kron(T_mat, T_new)
            LL = nd_loewner(samples0, svs0, itpl_part0)
            L = np.vstack([L, LL @ T_mat])
    return L


def make_bary_func(itpl_nodes, itpl_vals, coefs):
    """Return function handle for multivariate barycentric form."""
    def bary_func(*args):
        pd = 1
        # this loop is for pole cancellation which occurs at interpolation nodes
        for i in range(len(itpl_nodes)):
            d = args[i] - itpl_nodes[i]
            d_zero = d[np.abs(d) < 1e-14]
            if len(d_zero) > 0:
                d_min_idx = np.argmin(np.abs(d))
                d = np.eye(1, len(d), d_min_idx)
            else:
                d = np.reciprocal(d)
            pd = np.kron(pd, d)
        m = np.multiply(coefs.T, pd)
        num = np.tensordot(m, itpl_vals, axes=1)
        denom = np.inner(coefs.T, pd)
        nd = num / denom
        return np.atleast_2d(nd[0])

    return bary_func


def _ls_part(itpl_part, svs):
    """Compute least-squares partition based on interpolation partition."""
    ls_part = []
    for p, s in zip(itpl_part, svs):
        idx = np.arange(len(s))
        idx = np.delete(idx, p)
        ls_part.append(list(idx))
    return ls_part


def _post_processing(samples, svs, itpl_part, d_nsp, L_rk_tol):
    """Compute coefficients/partition to construct minimal interpolant."""
    num_vars = len(svs)
    max_idx = np.argmax([*(len(ip) for ip in itpl_part)])
    max_rks = []
    for i in range(num_vars):
        max_rk = 0
        # we don't need to compute this max rank since we exploit nullspace structure
        if i == max_idx:
            max_rks.append(len(itpl_part[max_idx])-1)
            continue
        shapes = []
        for j in range(num_vars):
            if i != j:
                shapes.append(samples.shape[j])
        # compute max ranks of all possible 1-D Loewner matrices
        for idc in itertools.product(*(range(s) for s in shapes)):
            l_idc = list(idc)
            l_idc.insert(i, slice(None))
            L = nd_loewner(samples[tuple(l_idc)], [svs[i]], [itpl_part[i]])
            rk = np.linalg.matrix_rank(L, tol=L_rk_tol)
            if rk > max_rk:
                max_rk = rk
        max_rks.append(max_rk)
    # exploit nullspace structure to obtain final max rank
    denom = np.prod([*(len(itpl_part[k])-max_rks[k] for k in range(len(itpl_part)))])
    if denom == 0 or d_nsp % denom != 0:
        return None, None
    max_rks[max_idx] = len(itpl_part[max_idx]) - d_nsp / denom
    max_rks[max_idx] = round(max_rks[max_idx])
    for i in range(len(max_rks)):
        itpl_part[i] = itpl_part[i][0:max_rks[i]+1]

    # solve LS problem
    L = full_nd_loewner(samples, svs, itpl_part)
    _, S, V = np.linalg.svd(L)
    VH = np.conj(V.T)
    coefs = VH[:, -1:]

    return coefs, itpl_part
