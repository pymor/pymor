# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import itertools

import numpy as np
import scipy.linalg as spla

from pymor.bindings.scipy import svd_lapack_driver
from pymor.core.base import BasicObject
from pymor.models.transfer_function import TransferFunction
from pymor.tools.random import new_rng


class PAAAReductor(BasicObject):
    """Reductor implementing the parametric AAA algorithm.

    The reductor implements the parametric AAA algorithm and can be used either with
    data or a given full-order model, which can be a |TransferFunction| or any model which
    has a `transfer_function` attribute. MIMO and non-parametric data is accepted. See
    :cite:`NST18` for the non-parametric and :cite:`CRBG23` for the parametric version of
    the algorithm. The reductor computes approximations based on the multivariate barycentric
    form :math:`H(s,p,...)` (see :func:`~pymor.reductors.aaa.make_bary_func`), where
    the inputs :math:`s,p,...` are referred to as the variables. Further, :math:`s` is
    the Laplace variable and :math:`p` as well as other remaining inputs are parameters.

    .. note::
        The dimension of the Loewner matrix which will be assembled in the algorithm has
        `len(sampling_values[0])*...*len(sampling_values[-1])` rows. This reductor should
        only be used with a low number of variables.

    Parameters
    ----------
    sampling_values
        Values where sample data has been evaluated or the full-order model should be evaluated.
        Sampling values are represented as a list of |NumPy arrays| such that `sampling_values[i]`
        corresponds to sampling values of the `i`-th variable given as a |NumPy array|. The first
        variable is the Laplace variable. In the non-parametric case (i.e., the only variable is
        the Laplace variable) this can also be a |NumPy array| representing the sampling values.
    samples_or_fom
        Can be either a full-order model (|TransferFunction| or |Model| with a `transfer_function`
        attribute) or data sampled at the values specified in `sampling_values` as a |NumPy array|.
        Samples are represented as a tensor `S`. E.g., for 3 inputs `S[i,j,k]` corresponds to the
        sampled value at `(sampling_values[0][i],sampling_values[1][j],sampling_values[2][k])`.
        The samples (i.e., `S[i,j,k]`) need to be provided as 2-dimensional |NumPy arrays|. E.g.,
        in the MIMO case `S[i,j,k]` represents a matrix of dimension `dim_output` times `dim_input`.
    conjugate
        Whether to compute complex conjugates of first sampling variables and enforce
        interpolation in complex conjugate pairs (allows for constructing real system matrices).
    nsp_tol
        Tolerance for null space of higher-dimensional Loewner matrix to check for
        interpolation or convergence.
    post_process
        Whether to do post-processing or not. If the Loewner matrix has a null space
        of dimension greater than 1, it is assumed that the algorithm converged to
        a non-minimal order interpolant which may cause numerical issues. In this case,
        the post-processing procedure computes an interpolant of minimal order.
    L_rk_tol
        Tolerance for ranks of 1-D Loewner matrices computed in post-processing.

    Attributes
    ----------
    itpl_part
        A nested list such that `itpl_part[i]` corresponds to indices of interpolated values
        with respect to the `i`-th variable. I.e., `self.sampling_values[i][itpl_part[i]]`
        represents a list of all interpolated samples of the `i`-th variable.
    """

    def __init__(self, sampling_values, samples_or_fom, conjugate=True, nsp_tol=1e-16, post_process=True,
                 L_rk_tol=1e-8):
        if isinstance(sampling_values, np.ndarray):
            sampling_values = [sampling_values]
        assert isinstance(sampling_values, list)
        assert all(isinstance(sv, np.ndarray) for sv in sampling_values)
        if isinstance(samples_or_fom, TransferFunction) or hasattr(samples_or_fom, 'transfer_function'):
            fom = samples_or_fom
            if not isinstance(samples_or_fom, TransferFunction):
                fom = fom.transfer_function
            self.num_vars = 1 + fom.parameters.dim

            assert len(sampling_values) == self.num_vars
            self._parameters = fom.parameters
            self.samples = np.empty([len(sv) for sv in sampling_values] + [fom.dim_output, fom.dim_input],
                                    dtype=sampling_values[0].dtype)
            for idx, vals in zip(np.ndindex(self.samples.shape[:-2]),
                                 itertools.product(*sampling_values)):
                params = fom.parameters.parse(vals[1:])
                self.samples[idx] = fom.eval_tf(vals[0], mu=params)
            if fom.dim_input == fom.dim_output == 1:
                self.samples = self.samples.reshape(self.samples.shape[:-2])
        else:
            self.samples = samples_or_fom
            # SISO case requires reshape
            if self.samples.shape[-2:] == (1, 1):
                self.samples = self.samples.reshape(self.samples.shape[:-2])
            self.num_vars = len(sampling_values)
            self._parameters = {'p': self.num_vars-1}

        # add complex conjugate samples
        if conjugate:
            s_conj_list = []
            samples_conj_list = None
            for i, s in enumerate(sampling_values[0]):
                if s.conj() not in sampling_values[0]:
                    s_conj_list.append(s.conj())
                    if samples_conj_list is None:
                        samples_conj_list = self.samples[i, None].conj()
                    else:
                        samples_conj_list = np.concatenate((samples_conj_list, self.samples[i, None].conj()))
            if s_conj_list:
                sampling_values[0] = np.append(sampling_values[0], s_conj_list)
                self.samples = np.concatenate((self.samples, samples_conj_list))

        # Transform samples for MIMO case
        if len(self.samples.shape) != len(sampling_values):
            assert len(self.samples.shape) == len(sampling_values) + 2
            self._dim_input = self.samples.shape[-1]
            self._dim_output = self.samples.shape[-2]
            samples_T = np.empty(self.samples.shape[:-2], dtype=self.samples.dtype)
            rng = new_rng(0)
            if any(np.iscomplex(sampling_values[0])):
                w = 1j * rng.normal(scale=np.sqrt(2)/2, size=(self._dim_output,)) \
                    + rng.normal(scale=np.sqrt(2)/2, size=(self._dim_output,))
                v = 1j * rng.normal(scale=np.sqrt(2)/2, size=(self._dim_input,)) \
                    + rng.normal(scale=np.sqrt(2)/2, size=(self._dim_input,))
            else:
                w = rng.normal(size=(self._dim_output,))
                v = rng.normal(size=(self._dim_input,))
            w /= np.linalg.norm(w)
            v /= np.linalg.norm(v)
            samples_T = self.samples @ v @ w
            self.MIMO_samples = self.samples
            self.samples = samples_T
        else:
            self._dim_input = 1
            self._dim_output = 1

        self.__auto_init(locals())

    def reduce(self, tol=1e-7, itpl_part=None, max_itpl=None):
        """Reduce using p-AAA.

        Parameters
        ----------
        tol
            Convergence tolerance for relative error of `rom` over the set of samples.
        itpl_part
            Initial partition for interpolation values. Should be `None` or a nested list
            such that `itpl_part[i]` corresponds to indices of interpolated values with
            respect to the `i`-th variable. I.e., `self.sampling_values[i][itpl_part[i]]`
            represents a list of all initially interpolated samples of the `i`-th variable.
            If `None` p-AAA will start with no interpolated values.
        max_itpl
            Maximum number of interpolation points to use with respect to each
            variable. Should be `None` or a list such that `self.num_vars == len(max_itpl)`.
            If `None` `max_itpl[i]` will be set to `len(self.sampling_values[i]) - 1`.

        Returns
        -------
        rom
            Reduced |TransferFunction| model.
        """
        svs = self.sampling_values
        samples = self.samples

        max_samples = np.max(np.abs(samples))
        rel_tol = tol * max_samples

        # initialize data partitions, error, max iterations
        err = np.inf
        if itpl_part is None:
            self.itpl_part = [[] for _ in range(self.num_vars)]
        else:
            assert len(itpl_part) == self.num_vars
            self.itpl_part = itpl_part
        if max_itpl is None:
            max_itpl = [len(s)-1 for s in svs]

        assert len(max_itpl) == len(svs)

        # start iteration with constant function
        bary_func = lambda *args: np.mean(samples)

        # iteration counter
        j = 0

        while any(len(i) < mi for i, mi in zip(self.itpl_part, max_itpl)):

            # compute approximation error over entire sampled data set
            err_mat = np.empty(samples.shape)
            for idx, vals in zip(np.ndindex(*[len(sv) for sv in svs]), itertools.product(*svs)):
                err_mat[idx] = np.abs(np.squeeze(bary_func(*vals)) - samples[idx])

            # set errors to zero such that new interpolation points are consistent with max_itpl
            zero_idx = []
            for i in range(self.num_vars):
                if len(self.itpl_part[i]) >= max_itpl[i]:
                    zero_idx.append(list(range(samples.shape[i])))
                else:
                    zero_idx.append(self.itpl_part[i])
            err_mat[np.ix_(*zero_idx)] = 0
            err = np.max(err_mat)

            j += 1
            self.logger.info(f'Relative error at step {j}: {err/max_samples:.5e}, '
                             f'number of interpolation points {[len(ip) for ip in self.itpl_part]}')

            # stopping criterion based on relative approximation error
            if err <= rel_tol:
                break

            greedy_idx = np.unravel_index(err_mat.argmax(), err_mat.shape)
            for i in range(self.num_vars):
                if greedy_idx[i] not in self.itpl_part[i] and len(self.itpl_part[i]) < max_itpl[i]:
                    self.itpl_part[i].append(greedy_idx[i])

                    # perform double interpolation step to allow real state-space representation
                    if i == 0 and self.conjugate and np.imag(svs[i][greedy_idx[i]]) != 0:
                        conj_sample = np.conj(svs[i][greedy_idx[i]])
                        conj_idx = np.where(svs[0] == conj_sample)[0]
                        self.itpl_part[i].append(conj_idx[0])

            # solve LS problem
            L = full_nd_loewner(samples, svs, self.itpl_part)

            _, S, V = spla.svd(L, full_matrices=False, lapack_driver=svd_lapack_driver())
            VH = V.T.conj()
            coefs = VH[:, -1]

            # post-processing for non-minimal interpolants
            d_nsp = np.sum(S/S[0] < self.nsp_tol)
            if d_nsp > 1:
                if self.post_process:
                    self.logger.info('Non-minimal order interpolant computed. Starting post-processing.')
                    pp_coefs = self._post_processing(d_nsp)
                    if pp_coefs is not None:
                        coefs = pp_coefs
                    else:
                        self.logger.warning('Post-processing failed. Consider reducing "L_rk_tol".')
                else:
                    self.logger.warning('Non-minimal order interpolant computed.')

            # update barycentric form
            itpl_samples = samples[np.ix_(*self.itpl_part)]
            itpl_samples = np.reshape(itpl_samples, -1)
            itpl_nodes = [sv[lp] for sv, lp in zip(svs, self.itpl_part)]
            bary_func = make_bary_func(itpl_nodes, itpl_samples, coefs)

            if self.post_process and d_nsp >= 1:
                self.logger.info('Converged due to non-trivial null space of Loewner matrix after post-processing.')
                break

        # in MIMO case construct barycentric form based on matrix/vector samples
        if self._dim_input != 1 or self._dim_output != 1:
            itpl_samples = self.MIMO_samples[np.ix_(*self.itpl_part)]
            itpl_samples = np.reshape(itpl_samples, (-1, self._dim_output, self._dim_input))

        bary_func = make_bary_func(itpl_nodes, itpl_samples, coefs)

        if self.num_vars > 1:
            return TransferFunction(self._dim_input, self._dim_output,
                                    lambda s, mu: bary_func(s, *mu.to_numpy()),
                                    parameters=self._parameters)
        else:
            return TransferFunction(self._dim_input, self._dim_output, bary_func)

    def _post_processing(self, d_nsp):
        """Compute coefficients/partition to construct minimal interpolant."""
        max_idx = np.argmax([len(ip) for ip in self.itpl_part])
        max_rks = []
        for i in range(self.num_vars):
            max_rk = 0
            # we don't need to compute this max rank since we exploit nullspace structure
            if i == max_idx:
                max_rks.append(len(self.itpl_part[max_idx])-1)
                continue
            shapes = []
            for j in range(self.num_vars):
                if i != j:
                    shapes.append(self.samples.shape[j])
            # compute max ranks of all possible 1-D Loewner matrices
            for idc in itertools.product(*(range(s) for s in shapes)):
                l_idc = list(idc)
                l_idc.insert(i, slice(None))
                L = full_nd_loewner(self.samples[tuple(l_idc)], [self.sampling_values[i]], [self.itpl_part[i]])
                rk = np.linalg.matrix_rank(L, tol=self.L_rk_tol)
                if rk > max_rk:
                    max_rk = rk
            max_rks.append(max_rk)
        # exploit nullspace structure to obtain final max rank
        denom = np.prod([len(self.itpl_part[k])-max_rks[k] for k in range(len(self.itpl_part))])
        if denom == 0 or d_nsp % denom != 0:
            return None
        max_rks[max_idx] = len(self.itpl_part[max_idx]) - d_nsp / denom
        max_rks[max_idx] = round(max_rks[max_idx])
        for i in range(len(max_rks)):
            self.itpl_part[i] = self.itpl_part[i][0:max_rks[i]+1]

        # solve LS problem
        L = full_nd_loewner(self.samples, self.sampling_values, self.itpl_part)
        _, S, V = spla.svd(L, full_matrices=False, lapack_driver=svd_lapack_driver())
        VH = np.conj(V.T)
        coefs = VH[:, -1]

        return coefs


def _cauchy_itpl(s, itpl_part):
    """Compute a modified Cauchy matrix for Loewner matrix construction."""
    N = s.shape[0]
    k = len(itpl_part)
    ls_part = sorted(set(range(len(s))) - set(itpl_part))

    C = np.zeros((k, N), dtype=s.dtype)
    C[:, itpl_part] = np.eye(k)
    C[:, ls_part] = 1.0 / (s[ls_part] - s[itpl_part][:, None])

    return C.T

def full_nd_loewner(samples, svs, itpl_part):
    """Compute higher-dimensional Loewner matrix using all combinations of partitions.

    .. note::
       For non-parametric data this is simply the regular Loewner matrix.

    Parameters
    ----------
    samples
        Tensor of samples (see :class:`PAAAReductor`).
    svs
        List of sampling values (see :class:`PAAAReductor`).
    itpl_part
        Nested list such that `itpl_part[i]` is a list of indices for interpolated
        sampling values in `svs[i]`.

    Returns
    -------
    L
        (Parametric) Loewner matrix based on all combinations of partitions.
    """
    itpl_samples = samples[np.ix_(*itpl_part)]

    kron_C = 1
    zr_idc = True
    for i in range(len(svs)):
        # form modified Cauchy matrix kronecker product
        C = _cauchy_itpl(svs[i], itpl_part[i])
        kron_C = np.kron(kron_C, C)

        # construction of zero rows indices
        zr_idx = np.zeros(len(svs[i]), dtype=bool)
        zr_idx[itpl_part[i]] = 1
        zr_idc = np.kron(zr_idx,zr_idc)

    L = samples.reshape(-1,1) * kron_C - (itpl_samples.reshape(-1,1) * kron_C.T).T

    return L[np.invert(zr_idc),:]


def make_bary_func(itpl_nodes, itpl_vals, coefs, removable_singularity_tol=1e-14):
    r"""Return function for (multivariate) barycentric form.

    The multivariate barycentric form for two variables is given by

    .. math::
        H(s,p) = \frac{\sum_{i=1}^{k}\sum_{j=1}^{q}\frac{\alpha_{ij}H(s_i,p_j)}{(s-s_i)(p-p_j)}}
            {\sum_{i=1}^{k}\sum_{j=1}^{q}\frac{\alpha_{ij}}{(s-s_i)(p-p_j)}}

    where for :math:`i=1,\ldots,k` and :math:`j=1,\ldots,q` we have that :math:`s_i` and :math:`p_j`
    are interpolation nodes, :math:`H(s_i,p_j)` are interpolation values and :math:`\alpha_{ij}`
    represent the barycentric coefficients. This implementation can also handle single-variable
    barycentric forms as well as barycentric forms with more than two variables.

    Parameters
    ----------
    itpl_nodes
        Nested list such that `itpl_nodes[i]` contains interpolation nodes of the
        `i`-th variable.
    itpl_vals
        Vector of interpolation values with `len(itpl_nodes[0])*...*len(itpl_nodes[-1])`
        entries.
    coefs
        Vector of barycentric coefficients with `len(itpl_nodes[0])*...*len(itpl_nodes[-1])`
        entries.
    removable_singularity_tol
        Tolerance for evaluating the barycentric function at a removable singularity
        and performing pole cancellation.

    Returns
    -------
    bary_func
        (Multi-variate) rational function in barycentric form.
    """
    def bary_func(*args):
        pd = 1
        # this loop is for pole cancellation which occurs at interpolation nodes
        for arg, itpl_node in zip(args, itpl_nodes):
            d = arg - itpl_node
            d_zero = d[np.abs(d) < removable_singularity_tol]
            if len(d_zero) > 0:
                d_min_idx = np.argmin(np.abs(d))
                d = np.eye(1, len(d), d_min_idx)
            else:
                d = 1 / d
            pd = np.kron(pd, d)
        coefs_pd = coefs * pd
        num = np.tensordot(coefs_pd, itpl_vals, axes=1)
        denom = np.sum(coefs_pd)
        nd = num / denom
        return np.atleast_2d(nd)

    return bary_func
