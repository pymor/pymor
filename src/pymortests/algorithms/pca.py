# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from hypothesis import HealthCheck, assume, settings
from hypothesis.strategies import sampled_from
import pytest
import scipy.linalg as spla

from pymor.algorithms.basic import almost_equal, contains_zero_vector
from pymor.algorithms.pod import pod
from pymor.algorithms.pca import pca
from pymortests.strategies import given_vector_arrays

methods = ['method_of_snapshots', 'qr_svd']


@settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much,
          HealthCheck.too_slow, HealthCheck.data_too_large])
@given_vector_arrays(method=sampled_from(methods))
def test_pca(vector_array, method):
    """PCA should subtract the empirical mean and call POD on the mean-centered data.

    Verifies:
    - A is not modified
    - returned mean equals the empirical mean
    - modes and singular values equal POD applied to the mean-centered data
    """
    A = vector_array
    assume(len(A) > 0 and (len(A) > 1 or A.dim > 1))
    assume(not contains_zero_vector(A, rtol=1e-13, atol=1e-13))

    B = A.copy()

    # compute expected mean and centered data
    mean_expected = A.mean()
    A_mean_expected = A - mean_expected

    # reference POD on centered data
    U_ref, s_ref = pod(A_mean_expected, method=method)

    # call PCA
    pod_results, mean_pca = pca(A, method=method)

    # input must not be modified
    assert np.all(almost_equal(A, B))

    # mean must match
    assert np.all(almost_equal(mean_pca, mean_expected))

    # modes and singular values must match POD on centered data
    assert len(pod_results[0]) == len(pod_results[1]) == len(U_ref) == len(s_ref)
    assert np.allclose(pod_results[1], s_ref, rtol=1e-12, atol=1e-12)
    # orthonormality of returned modes (sanity)
    assert np.allclose(pod_results[0].gramian(), np.eye(len(pod_results[1])), atol=1e-10)

@settings(deadline=None, suppress_health_check=[HealthCheck.filter_too_much,
          HealthCheck.too_slow, HealthCheck.data_too_large])
@given_vector_arrays(method=sampled_from(methods))
def test_pca_with_coefficients(vector_array, method):
    """PCA should subtract the empirical mean and call POD on the mean-centered data.

    Verifies:
    - A is not modified
    - returned mean equals the empirical mean
    - modes and singular values equal POD applied to the mean-centered data
    """
    A = vector_array
    assume(len(A) > 0 and (len(A) > 1 or A.dim > 1))
    assume(not contains_zero_vector(A, rtol=1e-13, atol=1e-13))

    B = A.copy()

    # compute expected mean and centered data
    mean_expected = A.mean()
    A_mean_expected = A - mean_expected

    # reference POD on centered data
    U_ref, s_ref, c_ref = pod(A_mean_expected, method=method, return_reduced_coefficients=True)

    # call PCA
    pod_results, mean_pca = pca(A, method=method, return_reduced_coefficients=True)

    # input must not be modified
    assert np.all(almost_equal(A, B))

    # mean must match
    assert np.all(almost_equal(mean_pca, mean_expected))

    # modes and singular values must match POD on centered data
    assert len(pod_results[0]) == len(pod_results[1]) == len(U_ref) == len(s_ref)
    assert np.allclose(pod_results[1], s_ref, rtol=1e-12, atol=1e-12)
    # orthonormality of returned modes (sanity)
    assert np.allclose(pod_results[0].gramian(), np.eye(len(pod_results[1])), atol=1e-10)
    # reconstruction check of mean centered data
    U_ref.scal(s_ref)
    UsVh_ref = U_ref.lincomb(c_ref)
    pod_results[0].scal(pod_results[1])
    UsVh_pca = pod_results[0].lincomb(pod_results[2])
    assert spla.norm((UsVh_ref - UsVh_pca).norm()) / spla.norm(A_mean_expected.norm()) < 1e-8
    # reconstruction check of original data
    recon = UsVh_pca + mean_pca
    assert spla.norm((A - recon).norm()) / spla.norm(A.norm()) < 1e-8