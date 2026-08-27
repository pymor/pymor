# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.image import estimate_image
from pymor.operators.block import BlockRowOperator
from pymor.operators.constructions import LincombOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.functionals import ProjectionParameterFunctional

tol = 1e-12


def affine_operator(rng, num_range, num_source):
    ops = [NumpyMatrixOperator(rng.random((num_range, num_source))) for _ in range(2)]
    return LincombOperator(ops, [1., ProjectionParameterFunctional('param')])


def max_residual(op, image, domain, num_params=5):
    basis = image.to_numpy()
    residual = 0.
    parameters = op.parameters.space(0, 1).sample_randomly(num_params)
    for mu in parameters:
        target = op.apply(domain, mu=mu).to_numpy()
        coeffs = np.linalg.lstsq(basis, target, rcond=None)[0]
        residual = max(residual, np.linalg.norm(basis @ coeffs - target))
    return residual


def test_estimate_image_affine(rng):
    op = affine_operator(rng, 6, 3)
    domain = op.source.random(6)
    image = estimate_image([op], domain=domain)
    assert max_residual(op, image, domain) < tol


def test_estimate_image_block_row_operator(rng):
    op = BlockRowOperator([affine_operator(rng, 5, 2), affine_operator(rng, 5, 3)])
    domain = op.source.random(6)
    image = estimate_image([op], domain=domain)
    assert max_residual(op, image, domain) < tol
