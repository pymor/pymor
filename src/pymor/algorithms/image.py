# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.core.exceptions import ImageCollectionError
from pymor.core.logger import getLogger
from pymor.operators.constructions import AdjointOperator, Concatenation, LincombOperator, SelectionOperator
from pymor.operators.ei import EmpiricalInterpolatedOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace, NumpyVectorArray


def estimate_image(operators=tuple(), functionals=tuple(), domain=None, extends=False,
                   orthonormalize=True, product=None, riesz_representatives=False):
    """Estimate the image of given operators for all mu.

    Given lists of |Operators| and |Functionals| and a |VectorArray| `domain` of
    vectors in the source space, this algorithms determines a |VectorArray| `image` of
    range vectors such that:

        For all possible |Parameters| `mu` and all |VectorArrays| `U` contained in the
        linear span of `domain`, `op.apply(U, mu=mu)` and `f.as_vector(mu)` are contained
        in the linear span of `image` for each of the given functionals and operators.

    The algorithm will try to choose `image` as small as possible. However, no optimality
    is guaranteed.

    Parameters
    ----------
    operators
        See above.
    functionals
        See above.
    domain
        See above. If `None`, an empty `domain` |VectorArray| is assumed.
    extends
        For some operators, e.g. |EmpiricalInterpolatedOperator|, as well as for all
        `functionals`, `image` is estimated independently from the choice of `domain`.
        If `extends` is `True`, such operators are ignored. (This is useful in case
        these vectors have already been obtained by earlier calls to this function.)
    orthonormalize
        Compute an orthonormal basis for the linear span of `image` using the
        :func:`~pymor.algorithms.gram_schmidt.gram_schmidt` algorithm.
    product
        Inner product |Operator| w.r.t. which to orthonormalize.
    riesz_representatives
        If `True`, compute Riesz representatives of the vectors in `image` before
        orthonormalizing. (Useful for dual norm computation.)

    Returns
    -------
    The |VectorArray| `image`.

    Raises
    ------
    ImageCollectionError
        Is raised when for a given |Operator| or |Functional| no image estimate
        is possible.
    """
    assert operators or functionals
    domain_space = operators[0].source if operators else functionals[0].source
    image_space = operators[0].range if operators else functionals[0].source
    assert all(f.range == NumpyVectorSpace(1) and f.source == image_space and f.linear for f in functionals)
    assert all(op.source == domain_space and op.range == image_space for op in operators)
    assert domain is None or domain in domain_space
    assert product is None or product.source == product.range == image_space

    def collect_operator_ranges(op, source, image):
        if isinstance(op, (LincombOperator, SelectionOperator)):
            for o in op.operators:
                collect_operator_ranges(o, source, image)
        elif isinstance(op, EmpiricalInterpolatedOperator):
            if hasattr(op, 'collateral_basis') and not extends:
                image.append(op.collateral_basis)
        elif isinstance(op, Concatenation):
            firstrange = op.first.range.empty()
            collect_operator_ranges(op.first, source, firstrange)
            collect_operator_ranges(op.second, firstrange, image)
        elif op.linear and not op.parametric:
            image.append(op.apply(source))
        else:
            raise ImageCollectionError(op)

    def collect_functional_ranges(op, image):
        if isinstance(op, (LincombOperator, SelectionOperator)):
            for o in op.operators:
                collect_functional_ranges(o, image)
        elif isinstance(op, AdjointOperator):
            operator = Concatenation(op.range_product, op.operator) if op.range_product else op.operator
            collect_operator_ranges(operator, NumpyVectorArray(np.ones(1)), image)
        elif op.linear and not op.parametric:
            image.append(op.as_vector())
        else:
            raise ImageCollectionError(op)

    if domain is None:
        domain = domain_space.empty()
    image = image_space.empty()
    if not extends:
        for f in functionals:
            collect_functional_ranges(f, image)

    for op in operators:
        collect_operator_ranges(op, domain, image)

    if riesz_representatives and product:
        image = product.apply_inverse(image)

    if orthonormalize:
        gram_schmidt(image, product=product, copy=False)

    return image


def estimate_image_hierarchical(operators=tuple(), functionals=tuple(), domain=None, extends=None,
                                orthonormalize=True, product=None, riesz_representatives=False):
    """Estimate the image of given operators for all mu.

    This is an extended version of :func:`estimate_image`, which calls
    :func:`estimate_image` individually for each vector of `domain`.

    As a result, the vectors in the returned `image` |VectorArray| will
    be ordered by the `domain` vector they correspond to (starting with
    vectors which correspond to the `functionals` and to the |Operators| for
    which the image is estimated independently from `domain`).

    This function also returns an `image_dims` list, such that the first
    `image_dims[i+1]` vectors of `image` correspond to the first `i`
    vectors of `domain` (the first `image_dims[0]` vectors are the vectors
    corresponding to the `functionals` and to the |Operators| with fixed
    image estimate).

    Parameters
    ----------
    operators
        See :func:`estimate_image`.
    functionals
        See :func:`estimate_image`.
    domain
        See :func:`estimate_image`.
    extends
        When additional vectors have been appended to the `domain` |VectorArray|
        after :func:`estimate_image_hierarchical` has been called, and
        :func:`estimate_image_hierarchical` shall be called again for the extended
        `domain` array, `extends` can be set to `(image, image_dims)`, where
        `image`, `image_dims` are the return values of the last
        :func:`estimate_image_hierarchical` call. The old `domain` vectors will
        then be skipped during computation and `image`, `image_dims` will be
        modified in-place.
    orthonormalize
        See :func:`estimate_image`.
    product
        See :func:`estimate_image`.
    riesz_representatives
        See :func:`estimate_image`.

    Returns
    -------
    image
        See above.
    image_dims
        See above.

    Raises
    ------
    ImageCollectionError
        Is raised when for a given |Operator| or |Functional| no image estimate
        is possible.
    """
    assert operators or functionals
    domain_space = operators[0].source if operators else functionals[0].source
    image_space = operators[0].range if operators else functionals[0].source
    assert all(f.range == NumpyVectorSpace(1) and f.source == image_space and f.linear for f in functionals)
    assert all(op.source == domain_space and op.range == image_space for op in operators)
    assert domain is None or domain in domain_space
    assert product is None or product.source == product.range == image_space
    assert extends is None or len(extends) == 2

    logger = getLogger('pymor.algorithms.image.estimate_image_hierarchical')

    if domain is None:
        domain = domain_space.empty()

    if extends:
        image = extends[0]
        image_dims = extends[1]
        ind_range = range(len(image_dims) - 1, len(domain))
    else:
        image = image_space.empty()
        image_dims = []
        ind_range = range(-1, len(domain))

    for i in ind_range:
        logger.info('Estimating image for basis vector {} ...'.format(i))
        if i == -1:
            new_image = estimate_image(operators, functionals, None, extends=False,
                                       orthonormalize=False, product=product,
                                       riesz_representatives=riesz_representatives)
        else:
            new_image = estimate_image(operators, [], domain.copy(i), extends=True,
                                       orthonormalize=False, product=product,
                                       riesz_representatives=riesz_representatives)

        if riesz_representatives and product:
            logger.info('Computing Riesz representatives for basis vector {} ...'.format(i))
            new_image = product.apply_inverse(new_image)

        gram_schmidt_offset = len(image)
        image.append(new_image, remove_from_other=True)
        if orthonormalize:
            logger.info('Orthonormalizing ...')
            gram_schmidt(image, offset=gram_schmidt_offset, product=product, copy=False)
            image_dims.append(len(image))

    return image, image_dims
