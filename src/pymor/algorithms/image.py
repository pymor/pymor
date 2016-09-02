# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.core.exceptions import ImageCollectionError
from pymor.core.logger import getLogger
from pymor.operators.constructions import AdjointOperator, Concatenation, LincombOperator, SelectionOperator
from pymor.operators.ei import EmpiricalInterpolatedOperator
from pymor.vectorarrays.interfaces import VectorArrayInterface
from pymor.vectorarrays.numpy import NumpyVectorSpace, NumpyVectorArray


def estimate_image(operators=(), vectors=(),
                   domain=None, extends=False, orthonormalize=True, product=None,
                   riesz_representatives=False):
    """Estimate the image of given |Operators| for all mu.

    Let `operators` be a list of |Operators| with common source and range, and let
    `vectors` be a list of |VectorArrays| or vector-like |Operators| in the range
    of these operators. Given a |VectorArray| `domain` of vectors in the source of the
    operators, this algorithms determines a |VectorArray| `image` of range vectors
    such that the linear span of `image` contains:

    - `op.apply(U, mu=mu)` for all operators `op` in `operators`, for all possible |Parameters|
      `mu` and for all |VectorArrays| `U` contained in the linear span of `domain`,
    - `U` for all |VectorArrays| in `vectors`,
    - `v.as_vector(mu)` for all |Operators| in `vectors` and all possible |Parameters| `mu`.

    The algorithm will try to choose `image` as small as possible. However, no optimality
    is guaranteed.

    Parameters
    ----------
    operators
        See above.
    vectors
        See above.
    domain
        See above. If `None`, an empty `domain` |VectorArray| is assumed.
    extends
        For some operators, e.g. |EmpiricalInterpolatedOperator|, as well as for all
        elements of `vectors`, `image` is estimated independently from the choice of
        `domain`.  If `extends` is `True`, such operators are ignored. (This is useful
        in case these vectors have already been obtained by earlier calls to this
        function.)
    orthonormalize
        Compute an orthonormal basis for the linear span of `image` using the
        :func:`~pymor.algorithms.gram_schmidt.gram_schmidt` algorithm.
    product
        Inner product |Operator| w.r.t. which to orthonormalize.
    riesz_representatives
        If `True`, compute Riesz representatives of the vectors in `image` before
        orthonormalizing (useful for dual norm computation when the range of the
        `operators` is a dual space).

    Returns
    -------
    The |VectorArray| `image`.

    Raises
    ------
    ImageCollectionError
        Is raised when for a given |Operator| no image estimate is possible.
    """
    assert operators or vectors
    domain_space = operators[0].source if operators else None
    image_space = operators[0].range if operators \
        else vectors[0].space if isinstance(vectors[0], VectorArrayInterface) \
        else vectors[0].range if vectors[0].range != NumpyVectorSpace(1) \
        else vectors[0].source
    assert all(
        isinstance(v, VectorArrayInterface) and v in image_space or
        v.source == NumpyVectorSpace(1) and v.range == image_space and v.linear or
        v.range == NumpyVectorSpace(1) and v.source == image_space and v.linear
        for v in vectors
    )
    assert all(op.source == domain_space and op.range == image_space for op in operators)
    assert domain is None or domain_space is None or domain in domain_space
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

    def collect_vector_ranges(op, image):
        if isinstance(op, (LincombOperator, SelectionOperator)):
            for o in op.operators:
                collect_vector_ranges(o, image)
        elif isinstance(op, AdjointOperator):
            if op.source not in image_space:
                raise ImageCollectionError(op)  # Not implemented
            operator = Concatenation(op.range_product, op.operator) if op.range_product else op.operator
            collect_operator_ranges(operator, NumpyVectorArray(np.ones(1)), image)
        elif op.linear and not op.parametric:
            image.append(op.as_vector())
        else:
            raise ImageCollectionError(op)

    image = image_space.empty()
    if not extends:
        for f in vectors:
            collect_vector_ranges(f, image)

    if operators and domain is None:
        domain = domain_space.empty()
    for op in operators:
        collect_operator_ranges(op, domain, image)

    if riesz_representatives and product:
        image = product.apply_inverse(image)

    if orthonormalize:
        gram_schmidt(image, product=product, copy=False)

    return image


def estimate_image_hierarchical(operators=(), vectors=(), domain=None, extends=None,
                                orthonormalize=True, product=None, riesz_representatives=False):
    """Estimate the image of given |Operators| for all mu.

    This is an extended version of :func:`estimate_image`, which calls
    :func:`estimate_image` individually for each vector of `domain`.

    As a result, the vectors in the returned `image` |VectorArray| will
    be ordered by the `domain` vector they correspond to (starting with
    vectors which correspond to the `functionals` and to |Operators| for
    which the image is estimated independently from `domain`).

    This function also returns an `image_dims` list, such that the first
    `image_dims[i+1]` vectors of `image` correspond to the first `i`
    vectors of `domain` (the first `image_dims[0]` vectors correspond
    to `vectors` and to |Operators| with fixed image estimate).

    Parameters
    ----------
    operators
        See :func:`estimate_image`.
    vectors
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
        Is raised when for a given |Operator| no image estimate is possible.
    """
    assert operators or vectors
    domain_space = operators[0].source if operators else None
    image_space = operators[0].range if operators \
        else vectors[0].space if isinstance(vectors[0], VectorArrayInterface) \
        else vectors[0].range if vectors[0].range != NumpyVectorSpace(1) \
        else vectors[0].source
    assert all(
        isinstance(v, VectorArrayInterface) and v in image_space or
        v.source == NumpyVectorSpace(1) and v.range == image_space and v.linear or
        v.range == NumpyVectorSpace(1) and v.source == image_space and v.linear
        for v in vectors
    )
    assert all(op.source == domain_space and op.range == image_space for op in operators)
    assert domain is None or domain_space is None or domain in domain_space
    assert product is None or product.source == product.range == image_space
    assert extends is None or len(extends) == 2

    logger = getLogger('pymor.algorithms.image.estimate_image_hierarchical')

    if operators and domain is None:
        domain = domain_space.empty()

    if extends:
        image = extends[0]
        image_dims = extends[1]
        ind_range = list(range(len(image_dims) - 1, len(domain))) if operators else list(range(len(image_dims) - 1, 0))
    else:
        image = image_space.empty()
        image_dims = []
        ind_range = list(range(-1, len(domain))) if operators else [-1]

    for i in ind_range:
        logger.info('Estimating image for basis vector {} ...'.format(i))
        if i == -1:
            new_image = estimate_image(operators, vectors, None, extends=False,
                                       orthonormalize=False, product=product,
                                       riesz_representatives=riesz_representatives)
        else:
            new_image = estimate_image(operators, [], domain.copy(i), extends=True,
                                       orthonormalize=False, product=product,
                                       riesz_representatives=riesz_representatives)

        gram_schmidt_offset = len(image)
        image.append(new_image, remove_from_other=True)
        if orthonormalize:
            with logger.block('Orthonormalizing ...'):
                gram_schmidt(image, offset=gram_schmidt_offset, product=product, copy=False)
            image_dims.append(len(image))

    return image, image_dims
