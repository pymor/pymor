# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.algorithms import trivial_basis_extension, gram_schmidt_basis_extension
from pymor.core import getLogger
from pymor.core.exceptions import ExtensionError
from pymor.playground.la import BlockVectorArray
from pymor.playground.operators import BlockOperator


def trivial_block_basis_extension(basis, U, copy_basis=True, copy_U=True, require_all=False):
    """Block variant of |trivial_basis_extension|
    """

    logger = getLogger('pymor.algorithms.blockbasisextension.trivial_block_basis_extension')

    assert isinstance(U, BlockVectorArray)
    if not copy_U:
        logger.warn('The option copy_U==False is not supported for BlockVectorArrays!')
    num_blocks = U.num_blocks
    if basis is None:
        basis = tuple(None for ii in np.arange(num_blocks))
    assert isinstance(basis, list)
    assert len(basis) == num_blocks

    failure = [True for ii in np.arange(num_blocks)]
    new_basis = [None for ii in np.arange(num_blocks)]
    hierarchic = [False for ii in np.arange(num_blocks)]

    for ii in np.arange(num_blocks):
        try:
            nb, ed = trivial_basis_extension(basis[ii],
                                             U._blocks[ii],
                                             copy_basis=copy_basis,
                                             copy_U=True)
            failure[ii] = False
            new_basis[ii] = nb
            assert ed.keys() == ['hierarchic']
            hierarchic[ii] = ed['hierarchic']
        except ExtensionError:
            new_basis[ii] = basis[ii]
            hierarchic[ii] = True

    num_failures = sum(1 if ff else 0 for ff in failure)
    if require_all and any(failure):
        raise ExtensionError
    elif not require_all and all(failure):
        raise ExtensionError
    elif num_failures > 0:
        logger.warn('Extension failed for {} out of {} block{}!'.format(num_failures,
                                                                        num_blocks,
                                                                        's' if num_blocks > 1 else ''))

    return tuple(new_basis), {'hierarchic': all(hierarchic)}


def gram_schmidt_block_basis_extension(basis, U, product=None, copy_basis=True, copy_U=True, require_all=False):
    """Block variant of |gram_schmidt_basis_extension|.
    """

    logger = getLogger('pymor.algorithms.blockbasisextension.gram_schmidt_block_basis_extension')

    if isinstance(U, BlockVectorArray):
        blocks = U._blocks
    elif isinstance(U, list):
        blocks = tuple(U)
    elif isinstance(U, tuple):
        blocks = U
    else:
        raise ExtensionError('U of unknown type given: {}'.format(type(U)))
    if isinstance(U, BlockVectorArray) and not copy_U:
        logger.warn('The option copy_U==False is not supported for BlockVectorArrays!')
    num_blocks = len(blocks)

    if basis is None:
        basis = tuple(None for ii in np.arange(num_blocks))
    elif isinstance(basis, list):
        basis = tuple(basis)
    assert isinstance(basis, tuple)
    assert len(basis) == num_blocks
    if product is None:
        product = [None for ii in np.arange(num_blocks)]
    elif isinstance(product, BlockOperator):
        assert product.is_diagonal and product.num_source_blocks == num_blocks
        product = [product._blocks[ii][ii] for ii in num_blocks]
    elif not isinstance(product, list):
        product = [product for ii in np.arange(num_blocks)]
    assert isinstance(product, list)
    assert len(product) == num_blocks

    failure = [True for ii in np.arange(num_blocks)]
    new_basis = [None for ii in np.arange(num_blocks)]
    hierarchic = [False for ii in np.arange(num_blocks)]

    for ii in np.arange(num_blocks):
        try:
            nb, ed = gram_schmidt_basis_extension(basis[ii],
                                                  blocks[ii],
                                                  product=product[ii],
                                                  copy_basis=copy_basis,
                                                  copy_U=True)
            failure[ii] = False
            new_basis[ii] = nb
            assert ed.keys() == ['hierarchic']
            hierarchic[ii] = ed['hierarchic']
        except ExtensionError:
            new_basis[ii] = basis[ii]
            hierarchic[ii] = True

    num_failures = sum(1 if ff else 0 for ff in failure)
    if require_all and any(failure):
        raise ExtensionError
    elif not require_all and all(failure):
        raise ExtensionError
    elif num_failures > 0:
        logger.warn('Extension failed for {} out of {} block{}!'.format(num_failures,
                                                                        num_blocks,
                                                                        's' if num_blocks > 1 else ''))

    return tuple(new_basis), {'hierarchic': all(hierarchic)}

