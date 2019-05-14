# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from hypothesis import assume
from hypothesis._strategies import just, one_of, sampled_from
from hypothesis.stateful import rule, RuleBasedStateMachine, Bundle

from pymor.algorithms import gram_schmidt
from pymor.algorithms.projection import project
from pymor.core.exceptions import ExtensionError
from pymor.operators.constructions import VectorArrayOperator
from pymor.reductors.basic import extend_basis
from pymor.tools.floatcmp import float_cmp
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymortests.strategies import hy_dims, hy_float_array_elements, numpy_vector_array, vector_arrays


class BasisMachine(RuleBasedStateMachine):
    """
    rules are meant to generate a new basis if they have `target` argument
    """
    Bases = Bundle('bases')

    def __init__(self):
        super(BasisMachine, self).__init__()

    @rule(target=Bases, dim=hy_dims(count=1, compatible=True), scale=hy_float_array_elements)
    def generate(self, dim, scale):
        assume(not np.allclose(scale, 0))
        dim = dim[0]
        space = NumpyVectorSpace(dim)
        arr = np.eye(dim, dim)
        arr *= scale
        return space.from_data(arr)

    @rule(basis=Bases, target=Bases)
    def id_project(self, basis):
        op = VectorArrayOperator(basis)
        proj_op = project(source_basis=basis, range_basis=basis, op=op)
        return proj_op.as_source_array()

    @rule(basis=Bases, target=Bases)
    def gram_schmidt(self, basis):
        return gram_schmidt.gram_schmidt(basis)

    @rule(basis=Bases, vector_tuple=numpy_vector_array(count=1, length=just([1]), dtype=np.float_))
    def coefficients(self, basis, vector_tuple):
        assume(basis.dim == vector_tuple[0].dim)
        assume(basis.dim > 0)
        x = vector_tuple[0].data[0, :]
        A = basis.data
        coeff = np.linalg.solve(A, x)
        assert len(basis) == basis.space.dim == len(coeff)

    @rule(basis=Bases, vector_tuple=numpy_vector_array(count=1, length=just([1]), dtype=np.float_),
          method=sampled_from(('trivial', 'gram_schmidt', 'pod')))
    def extend(self, basis, vector_tuple, method):
        vector = vector_tuple[0]
        assume(basis.dim == vector.dim)
        try:
            ret = extend_basis(vector, basis, method=method)
        except ExtensionError:
            ret = vector
        return ret


TestBasis = BasisMachine.TestCase


class ScalarMachine(RuleBasedStateMachine):
    Vectors = Bundle('vectors')

    def __init__(self):
        super(ScalarMachine, self).__init__()

    @rule(target=Vectors, vect=vector_arrays(count=1, length=just([1])).filter(lambda x: x[0].dim == 1))
    def generate(self, vect):
        vec = vect[0]
        assert vec.dim == 1
        assert len(vec) == 1
        return vec

    @rule(target=Vectors, vector=Vectors, scale=hy_float_array_elements)
    def scal(self, vector, scale):
        ref = vector[0].data[0] * scale
        vector.scal(scale)
        assert float_cmp(ref, vector[0].data[0])
        return vector

    @rule(target=Vectors, vector=Vectors, alpha=hy_float_array_elements, x=Vectors)
    def axpy(self, vector, alpha, x):
        assume(type(x) == type(vector))
        ref = alpha * x[0].data[0] + vector[0].data[0]
        vector.axpy(alpha, x)
        assert float_cmp(ref, vector[0].data[0])
        return vector

    @rule(target=Vectors, a=Vectors, b=Vectors)
    def plus(self, a, b):
        assume(type(a) == type(b))
        ref = a[0].data[0] + b[0].data[0]
        c = a + b
        assert float_cmp(ref, c[0].data[0])
        return c


TestScalar = ScalarMachine.TestCase