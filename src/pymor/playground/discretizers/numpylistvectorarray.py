# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.operators.constructions import LincombOperator
from pymor.operators.numpy import NumpyMatrixOperator

from pymor.playground.operators.numpy import NumpyListVectorArrayMatrixOperator


def convert_to_numpy_list_vector_array(d):
    """Use NumpyListVectorArrayMatrixOperator instead of NumpyMatrixOperator.

    This simple function converts linear, affinely decomposed discretizations
    to use :class:`~pymor.playground.operators.numpy.NumpyListVectorArrayMatrixOperator`
    instead of |NumpyMatrixOperator|.
    """

    def convert_operator(op, functional=False, vector=False):
        if isinstance(op, LincombOperator):
            return op.with_(operators=[convert_operator(o) for o in op.operators])
        elif not op.parametric:
            op = op.assemble()
            if isinstance(op, NumpyMatrixOperator):
                return NumpyListVectorArrayMatrixOperator(op.matrix,
                                                          source_id=op.source.id, range_id=op.range.id,
                                                          name=op.name)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    operators = {k: convert_operator(v) for k, v in d.operators.items()}
    products = {k: convert_operator(v) for k, v in d.products.items()}

    return d.with_(operators=operators, products=products)
