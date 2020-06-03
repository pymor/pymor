# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.algorithms.preassemble import preassemble
from pymor.algorithms.rules import RuleTable, match_class
from pymor.models.interface import Model
from pymor.operators.constructions import (AdjointOperator, AffineOperator, Concatenation,
                                           FixedParameterOperator, LincombOperator,
                                           SelectionOperator, VectorArrayOperator)
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.playground.operators.numpy import NumpyListVectorArrayMatrixOperator
from pymor.vectorarrays.list import NumpyListVectorSpace


def convert_to_numpy_list_vector_array(obj):
    """Use NumpyListVectorArrayMatrixOperator instead of NumpyMatrixOperator.

    This simple function recursively converts |NumpyMatrixOperators| to corresponding
    :class:`NumpyListVectorArrayMatrixOperators <pymor.playground.operators.numpy.NumpyListVectorArrayMatrixOperator>`.

    Parameters
    ----------
    obj
        Either an |Operator|, e.g. |NumpyMatrixOperator| or a |LincombOperator| of
        |NumpyMatrixOperators|, or an entire |Model| that is to be converted.

    Returns
    -------
    The converted |Operator| or |Model|.
    """
    obj = preassemble(obj)
    return ConvertToNumpyListVectorArrayRules().apply(obj)


class ConvertToNumpyListVectorArrayRules(RuleTable):

    def __init__(self):
        super().__init__(use_caching=True)

    @match_class(AdjointOperator, AffineOperator, Concatenation,
                 FixedParameterOperator, LincombOperator, SelectionOperator,
                 Model)
    def action_recurse(self, op):
        return self.replace_children(op)

    @match_class(NumpyMatrixOperator)
    def action_NumpyMatrixOperator(self, op):
        return op.with_(new_type=NumpyListVectorArrayMatrixOperator)

    @match_class(VectorArrayOperator)
    def action_VectorArrayOperator(self, op):
        space = NumpyListVectorSpace(op.array.dim, op.array.space.id)
        return op.with_(new_type=VectorArrayOperator,
                        array=space.from_numpy(op.array.to_numpy()))
