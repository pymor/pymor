# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.algorithms.preassemble import preassemble
from pymor.algorithms.rules import RuleTable, RuleNotMatchingError, match_class, match_always
from pymor.models.interface import Model
from pymor.operators.constructions import (AdjointOperator, AffineOperator, ConcatenationOperator,
                                           FixedParameterOperator, LincombOperator,
                                           SelectionOperator, VectorArrayOperator)
from pymor.operators.list import NumpyListVectorArrayMatrixOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.list import NumpyListVectorSpace


def convert_to_numpy_list_vector_array(obj, space=None):
    """Use NumpyListVectorArrayMatrixOperator instead of NumpyMatrixOperator.

    This simple function recursively converts |NumpyMatrixOperators| to corresponding
    :class:`NumpyListVectorArrayMatrixOperators
    <pymor.operators.list.NumpyListVectorArrayMatrixOperator>`.

    Parameters
    ----------
    obj
        Either an |Operator|, e.g. |NumpyMatrixOperator| or a |LincombOperator| of
        |NumpyMatrixOperators|, or an entire |Model| that is to be converted.
    space
        The |VectorSpace| that is to be converted.

    Returns
    -------
    The converted |Operator| or |Model|.
    """
    if isinstance(obj, Model) and space is None:
        space = obj.solution_space
    assert space is not None
    obj = preassemble(obj)
    return ConvertToNumpyListVectorArrayRules(space).apply(obj)


class ConvertToNumpyListVectorArrayRules(RuleTable):

    def __init__(self, space):
        super().__init__(use_caching=True)
        self.space = space

    @match_class(AdjointOperator, AffineOperator, ConcatenationOperator,
                 FixedParameterOperator, LincombOperator, SelectionOperator,
                 Model)
    def action_recurse(self, op):
        return self.replace_children(op)

    @match_always
    def action_only_range(self, op):
        if not (op.range == self.space and op.source != self.space):
            raise RuleNotMatchingError
        range = NumpyListVectorSpace(op.range.dim, op.range.id)
        return VectorArrayOperator(range.from_numpy(op.as_range_array().to_numpy()), adjoint=False, name=op.name)

    @match_always
    def action_only_source(self, op):
        if not (op.range != self.space and op.source == self.space):
            raise RuleNotMatchingError
        source = NumpyListVectorSpace(op.source.dim, op.source.id)
        return VectorArrayOperator(source.from_numpy(op.as_source_array().to_numpy()), adjoint=True, name=op.name)

    @match_class(NumpyMatrixOperator)
    def action_NumpyMatrixOperator(self, op):
        return op.with_(new_type=NumpyListVectorArrayMatrixOperator)
