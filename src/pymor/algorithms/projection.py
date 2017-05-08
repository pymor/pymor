# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.rules import RuleTable, match_class, match_generic
from pymor.core.exceptions import RuleNotMatchingError
from pymor.operators.basic import ProjectedOperator
from pymor.operators.constructions import (LincombOperator, Concatenation, ConstantOperator,
                                           ZeroOperator, AffineOperator, AdjointOperator, SelectionOperator,
                                           VectorArrayOperator)
from pymor.operators.ei import EmpiricalInterpolatedOperator, ProjectedEmpiciralInterpolatedOperator
from pymor.operators.interfaces import OperatorInterface
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace


def project(op, range_basis, source_basis, product=None):
    """Petrov-Galerkin projection of a given |Operator|.

    Given an inner product `( ⋅, ⋅)`, source vectors `b_1, ..., b_N`
    and range vectors `c_1, ..., c_M`, the projection `op_proj` of `op`
    is defined by ::

        [ op_proj(e_j) ]_i = ( c_i, op(b_j) )

    for all i,j, where `e_j` denotes the j-th canonical basis vector of R^N.

    In particular, if the `c_i` are orthonormal w.r.t. the given product,
    then `op_proj` is the coordinate representation w.r.t. the `b_i/c_i` bases
    of the restriction of `op` to `span(b_i)` concatenated with the
    orthogonal projection onto `span(c_i)`.

    From another point of view, if `op` is viewed as a bilinear form
    (see :meth:`apply2`) and `( ⋅, ⋅ )` is the Euclidean inner
    product, then `op_proj` represents the matrix of the bilinear form restricted
    to `span(b_i) / span(c_i)` (w.r.t. the `b_i/c_i` bases).

    How the projection is realized will depend on the given |Operator|.
    While a projected |NumpyMatrixOperator| will
    again be a |NumpyMatrixOperator|, only a generic
    :class:`~pymor.operators.basic.ProjectedOperator` can be returned
    in general. The exact algorithm is specified in :class:`ProjectRules`.

    Parameters
    ----------
    range_basis
        The vectors `c_1, ..., c_M` as a |VectorArray|. If `None`, no
        projection in the range space is performed.
    source_basis
        The vectors `b_1, ..., b_N` as a |VectorArray| or `None`. If `None`,
        no restriction of the source space is performed.
    product
        An |Operator| representing the inner product.  If `None`, the
        Euclidean inner product is chosen.

    Returns
    -------
    The projected |Operator| `op_proj`.
    """
    assert source_basis is None or source_basis in op.source
    assert range_basis is None or range_basis in op.range
    assert product is None or product.source == product.range == op.range

    return ProjectRules.apply(op, range_basis, source_basis, product=product)


class ProjectRules(RuleTable):
    """|RuleTable| for the :func:`project` algorithm."""

    @match_class(ZeroOperator)
    def action_ZeroOperator(self, op, range_basis, source_basis, product=None):
        if source_basis is not None and range_basis is not None:
            from pymor.operators.numpy import NumpyMatrixOperator
            return NumpyMatrixOperator(np.zeros((len(range_basis), len(source_basis))),
                                       source_id=op.source.id, range_id=op.range.id,
                                       name=op.name)
        else:
            new_source = (NumpyVectorSpace(len(source_basis), op.source.id) if source_basis is not None else
                          op.source)
            new_range = (NumpyVectorSpace(len(range_basis), op.range.id) if range_basis is not None else
                         op.range)
            return ZeroOperator(new_source, new_range, name=op.name)

    @match_class(ConstantOperator)
    def action_ConstantOperator(self, op, range_basis, source_basis, product=None):
        if range_basis is not None:
            if product:
                projected_value = NumpyVectorSpace.make_array(product.apply2(range_basis, op._value).T, op.range.id)
            else:
                projected_value = NumpyVectorSpace.make_array(range_basis.dot(op._value).T, op.range.id)
        else:
            projected_value = op._value
        if source_basis is None:
            return ConstantOperator(projected_value, op.source, name=op.name)
        else:
            return ConstantOperator(projected_value, NumpyVectorSpace(len(source_basis), op.source.id),
                                    name=op.name)

    @match_generic(lambda op: op.linear and not op.parametric, 'linear and not parametric')
    def action_apply_basis(self, op, range_basis, source_basis, product=None):
        if source_basis is None:
            if range_basis is None:
                return op
            else:
                try:
                    V = op.apply_transpose(product.apply(range_basis) if product else range_basis)
                except NotImplementedError:
                    raise RuleNotMatchingError('apply_transpose not implemented')
                if isinstance(op.source, NumpyVectorSpace):
                    from pymor.operators.numpy import NumpyMatrixOperator
                    return NumpyMatrixOperator(V.data,
                                               source_id=op.source.id,
                                               range_id=op.range.id,
                                               name=op.name)
                else:
                    from pymor.operators.constructions import VectorArrayOperator
                    return VectorArrayOperator(V, transposed=True, space_id=op.range.id, name=op.name)
        else:
            if range_basis is None:
                V = op.apply(source_basis)
                if isinstance(op.range, NumpyVectorSpace):
                    from pymor.operators.numpy import NumpyMatrixOperator
                    return NumpyMatrixOperator(V.data.T,
                                               source_id=op.source.id,
                                               range_id=op.range.id,
                                               name=op.name)
                else:
                    from pymor.operators.constructions import VectorArrayOperator
                    return VectorArrayOperator(V, transposed=False, space_id=op.source.id, name=op.name)
            elif product is None:
                from pymor.operators.numpy import NumpyMatrixOperator
                return NumpyMatrixOperator(op.apply2(range_basis, source_basis),
                                           source_id=op.source.id,
                                           range_id=op.range.id,
                                           name=op.name)
            else:
                from pymor.operators.numpy import NumpyMatrixOperator
                V = op.apply(source_basis)
                return NumpyMatrixOperator(product.apply2(range_basis, V),
                                           source_id=op.source.id,
                                           range_id=op.range.id,
                                           name=op.name)

    @match_class(Concatenation)
    def action_Concatenation(self, op, range_basis, source_basis, product=None):
        """project linear non-parametric Concatenations"""
        if op.parametric or not op.linear:
            raise RuleNotMatchingError('Only implemented for non-parametric linear Concatenations')
        projected_first = self.apply(op.first, None, source_basis, product=None)
        if isinstance(projected_first, VectorArrayOperator) and not projected_first.transposed:
            return self.apply(op.second, range_basis, projected_first._array, product=product)
        else:
            projected_second = self.apply(op.second, range_basis, None, product=product)
            return Concatenation(projected_second, projected_first, name=op.name)

    @match_class(AdjointOperator)
    def action_AdjointOperator(self, op, range_basis, source_basis, product=None):
        if range_basis is not None:
            if product is not None:
                range_basis = product.apply(range_basis)
            if op.source_product:
                range_basis = op.source_product.apply_inverse(range_basis)

        if source_basis is not None and op.range_product:
            source_basis = op.range_product.apply(source_basis)

        operator = self.apply(op.operator, source_basis, range_basis)
        range_product = op.range_product if source_basis is None else None
        source_product = op.source_product if range_basis is None else None
        return AdjointOperator(operator, source_product=source_product, range_product=range_product,
                               name=op.name)

    @match_class(EmpiricalInterpolatedOperator)
    def action_EmpiricalInterpolatedOperator(self, op, range_basis, source_basis, product=None):
        if len(op.interpolation_dofs) == 0:
            return self.apply(ZeroOperator(op.source, op.range, op.name), range_basis, source_basis, product)
        elif not hasattr(op, 'restricted_operator') or source_basis is None:
            raise RuleNotMatchingError('Has no restricted operator or source_basis is None')
        else:
            if range_basis is not None:
                if product is None:
                    projected_collateral_basis = NumpyVectorSpace.make_array(op.collateral_basis.dot(range_basis),
                                                                             op.range.id)
                else:
                    projected_collateral_basis = NumpyVectorSpace.make_array(product.apply2(op.collateral_basis,
                                                                                            range_basis),
                                                                             op.range.id)
            else:
                projected_collateral_basis = op.collateral_basis

            return ProjectedEmpiciralInterpolatedOperator(op.restricted_operator, op.interpolation_matrix,
                                                          NumpyVectorSpace.make_array(source_basis.components(op.source_dofs)),
                                                          projected_collateral_basis, op.triangular,
                                                          op.source.id, None, op.name)

    @match_class(AffineOperator)
    def action_AffineOperator(self, op, range_basis, source_basis, product=None):
        return self.apply(op.affine_shift + op.linear_part, range_basis, source_basis, product)

    @match_class(LincombOperator, SelectionOperator)
    def action_recurse(self, op, range_basis, source_basis, product=None):
        return self.replace_children(op, range_basis, source_basis, product)

    @match_class(OperatorInterface)
    def action_generic_projection(self, op, range_basis, source_basis, product=None):
        op.logger.warn('Using inefficient generic projection operator')
        return ProjectedOperator(op, range_basis, source_basis, product)


def project_to_subbasis(op, dim_range=None, dim_source=None):
    """Project given already projected |Operator| to a subbasis.

    The purpose of this method is to further project an operator that has been
    obtained through :meth:`project` to subbases of the original projection bases, i.e. ::

        project_to_subbasis(project(op, r_basis, s_basis, prod), dim_range, dim_source)

    should be the same as ::

        project(op, r_basis[:dim_range], s_basis[:dim_source], prod)

    For a |NumpyMatrixOperator| this amounts to extracting the upper-left
    (dim_range, dim_source) corner of its matrix.

    The subbasis projection algorithm is specified in :class:`ProjectToSubbasisRules`.

    Parameters
    ----------
    dim_range
        Dimension of the range subbasis.
    dim_source
        Dimension of the source subbasis.

    Returns
    -------
    The projected |Operator|.
    """
    assert dim_source is None or (isinstance(op.source, NumpyVectorSpace) and dim_source <= op.source.dim)
    assert dim_range is None or (isinstance(op.range, NumpyVectorSpace) and dim_range <= op.range.dim)

    return ProjectToSubbasisRules.apply(op, dim_range, dim_source)


class ProjectToSubbasisRules(RuleTable):
    """|RuleTable| for the :func:`project_to_subbasis` algorithm."""

    @match_class(LincombOperator)
    def action_recurse(self, op, dim_range=None, dim_source=None):
        return self.replace_children(op, dim_range, dim_source)

    @match_class(NumpyMatrixOperator)
    def action_NumpyMatrixOperator(self, op, dim_range=None, dim_source=None):
        # copy instead of just slicing the matrix to ensure contiguous memory
        return NumpyMatrixOperator(op._matrix[:dim_range, :dim_source].copy(),
                                   source_id=op.source.id,
                                   range_id=op.range.id,
                                   solver_options=op.solver_options,
                                   name=op.name)

    @match_class(ConstantOperator)
    def action_ConstantOperator(self, op, dim_range=None, dim_source=None):
        source = op.source if dim_source is None else NumpyVectorSpace(dim_source, op.source.id)
        value = op._value if dim_range is None else NumpyVectorSpace(op._value.data[:, :dim_range], op.range.id)
        return ConstantOperator(value, source, name=op.name)

    @match_class(ProjectedEmpiciralInterpolatedOperator)
    def action_ProjectedEmpiciralInterpolatedOperator(self, op, dim_range=None, dim_source=None):
        if not isinstance(op.projected_collateral_basis.space, NumpyVectorSpace):
            raise NotImplementedError

        restricted_operator = op.restricted_operator

        old_pcb = op.projected_collateral_basis
        projected_collateral_basis = NumpyVectorSpace.make_array(old_pcb.data[:, :dim_range],
                                                                 old_pcb.space.id)

        old_sbd = op.source_basis_dofs
        source_basis_dofs = NumpyVectorSpace.make_array(old_sbd.data[:dim_source])

        return ProjectedEmpiciralInterpolatedOperator(restricted_operator, op.interpolation_matrix,
                                                      source_basis_dofs, projected_collateral_basis, op.triangular,
                                                      op.source.id, solver_options=op.solver_options, name=op.name)

    @match_class(ProjectedOperator)
    def action_ProjectedOperator(self, op, dim_range=None, dim_source=None):
        source_basis = op.source_basis if dim_source is None \
            else op.source_basis[:dim_source]
        range_basis = op.range_basis if dim_range is None \
            else op.range_basis[:dim_range]
        return ProjectedOperator(op.operator, range_basis, source_basis, product=None,
                                 solver_options=op.solver_options)
