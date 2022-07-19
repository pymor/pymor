# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

# flake8: noqa

# define substitutions for all important interface classes
interfaces = '''

.. |BasicObject| replace:: :class:`~pymor.core.base.BasicObject`
.. |BoundaryInfo| replace:: :class:`BoundaryInfo <pymor.discretizers.builtin.grids.interfaces.BoundaryInfo>`
.. |BoundaryInfos| replace:: :class:`BoundaryInfos <pymor.discretizers.builtin.grids.interfaces.BoundaryInfo>`
.. |Models| replace:: :class:`Models <pymor.models.interface.Model>`
.. |Model| replace:: :class:`Model <pymor.models.interface.Model>`
.. |DomainDescriptions| replace:: :class:`DomainDescriptions <pymor.analyticalproblems.domaindescriptions.DomainDescription>`
.. |DomainDescription| replace:: :class:`DomainDescription <pymor.analyticalproblems.domaindescriptions.DomainDescription>`
.. |Functionals| replace:: :class:`Functionals <pymor.operators.interface.Operator>`
.. |Functional| replace:: :class:`Functional <pymor.operators.interface.Operator>`
.. |Functions| replace:: :class:`Functions <pymor.analyticalproblems.functions.Function>`
.. |Function| replace:: :class:`Function <pymor.analyticalproblems.functions.Function>`
.. |Grids| replace:: :class:`Grids <pymor.discretizers.builtin.grids.interfaces.Grid>`
.. |Grid| replace:: :class:`Grid <pymor.discretizers.builtin.grids.interfaces.Grid>`
.. |ImmutableObject| replace:: :class:`~pymor.core.base.ImmutableObject`
.. |immutable| replace:: :class:`immutable <pymor.core.base.ImmutableObject>`
.. |Immutable| replace:: :class:`Immutable <pymor.core.base.ImmutableObject>`
.. |LincombOperators| replace:: :class:`LincombOperators <pymor.operators.constructions.LincombOperator>`
.. |LincombOperator| replace:: :class:`LincombOperator <pymor.operators.constructions.LincombOperator>`
.. |Operators| replace:: :class:`Operators <pymor.operators.interface.Operator>`
.. |Operator| replace:: :class:`Operator <pymor.operators.interface.Operator>`
.. |ReferenceElements| replace:: :class:`ReferenceElements <pymor.discretizers.builtin.grids.interfaces.ReferenceElement>`
.. |ReferenceElement| replace:: :class:`ReferenceElement <pymor.discretizers.builtin.grids.interfaces.ReferenceElement>`
.. |RemoteObject| replace:: :class:`RemoteObject <pymor.parallel.interface.RemoteObject>`
.. |RemoteObjects| replace:: :class:`RemoteObjects <pymor.parallel.interface.RemoteObject>`
.. |VectorArrays| replace:: :class:`VectorArrays <pymor.vectorarrays.interface.VectorArray>`
.. |VectorArray| replace:: :class:`VectorArray <pymor.vectorarrays.interface.VectorArray>`
.. |VectorSpace| replace:: :class:`VectorSpace <pymor.vectorarrays.interface.VectorSpace>`
.. |VectorSpaces| replace:: :class:`VectorSpaces <pymor.vectorarrays.interface.VectorSpace>`
.. |BlockVectorSpace| replace:: :class:`BlockVectorSpace <pymor.vectorarrays.block.BlockVectorSpace>`
.. |WorkerPool| replace:: :class:`WorkerPool <pymor.parallel.interface.WorkerPool>`
.. |WorkerPools| replace:: :class:`WorkerPools <pymor.parallel.interface.WorkerPool>`

'''

# substitutions for the most important classes and methods in pyMOR
common = '''
.. |analytical problem| replace:: :mod:`analytical problem <pymor.analyticalproblems>`
.. |analytical problems| replace:: :mod:`analytical problems <pymor.analyticalproblems>`

.. |default| replace:: :mod:`default <pymor.core.defaults>`
.. |defaults| replace:: :mod:`~pymor.core.defaults`

.. |CacheRegion| replace:: :class:`~pymor.core.cache.CacheRegion`
.. |CacheableObject| replace:: :class:`~pymor.core.cache.CacheableObject`

.. |StationaryProblem| replace:: :class:`~pymor.analyticalproblems.elliptic.StationaryProblem`
.. |StationaryProblems| replace:: :class:`StationaryProblems <pymor.analyticalproblems.elliptic.StationaryProblem>`
.. |InstationaryProblem| replace:: :class:`~pymor.analyticalproblems.instationary.InstationaryProblem`

.. |RectDomain| replace:: :class:`~pymor.analyticalproblems.domaindescriptions.RectDomain`
.. |PolygonalDomain| replace:: :class:`~pymor.analyticalproblems.domaindescriptions.PolygonalDomain`
.. |CylindricalDomain| replace:: :class:`~pymor.analyticalproblems.domaindescriptions.CylindricalDomain`
.. |TorusDomain| replace:: :class:`~pymor.analyticalproblems.domaindescriptions.TorusDomain`
.. |LineDomain| replace:: :class:`~pymor.analyticalproblems.domaindescriptions.LineDomain`
.. |CircleDomain| replace:: :class:`~pymor.analyticalproblems.domaindescriptions.CircleDomain`
.. |discretize_domain_default| replace:: :func:`~pymor.discretizers.builtin.domaindiscretizers.default.discretize_domain_default`

.. |BitmapFunction| replace:: :class:`~pymor.analyticalproblems.functions.BitmapFunction`
.. |BitmapFunctions| replace:: :class:`BitmapFunctions <pymor.analyticalproblems.functions.BitmapFunction>`
.. |ConstantFunction| replace:: :class:`~pymor.analyticalproblems.functions.ConstantFunction`
.. |ConstantFunctions| replace:: :class:`ConstantFunctions <pymor.analyticalproblems.functions.ConstantFunction>`
.. |ExpressionFunction| replace:: :class:`~pymor.analyticalproblems.functions.ExpressionFunction`
.. |ExpressionFunctions| replace:: :class:`ExpressionFunctions <pymor.analyticalproblems.functions.ExpressionFunction>`
.. |LincombFunction| replace:: :class:`~pymor.analyticalproblems.functions.LincombFunction`
.. |LincombFunctions| replace:: :class:`LincombFunctions <pymor.analyticalproblems.functions.LincombFunction>`

.. |ExpressionParameterFunctional| replace:: :class:`~pymor.parameters.functionals.ExpressionParameterFunctional`
.. |ExpressionParameterFunctionals| replace:: :class:`ExpressionParameterFunctionals <pymor.parameters.functionals.ExpressionParameterFunctional>`

.. |OnedGrid| replace:: :class:`~pymor.discretizers.builtin.grids.oned.OnedGrid`
.. |RectGrid| replace:: :class:`~pymor.discretizers.builtin.grids.rect.RectGrid`
.. |TriaGrid| replace:: :class:`~pymor.discretizers.builtin.grids.tria.TriaGrid`
.. |GmshGrid| replace:: :class:`~pymor.discretizers.builtin.grids.gmsh.GmshGrid`

.. |NumpyVectorArray| replace:: :class:`~pymor.vectorarrays.numpy.NumpyVectorArray`
.. |NumpyVectorArrays| replace:: :class:`NumpyVectorArrays <pymor.vectorarrays.numpy.NumpyVectorArray>`
.. |ListVectorArray| replace:: :class:`~pymor.vectorarrays.list.ListVectorArray`
.. |ListVectorArrays| replace:: :class:`ListVectorArrays <pymor.vectorarrays.list.ListVectorArray>`

.. |NumpyHankelOperator| replace:: :class:`~pymor.operators.numpy.NumpyHankelOperator`
.. |NumpyHankelOperators| replace:: :class:`NumpyHankelOperators <pymor.operators.numpy.NumpyHankelOperator>`
.. |NumpyMatrixOperator| replace:: :class:`~pymor.operators.numpy.NumpyMatrixOperator`
.. |NumpyMatrixOperators| replace:: :class:`NumpyMatrixOperators <pymor.operators.numpy.NumpyMatrixOperator>`
.. |NumpyMatrixBasedOperator| replace:: :class:`~pymor.operators.numpy.NumpyMatrixBasedOperator`
.. |NumpyMatrixBasedOperators| replace:: :class:`NumpyMatrixBasedOperators <pymor.operators.numpy.NumpyMatrixBasedOperator>`
.. |NumpyGenericOperator| replace:: :class:`~pymor.operators.numpy.NumpyGenericOperator`
.. |EmpiricalInterpolatedOperator| replace:: :class:`~pymor.operators.ei.EmpiricalInterpolatedOperator`
.. |EmpiricalInterpolatedOperators| replace:: :class:`EmpiricalInterpolatedOperators <pymor.operators.ei.EmpiricalInterpolatedOperator>`
.. |ConcatenationOperator| replace:: :class:`~pymor.operators.constructions.ConcatenationOperator`
.. |ConcatenationOperators| replace:: :class:`ConcatenationOperators <pymor.operators.constructions.ConcatenationOperator>`
.. |VectorOperator| replace:: :class:`~pymor.operators.constructions.VectorOperator`
.. |VectorFunctional| replace:: :class:`~pymor.operators.constructions.VectorFunctional`
.. |NumpyVectorSpace| replace:: :func:`~pymor.vectorarrays.numpy.NumpyVectorSpace`
.. |NumpyVectorSpaces| replace:: :func:`NumpyVectorSpaces <pymor.vectorarrays.numpy.NumpyVectorSpace>`

.. |StationaryModel| replace:: :class:`~pymor.models.basic.StationaryModel`
.. |StationaryModels| replace:: :class:`StationaryModels <pymor.models.basic.StationaryModel>`
.. |InstationaryModel| replace:: :class:`~pymor.models.basic.InstationaryModel`

.. |LTIModel| replace:: :class:`~pymor.models.iosys.LTIModel`
.. |LTIModels| replace:: :class:`LTIModels <pymor.models.iosys.LTIModel>`
.. |PHLTIModel| replace:: :class:`~pymor.models.iosys.PHLTIModel`
.. |PHLTIModels| replace:: :class:`PHLTIModels <pymor.models.iosys.PHLTIModel>`
.. |TransferFunction| replace:: :class:`~pymor.models.transfer_function.TransferFunction`
.. |TransferFunctions| replace:: :class:`TransferFunctions <pymor.models.transfer_function.TransferFunction>`
.. |SecondOrderModel| replace:: :class:`~pymor.models.iosys.SecondOrderModel`
.. |SecondOrderModels| replace:: :class:`SecondOrderModels <pymor.models.iosys.SecondOrderModel>`
.. |LinearDelayModel| replace:: :class:`~pymor.models.iosys.LinearDelayModel`
.. |LinearDelayModels| replace:: :class:`LinearDelayModels <pymor.models.iosys.LinearDelayModel>`
.. |NeuralNetworkModel| replace:: :class:`~pymor.models.neural_network.NeuralNetworkModel`
.. |QuadraticHamiltonianModel| replace:: :class:`~pymor.model.symplectic.QuadraticHamiltonianModel`

.. |MoebiusTransformation| replace:: :class:`~pymor.models.transforms.MoebiusTransformation`
.. |MoebiusTransformations| replace:: :class:`MoebiusTransformations <pymor.models.transforms.MoebiusTransformation>`
.. |BilinearTransformation| replace:: :class:`~pymor.models.transforms.BilinearTransformation`
.. |CayleyTransformation| replace:: :class:`~pymor.models.transforms.CayleyTransformation`

.. |Parameter| replace:: :class:`Parameter <pymor.parameters.base.Parameters>`
.. |Parameters| replace:: :class:`~pymor.parameters.base.Parameters`
.. |Parameter values| replace:: :class:`Parameter values <pymor.parameters.base.Mu>`
.. |parameter values| replace:: :class:`parameter values <pymor.parameters.base.Mu>`
.. |Parameter value| replace:: :class:`Parameter value <pymor.parameters.base.Mu>`
.. |parameter value| replace:: :class:`parameter value <pymor.parameters.base.Mu>`
.. |ParametricObject| replace:: :class:`~pymor.parameters.base.ParametricObject`
.. |ParametricObjects| replace:: :class:`ParametricObjects <pymor.parameters.base.ParametricObject>`
.. |parametric| replace:: :attr:`~pymor.parameters.base.ParametricObject.parametric`
.. |ParameterFunctional| replace:: :class:`~pymor.parameters.functionals.ParameterFunctional`
.. |ParameterFunctionals| replace:: :class:`ParameterFunctionals <pymor.parameters.functionals.ParameterFunctional>`
.. |ParameterSpace| replace:: :class:`ParameterSpace <pymor.parameters.base.ParameterSpace>`
.. |ParameterSpaces| replace:: :class:`ParameterSpaces <pymor.parameters.base.ParameterSpace>`

.. |NumPy| replace:: :mod:`NumPy <numpy>`
.. |NumPy array| replace:: :class:`NumPy array <numpy.ndarray>`
.. |NumPy arrays| replace:: :class:`NumPy arrays <numpy.ndarray>`
.. |Numpy array| replace:: :class:`NumPy array <numpy.ndarray>`
.. |Numpy arrays| replace:: :class:`NumPy arrays <numpy.ndarray>`
.. |array| replace:: :class:`NumPy array <numpy.ndarray>`
.. |Array| replace:: :class:`NumPy array <numpy.ndarray>`

.. |SciPy| replace:: :mod:`SciPy <scipy>`
.. |SciPy spmatrix| replace:: :class:`SciPy spmatrix <scipy.sparse.spmatrix>`
.. |SciPy spmatrices| replace:: :class:`SciPy spmatrices <scipy.sparse.spmatrix>`
.. |Scipy spmatrix| replace:: :class:`SciPy spmatrix <scipy.sparse.spmatrix>`
.. |Scipy spmatrices| replace:: :class:`SciPy spmatrices <scipy.sparse.spmatrix>`

.. |OrderedDict| replace:: :class:`~collections.OrderedDict`

.. |solver_options| replace:: :attr:`~pymor.operators.interface.Operator.solver_options`

.. |RuleTable| replace:: :class:`~pymor.algorithms.rules.RuleTable`
.. |RuleTables| replace:: :class:`RuleTables <pymor.algorithms.rules.RuleTable>`
.. |rule| replace:: :class:`~pymor.algorithms.rules.rule`
.. |rules| replace:: :class:`rules <pymor.algorithms.rules.rule>`
.. |project| replace:: :func:`~pymor.algorithms.projection.project`

.. |SymplecticBasis| replace:: :class:`~pymor.algorithms.symplectic.SymplecticBasis`
.. |CanonicalSymplecticFormOperator| replace:: :class:`~pymor.operators.symplectic.CanonicalSymplecticFormOperator`
'''

substitutions = interfaces + common

# list of (key, jinja_safe_key, substitution_value)
jinja_subst = []
for line in substitutions.split('\n'):
    if line == "":
        continue
    key, subst = line.split(' replace:: ')
    key = key.strip()
    key = key.replace('.. |', '').replace('|', '')
    subst = subst.replace(':', '{', 1).replace(':', '}', 2)
    jinja_subst.append((key, key.replace(' ', '_'), subst))

inline_directives = ['math', 'meth', 'class', 'ref', 'mod', 'attr', 'doc', ]

if __name__ == '__main__':
    with open('rst_to_myst.sed', 'wt') as out:
        for dr in inline_directives:
            out.write(f's;:{dr}:;{{{dr}}};g\n')
        l = '{{\\ '
        r = '\\ }}'
        for key, safe_key, _ in jinja_subst:
            out.write(f's;|{key}|;{l}{safe_key}{r};g\n')

myst_substitutions = {safe_key: subst for _, safe_key, subst in jinja_subst}
