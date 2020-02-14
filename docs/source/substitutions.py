# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

# flake8: noqa

# define substitutions for all important interface classes
interfaces = '''

.. |AffineGrids| replace:: :class:`AffineGrids <pymor.discretizers.builtin.grids.interfaces.AffineGrid>`
.. |AffineGrid| replace:: :class:`AffineGrid <pymor.discretizers.builtin.grids.interfaces.AffineGrid>`
.. |BasicObject| replace:: :class:`~pymor.core.base.BasicObject`
.. |BoundaryInfo| replace:: :class:`BoundaryInfo <pymor.discretizers.builtin.grids.interfaces.BoundaryInfo>`
.. |BoundaryInfos| replace:: :class:`BoundaryInfos <pymor.discretizers.builtin.grids.interfaces.BoundaryInfo>`
.. |ConformalTopologicalGrids| replace:: :class:`ConformalTopologicalGrids <pymor.discretizers.builtin.grids.interfaces.ConformalTopologicalGrid>`
.. |ConformalTopologicalGrid| replace:: :class:`ConformalTopologicalGrid <pymor.discretizers.builtin.grids.interfaces.ConformalTopologicalGrid>`
.. |Models| replace:: :class:`Models <pymor.models.interface.Model>`
.. |Model| replace:: :class:`Model <pymor.models.interface.Model>`
.. |DomainDescriptions| replace:: :class:`DomainDescriptions <pymor.analyticalproblems.domaindescriptions.DomainDescription>`
.. |DomainDescription| replace:: :class:`DomainDescription <pymor.analyticalproblems.domaindescriptions.DomainDescription>`
.. |Functionals| replace:: :class:`Functionals <pymor.operators.interface.Operator>`
.. |Functional| replace:: :class:`Functional <pymor.operators.interface.Operator>`
.. |Functions| replace:: :class:`Functions <pymor.analyticalproblems.functions.Function>`
.. |Function| replace:: :class:`Function <pymor.analyticalproblems.functions.Function>`
.. |Grid| replace:: :class:`Grid <pymor.discretizers.builtin.grids.interfaces.AffineGrid>`
.. |Grids| replace:: :class:`Grids <pymor.discretizers.builtin.grids.interfaces.AffineGrid>`
.. |ImmutableObject| replace:: :class:`~pymor.core.base.ImmutableObject`
.. |immutable| replace:: :class:`immutable <pymor.core.base.ImmutableObject>`
.. |Immutable| replace:: :class:`Immutable <pymor.core.base.ImmutableObject>`
.. |LincombOperators| replace:: :class:`LincombOperators <pymor.operators.constructions.LincombOperator>`
.. |LincombOperator| replace:: :class:`LincombOperator <pymor.operators.constructions.LincombOperator>`
.. |Operators| replace:: :class:`Operators <pymor.operators.interface.Operator>`
.. |Operator| replace:: :class:`Operator <pymor.operators.interface.Operator>`
.. |ParameterFunctionals| replace:: :class:`ParameterFunctionals <pymor.parameters.functionals.ParameterFunctional>`
.. |ParameterFunctional| replace:: :class:`ParameterFunctional <pymor.parameters.functionals.ParameterFunctional>`
.. |ParameterSpace| replace:: :class:`ParameterSpace <pymor.parameters.spaces.ParameterSpace>`
.. |ParameterSpaces| replace:: :class:`ParameterSpaces <pymor.parameters.spaces.ParameterSpace>`
.. |ReferenceElements| replace:: :class:`ReferenceElements <pymor.discretizers.builtin.grids.interfaces.ReferenceElement>`
.. |ReferenceElement| replace:: :class:`ReferenceElement <pymor.discretizers.builtin.grids.interfaces.ReferenceElement>`
.. |RemoteObject| replace:: :class:`RemoteObject <pymor.parallel.interface.RemoteObject>`
.. |RemoteObjects| replace:: :class:`RemoteObjects <pymor.parallel.interface.RemoteObject>`
.. |VectorArrays| replace:: :class:`VectorArrays <pymor.vectorarrays.interface.VectorArray>`
.. |VectorArray| replace:: :class:`VectorArray <pymor.vectorarrays.interface.VectorArray>`
.. |VectorSpace| replace:: :class:`VectorSpace <pymor.vectorarrays.interface.VectorSpace>`
.. |VectorSpaces| replace:: :class:`VectorSpaces <pymor.vectorarrays.interface.VectorSpace>`
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

.. |StationaryProblem| replace:: :class:`~pymor.analyticalproblems.elliptic.StationaryProblem`
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

.. |NumpyMatrixOperator| replace:: :class:`~pymor.operators.numpy.NumpyMatrixOperator`
.. |NumpyMatrixOperators| replace:: :class:`NumpyMatrixOperators <pymor.operators.numpy.NumpyMatrixOperator>`
.. |NumpyMatrixBasedOperator| replace:: :class:`~pymor.operators.numpy.NumpyMatrixBasedOperator`
.. |NumpyMatrixBasedOperators| replace:: :class:`NumpyMatrixBasedOperators <pymor.operators.numpy.NumpyMatrixBasedOperator>`
.. |NumpyGenericOperator| replace:: :class:`~pymor.operators.numpy.NumpyGenericOperator`
.. |EmpiricalInterpolatedOperator| replace:: :class:`~pymor.operators.ei.EmpiricalInterpolatedOperator`
.. |EmpiricalInterpolatedOperators| replace:: :class:`EmpiricalInterpolatedOperators <pymor.operators.ei.EmpiricalInterpolatedOperator>`
.. |Concatenation| replace:: :class:`~pymor.operators.constructions.Concatenation`
.. |NumpyVectorSpace| replace:: :func:`~pymor.vectorarrays.numpy.NumpyVectorSpace`
.. |NumpyVectorSpaces| replace:: :func:`NumpyVectorSpaces <pymor.vectorarrays.numpy.NumpyVectorSpace>`

.. |StationaryModel| replace:: :class:`~pymor.models.basic.StationaryModel`
.. |StationaryModels| replace:: :class:`StationaryModels <pymor.models.basic.StationaryModel>`
.. |InstationaryModel| replace:: :class:`~pymor.models.basic.InstationaryModel`

.. |LTIModel| replace:: :class:`~pymor.models.iosys.LTIModel`
.. |LTIModels| replace:: :class:`LTIModels <pymor.models.iosys.LTIModel>`
.. |TransferFunction| replace:: :class:`~pymor.models.iosys.TransferFunction`
.. |TransferFunctions| replace:: :class:`TransferFunctions <pymor.models.iosys.TransferFunction>`
.. |SecondOrderModel| replace:: :class:`~pymor.models.iosys.SecondOrderModel`
.. |SecondOrderModels| replace:: :class:`SecondOrderModels <pymor.models.iosys.SecondOrderModel>`
.. |LinearDelayModel| replace:: :class:`~pymor.models.iosys.LinearDelayModel`
.. |LinearDelayModels| replace:: :class:`LinearDelayModels <pymor.models.iosys.LinearDelayModel>`

.. |ParameterType| replace:: :class:`~pymor.parameters.base.ParameterType`
.. |ParameterTypes| replace:: :class:`ParameterTypes <pymor.parameters.base.ParameterType>`
.. |Parameter| replace:: :class:`~pymor.parameters.base.Parameter`
.. |Parameters| replace:: :class:`Parameters <pymor.parameters.base.Parameter>`
.. |Parametric| replace:: :class:`~pymor.parameters.base.Parametric`
.. |parametric| replace:: :attr:`~pymor.parameters.base.Parametric.parametric`

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

'''

substitutions = interfaces + common
