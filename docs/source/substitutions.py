# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

# flake8: noqa

# define substitutions for all important interface classes
interfaces = '''

.. |AffineGrids| replace:: :class:`AffineGrids <pymor.grids.interfaces.AffineGridInterface>`
.. |AffineGrid| replace:: :class:`AffineGrid <pymor.grids.interfaces.AffineGridInterface>`
.. |BasicInterface| replace:: :class:`~pymor.core.interfaces.BasicInterface`
.. |BoundaryInfo| replace:: :class:`BoundaryInfo <pymor.grids.interfaces.BoundaryInfoInterface>`
.. |BoundaryInfos| replace:: :class:`BoundaryInfos <pymor.grids.interfaces.BoundaryInfoInterface>`
.. |ConformalTopologicalGrids| replace:: :class:`ConformalTopologicalGrids <pymor.grids.interfaces.ConformalTopologicalGridInterface>`
.. |ConformalTopologicalGrid| replace:: :class:`ConformalTopologicalGrid <pymor.grids.interfaces.ConformalTopologicalGridInterface>`
.. |Models| replace:: :class:`Models <pymor.models.interfaces.ModelInterface>`
.. |Model| replace:: :class:`Model <pymor.models.interfaces.ModelInterface>`
.. |DomainDescriptions| replace:: :class:`DomainDescriptions <pymor.domaindescriptions.interfaces.DomainDescriptionInterface>`
.. |DomainDescription| replace:: :class:`DomainDescription <pymor.domaindescriptions.interfaces.DomainDescriptionInterface>`
.. |Functionals| replace:: :class:`Functionals <pymor.operators.interfaces.OperatorInterface>`
.. |Functional| replace:: :class:`Functional <pymor.operators.interfaces.OperatorInterface>`
.. |Functions| replace:: :class:`Functions <pymor.functions.interfaces.FunctionInterface>`
.. |Function| replace:: :class:`Function <pymor.functions.interfaces.FunctionInterface>`
.. |Grid| replace:: :class:`Grid <pymor.grids.interfaces.AffineGridInterface>`
.. |Grids| replace:: :class:`Grids <pymor.grids.interfaces.AffineGridInterface>`
.. |ImmutableInterface| replace:: :class:`~pymor.core.interfaces.ImmutableInterface`
.. |immutable| replace:: :class:`immutable <pymor.core.interfaces.ImmutableInterface>`
.. |Immutable| replace:: :class:`Immutable <pymor.core.interfaces.ImmutableInterface>`
.. |LincombOperators| replace:: :class:`LincombOperators <pymor.operators.constructions.LincombOperator>`
.. |LincombOperator| replace:: :class:`LincombOperator <pymor.operators.constructions.LincombOperator>`
.. |Operators| replace:: :class:`Operators <pymor.operators.interfaces.OperatorInterface>`
.. |Operator| replace:: :class:`Operator <pymor.operators.interfaces.OperatorInterface>`
.. |ParameterFunctionals| replace:: :class:`ParameterFunctionals <pymor.parameters.interfaces.ParameterFunctionalInterface>`
.. |ParameterFunctional| replace:: :class:`ParameterFunctional <pymor.parameters.interfaces.ParameterFunctionalInterface>`
.. |ParameterSpace| replace:: :class:`ParameterSpace <pymor.parameters.interfaces.ParameterSpaceInterface>`
.. |ParameterSpaces| replace:: :class:`ParameterSpaces <pymor.parameters.interfaces.ParameterSpaceInterface>`
.. |ReferenceElements| replace:: :class:`ReferenceElements <pymor.grids.interfaces.ReferenceElementInterface>`
.. |ReferenceElement| replace:: :class:`ReferenceElement <pymor.grids.interfaces.ReferenceElementInterface>`
.. |RemoteObject| replace:: :class:`RemoteObject <pymor.parallel.interfaces.RemoteObjectInterface>`
.. |RemoteObjects| replace:: :class:`RemoteObjects <pymor.parallel.interfaces.RemoteObjectInterface>`
.. |VectorArrays| replace:: :class:`VectorArrays <pymor.vectorarrays.interfaces.VectorArrayInterface>`
.. |VectorArray| replace:: :class:`VectorArray <pymor.vectorarrays.interfaces.VectorArrayInterface>`
.. |VectorSpace| replace:: :class:`VectorSpace <pymor.vectorarrays.interfaces.VectorSpaceInterface>`
.. |VectorSpaces| replace:: :class:`VectorSpaces <pymor.vectorarrays.interfaces.VectorSpaceInterface>`
.. |WorkerPool| replace:: :class:`WorkerPool <pymor.parallel.interfaces.WorkerPoolInterface>`
.. |WorkerPools| replace:: :class:`WorkerPools <pymor.parallel.interfaces.WorkerPoolInterface>`

'''

# substitutions for the most important classes and methods in pyMOR
common = '''
.. |analytical problem| replace:: :mod:`analytical problem <pymor.analyticalproblems>`
.. |analytical problems| replace:: :mod:`analytical problems <pymor.analyticalproblems>`

.. |default| replace:: :mod:`default <pymor.core.defaults>`
.. |defaults| replace:: :mod:`~pymor.core.defaults`

.. |state id| replace:: :ref:`state id <state id>`
.. |state ids| replace:: :ref:`state ids <state id>`

.. |CacheRegion| replace:: :class:`~pymor.core.cache.CacheRegion`

.. |StationaryProblem| replace:: :class:`~pymor.analyticalproblems.elliptic.StationaryProblem`
.. |InstationaryProblem| replace:: :class:`~pymor.analyticalproblems.instationary.InstationaryProblem`

.. |RectDomain| replace:: :class:`~pymor.domaindescriptions.basic.RectDomain`
.. |PolygonalDomain| replace:: :class:`~pymor.domaindescriptions.polygonal.PolygonalDomain`
.. |CylindricalDomain| replace:: :class:`~pymor.domaindescriptions.basic.CylindricalDomain`
.. |TorusDomain| replace:: :class:`~pymor.domaindescriptions.basic.TorusDomain`
.. |LineDomain| replace:: :class:`~pymor.domaindescriptions.basic.LineDomain`
.. |CircleDomain| replace:: :class:`~pymor.domaindescriptions.basic.CircleDomain`
.. |discretize_domain_default| replace:: :func:`~pymor.domaindiscretizers.default.discretize_domain_default`

.. |OnedGrid| replace:: :class:`~pymor.grids.oned.OnedGrid`
.. |RectGrid| replace:: :class:`~pymor.grids.rect.RectGrid`
.. |TriaGrid| replace:: :class:`~pymor.grids.tria.TriaGrid`
.. |GmshGrid| replace:: :class:`~pymor.grids.gmsh.GmshGrid`

.. |NumpyVectorArray| replace:: :class:`~pymor.vectorarrays.numpy.NumpyVectorArray`
.. |NumpyVectorArrays| replace:: :class:`NumpyVectorArrays <pymor.vectorarrays.numpy.NumpyVectorArray>`
.. |ListVectorArray| replace:: :class:`~pymor.vectorarrays.list.ListVectorArray`
.. |ListVectorArrays| replace:: :class:`ListVectorArrays <pymor.vectorarrays.list.ListVectorArray>`

.. |OperatorBase| replace:: :class:`~pymor.operators.basic.OperatorBase`
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

.. |solver_options| replace:: :attr:`~pymor.operators.interfaces.OperatorInterface.solver_options`

.. |RuleTable| replace:: :class:`~pymor.algorithms.rules.RuleTable`
.. |RuleTables| replace:: :class:`RuleTables <pymor.algorithms.rules.RuleTable>`
.. |rule| replace:: :class:`~pymor.algorithms.rules.rule`
.. |rules| replace:: :class:`rules <pymor.algorithms.rules.rule>`
.. |project| replace:: :func:`~pymor.algorithms.projection.project`

'''

substitutions = interfaces + common
