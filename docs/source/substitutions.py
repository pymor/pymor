# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

# flake8: noqa

# define substitutions for all important interface classes
interfaces = '''

.. |AffineGrid| replace:: :class:`AffineGrid <pymor.grids.interfaces.AffineGridInterface>`
.. |BoundaryInfo| replace:: :class:`BoundaryInfo <pymor.grids.interfaces.BoundaryInfoInterface>`
.. |ConformalTopologicalGrid| replace:: :class:`ConformalTopologicalGrid <pymor.grids.interfaces.ConformalTopologicalGrid>`
.. |Discretization| replace:: :class:`Discretization <pymor.discretizations.interfaces.DiscretizationInterface>`
.. |Discretizations| replace:: :class:`Discretizations <pymor.discretizations.interfaces.DiscretizationInterface>`
.. |DomainDescription| replace:: :class:`DomainDescription <pymor.domaindescriptions.interfaces.DomainDescriptionInterface>`
.. |Function| replace:: :class:`Function <pymor.functions.interfaces.FunctionInterface>`
.. |Functions| replace:: :class:`Functions <pymor.functions.interfaces.FunctionInterface>`
.. |LincombOperator| replace:: :class:`LincombOperator <pymor.operators.interfaces.LincombOperatorInterface>`
.. |Operator| replace:: :class:`Operator <pymor.operators.interfaces.OperatorInterface>`
.. |Operators| replace:: :class:`Operators <pymor.operators.interfaces.OperatorInterface>`
.. |Functional| replace:: :class:`Functional <pymor.operators.interfaces.OperatorInterface>`
.. |Functionals| replace:: :class:`Functionals <pymor.operators.interfaces.OperatorInterface>`
.. |ParameterFunctional| replace:: :class:`ParameterFunctional <pymor.parameters.interfaces.ParameterFunctionalInterface>`
.. |ParameterFunctionals| replace:: :class:`ParameterFunctionals <pymor.parameters.interfaces.ParameterFunctionalInterface>`
.. |ParameterSpace| replace:: :class:`ParameterSpace <pymor.parameters.interfaces.ParameterSpaceInterface>`
.. |ReferenceElement| replace:: :class:`ReferenceElement <pymor.grids.interfaces.ReferenceElementInterface>`
.. |VectorArray| replace:: :class:`VectorArray <pymor.la.interfaces.VectorArrayInterface>`
.. |VectorArrays| replace:: :class:`VectorArrays <pymor.la.interfaces.VectorArrayInterface>`

'''

# substitutions for the most important classes and methods in pyMOR
common = '''
.. |defaults| replace:: :attr:`~pymor.defaults.defaults`
.. |default| replace:: :attr:`default <pymor.defaults.defaults>`

.. |CacheRegion| replace:: :class:`~pymor.core.cache.CacheRegion`

.. |EllipticProblem| replace:: :class:`~pymor.analyticalproblems.elliptic.EllipticProblem`
.. |InstationaryAdvectionProblem| replace:: :class:`~pymor.analyticalproblems.advection.InstationaryAdvectionProblem`

.. |BoundaryType| replace:: :class:`~pymor.domaindescriptions.boundarytypes.BoundaryType`
.. |discretize_domain_default| replace:: :func:`~pymor.domaindiscretizers.default.discretize_domain_default`

.. |OnedGrid| replace:: :class:`~pymor.grids.oned.OnedGrid`
.. |RectGrid| replace:: :class:`~pymor.grids.rect.RectGrid`
.. |TriaGrid| replace:: :class:`~pymor.grids.tria.TriaGrid`

.. |NumpyVectorArray| replace:: :class:`~pymor.la.numpyvectorarray.NumpyVectorArray`
.. |ListVectorArray| replace:: :class:`~pymor.la.listvectorarray.ListVectorArray`

.. |NumpyMatrixOperator| replace:: :class:`~pymor.operators.basic.NumpyMatrixOperator`
.. |EmpiricalInterpolatedOperator| replace:: :class:`~pymor.operators.ei.EmpiricalInterpolatedOperator`
.. |EmpiricalInterpolatedOperators| replace:: :class:`EmpiricalInterpolatedOperators <pymor.operators.ei.EmpiricalInterpolatedOperator>`
.. |Concatenation| replace:: :class:`~pymor.operators.constructions.Concatenation`

.. |StationaryDiscretization| replace:: :class:`~pymor.discretizations.basic.StationaryDiscretization`
.. |InstationaryDiscretization| replace:: :class:`~pymor.discretizations.basic.InstationaryDiscretization`

.. |ParameterType| replace:: :class:`~pymor.parameters.base.ParameterType`
.. |Parameter| replace:: :class:`~pymor.parameters.base.Parameter`
.. |Parameters| replace:: :class:`Parameters <pymor.parameters.base.Parameter>`
.. |Parametric| replace:: :class:`~pymor.parameters.base.Parametric`

.. |reduce_generic_rb| replace:: :func:`~pymor.reductors.basic.reduce_generic_rb`

.. |NumPy array| replace:: :class:`NumPy array <numpy.ndarray>`
.. |Numpy array| replace:: :class:`NumPy array <numpy.ndarray>`
.. |array| replace:: :class:`NumPy array <numpy.ndarray>`
.. |Array| replace:: :class:`NumPy array <numpy.ndarray>`

'''

substitutions = interfaces + common
