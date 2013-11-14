# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.sparse.linalg import bicgstab
from scipy.sparse import issparse

from pymor import defaults
from pymor.core.cache import CacheableInterface, NO_CACHE_CONFIG
from pymor.playground.tools.dictproperty import dict_property
from pymor.discretizations.basic import StationaryDiscretization, DiscretizationBase
from pymor.playground.operators.dune import DuneLinearOperator, DuneLinearFunctional
from pymor.playground.la.dunevectorarray import DuneVectorArray, WrappedDuneVector
from pymor.parameters.spaces import CubicParameterSpace
from pymor.la import induced_norm, NumpyVectorArray

import dunelinearellipticcg2dsgrid as dune


class DuneLinearEllipticCGDiscretization(DiscretizationBase):

    disable_logging = False
    operator = dict_property('operators', 'operator')
    rhs = dict_property('operators', 'rhs')

    def __init__(self, parameter_range=(0.1, 1.), name=None):
        self.example = dune.LinearEllipticExampleCG()
        f = self.example.functional()
        functional = DuneLinearFunctional(f)
        self.solution_dim = f.len()
        ops = list(self.example.operators())
        ops = [DuneLinearOperator(op, dim=self.solution_dim) for op in ops]
        operator = ops[0].lincomb(ops, num_coefficients=len(ops) - 1, coefficients_name='diffusion', name='diffusion')

        operators = {'operator': operator, 'rhs': functional}
        products = {'h1': ops[0].lincomb(ops, coefficients=np.ones(self.example.paramSize() + 1))}
        super(DuneLinearEllipticCGDiscretization, self).__init__(operators=operators, products=products,
                                                                 caching=False, name=name)

        self.build_parameter_type(inherits=(operator,))
        self.parameter_space = CubicParameterSpace({'diffusion': self.example.paramSize()}, *parameter_range)

    def _solve(self, mu=None):
        mu = self.parse_parameter(mu)

        if not self.disable_logging:
            self.logger.info('Solving {} (sparse) for {} ...'.format(self.name, mu))

        return DuneVectorArray([WrappedDuneVector(self.example.solve(list(mu['diffusion'])))])

    with_arguments = StationaryDiscretization.with_arguments

    def with_(self, **kwargs):
        assert 'operators' in kwargs
        operators = kwargs.pop('operators')
        assert set(operators.keys()) == {'operator', 'rhs'}
        assert all(op.type_source == NumpyVectorArray for op in operators.itervalues())
        assert all(op.type_range == NumpyVectorArray for op in operators.itervalues())
        d = StationaryDiscretization(operator=operators['operator'], rhs=operators['rhs'])
        return d.with_(**kwargs)

    def visualize(self, U):
        import os
        assert isinstance(U, DuneVectorArray)
        assert len(U) == 1
        self.example.visualize(U._list[0]._vector, 'visualization', 'solution')
        os.system('paraview visualization.vtu')
