# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.sparse.linalg import bicgstab
from scipy.sparse import issparse

from pymor.core import defaults
from pymor.core.cache import Cachable, NO_CACHE_CONFIG
from pymor.tools import dict_property
from pymor.operators import LinearAffinelyDecomposedOperator
from pymor.discretizations.interfaces import DiscretizationInterface
from pymor.discretizations import StationaryLinearDiscretization
from pymor.playground.operators.dune import DuneLinearOperator, DuneLinearFunctional
from pymor.playground.la.dunevectorarray import DuneVectorArray, WrappedDuneVector
from pymor.parameters.spaces import CubicParameterSpace
from pymor.la import induced_norm

import dunelinearellipticcg2dsgrid as dune


class DuneLinearEllipticCGDiscretization(DiscretizationInterface):

    disable_logging = False
    operator = dict_property('operators', 'operator')
    rhs = dict_property('operators', 'rhs')

    def __init__(self, parameter_range=(0.1, 1.), name=None):
        super(DuneLinearEllipticCGDiscretization, self).__init__()
        Cachable.__init__(self, config=NO_CACHE_CONFIG)

        self.example = dune.LinearEllipticExampleCG()
        ops = list(self.example.operators())
        f = self.example.functional()
        self.solution_dim = f.len()

        ops = [DuneLinearOperator(op, dim=self.solution_dim) for op in ops]
        operator = LinearAffinelyDecomposedOperator(ops[:-1], ops[-1], name_map={'.coefficients': 'diffusion'}, name='diffusion')
        functional = DuneLinearFunctional(f)

        self.operators = {'operator': operator, 'rhs': functional}
        self.build_parameter_type(inherits={'operator': operator})

        self.h1_product = operator.assemble(mu={'diffusion': np.ones(self.example.paramSize())})
        self.h1_norm = induced_norm(self.h1_product)

        self.parameter_space = CubicParameterSpace({'diffusion': self.example.paramSize()}, *parameter_range)

        self.name = name

    def _solve(self, mu=None):
        mu = self.map_parameter(mu, 'operator')

        if not self.disable_logging:
            self.logger.info('Solving {} (sparse) for {} ...'.format(self.name, mu))

        return DuneVectorArray([WrappedDuneVector(self.example.solve(list(mu['diffusion'])))])

    def with_projected_operators(self, operators):
        assert set(operators.keys()) == {'operator', 'rhs'}
        return StationaryLinearDiscretization(operator=operators['operator'], rhs=operators['rhs'])

    def visualize(self, U):
        import os
        assert isinstance(U, DuneVectorArray)
        assert len(U) == 1
        self.example.visualize(U._list[0]._vector, 'visualization', 'solution')
        os.system('paraview visualization.vtu')
