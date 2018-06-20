# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""This module imports some commonly used methods and classes.

You can use ``from pymor.basic import *`` in interactive session
to have the most important parts of pyMOR directly available.
"""

# flake8: noqa

from pymor.algorithms.basic import almost_equal, relative_error, project_array
from pymor.algorithms.ei import interpolate_operators, ei_greedy, deim
from pymor.algorithms.error import reduction_error_analysis
from pymor.algorithms.gram_schmidt import gram_schmidt, gram_schmidt_biorth
from pymor.algorithms.greedy import greedy
from pymor.algorithms.adaptivegreedy import adaptive_greedy
from pymor.algorithms.newton import newton
from pymor.algorithms.pod import pod
from pymor.algorithms.preassemble import preassemble
from pymor.algorithms.projection import project, project_to_subbasis

from pymor.analyticalproblems.burgers import burgers_problem, burgers_problem_2d
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.helmholtz import helmholtz_problem
from pymor.analyticalproblems.instationary import InstationaryProblem
from pymor.analyticalproblems.thermalblock import thermal_block_problem
from pymor.analyticalproblems.text import text_problem

from pymor.core.cache import clear_caches, enable_caching, disable_caching
from pymor.core.defaults import print_defaults, write_defaults_to_file, load_defaults_from_file, set_defaults
from pymor.core.logger import set_log_levels, getLogger
from pymor.core.pickle import dump, dumps, load, loads

from pymor.discretizations.basic import StationaryDiscretization, InstationaryDiscretization
from pymor.discretizations.iosys import LTISystem, SecondOrderSystem, TransferFunction

from pymor.domaindescriptions.basic import RectDomain, CylindricalDomain, TorusDomain, LineDomain, CircleDomain
from pymor.domaindescriptions.polygonal import DiscDomain, CircularSectorDomain, PolygonalDomain

from pymor.domaindiscretizers.default import discretize_domain_default

from pymor.discretizers.cg import discretize_stationary_cg, discretize_instationary_cg
from pymor.discretizers.fv import discretize_stationary_fv, discretize_instationary_fv

from pymor.functions.basic import ConstantFunction, GenericFunction, ExpressionFunction, LincombFunction
from pymor.functions.bitmap import BitmapFunction

from pymor.grids.boundaryinfos import EmptyBoundaryInfo, BoundaryInfoFromIndicators, AllDirichletBoundaryInfo
from pymor.grids.oned import OnedGrid
from pymor.grids.rect import RectGrid
from pymor.grids.tria import TriaGrid

from pymor.operators.constructions import (LincombOperator, Concatenation, ComponentProjection, IdentityOperator,
                                           ConstantOperator, ZeroOperator, VectorArrayOperator, VectorOperator,
                                           VectorFunctional, FixedParameterOperator, AdjointOperator,
                                           SelectionOperator, induced_norm)
from pymor.operators.ei import EmpiricalInterpolatedOperator
from pymor.operators.numpy import NumpyGenericOperator, NumpyMatrixOperator

from pymor.parallel.default import new_parallel_pool
from pymor.parallel.manager import RemoteObjectManager

from pymor.parameters.base import Parameter
from pymor.parameters.functionals import (ProjectionParameterFunctional, GenericParameterFunctional,
                                          ExpressionParameterFunctional)
from pymor.parameters.spaces import CubicParameterSpace

from pymor.reductors.basic import GenericRBReductor
from pymor.reductors.bt import BTReductor, LQGBTReductor, BRBTReductor
from pymor.reductors.coercive import CoerciveRBReductor, SimpleCoerciveRBReductor
from pymor.reductors.h2 import IRKAReductor, TSIAReductor, TF_IRKAReductor
from pymor.reductors.interpolation import LTI_BHIReductor, SO_BHIReductor, TFInterpReductor
from pymor.reductors.parabolic import ParabolicRBReductor
from pymor.reductors.sobt import (SOBTpReductor, SOBTvReductor, SOBTpvReductor, SOBTvpReductor, SOBTfvReductor,
                                  SOBTReductor)
from pymor.reductors.sor_irka import SOR_IRKAReductor

from pymor.tools.random import new_random_state

from pymor.vectorarrays.constructions import cat_arrays
from pymor.vectorarrays.list import ListVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace
