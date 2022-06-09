# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""This module imports some commonly used methods and classes.

You can use ``from pymor.basic import *`` in interactive session
to have the most important parts of pyMOR directly available.
"""

# flake8: noqa

from pymor.algorithms.basic import almost_equal, relative_error, project_array
from pymor.algorithms.dmd import dmd
from pymor.algorithms.ei import interpolate_operators, interpolate_function, ei_greedy, deim
from pymor.algorithms.error import plot_reduction_error_analysis, reduction_error_analysis
from pymor.algorithms.gram_schmidt import gram_schmidt, gram_schmidt_biorth
from pymor.algorithms.greedy import rb_greedy
from pymor.algorithms.adaptivegreedy import rb_adaptive_greedy
from pymor.algorithms.newton import newton
from pymor.algorithms.pod import pod
from pymor.algorithms.preassemble import preassemble
from pymor.algorithms.projection import project, project_to_subbasis
from pymor.algorithms.simplify import expand

from pymor.analyticalproblems.burgers import burgers_problem, burgers_problem_2d
from pymor.analyticalproblems.domaindescriptions import (RectDomain, CylindricalDomain, TorusDomain, LineDomain,
                                                         CircleDomain)
from pymor.analyticalproblems.domaindescriptions import DiscDomain, CircularSectorDomain, PolygonalDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import (ConstantFunction, GenericFunction, ExpressionFunction, LincombFunction,
                                                BitmapFunction)
from pymor.analyticalproblems.helmholtz import helmholtz_problem
from pymor.analyticalproblems.instationary import InstationaryProblem
from pymor.analyticalproblems.thermalblock import thermal_block_problem
from pymor.analyticalproblems.text import text_problem

from pymor.core.cache import clear_caches, enable_caching, disable_caching
from pymor.core.defaults import print_defaults, write_defaults_to_file, load_defaults_from_file, set_defaults
from pymor.core.logger import set_log_levels, getLogger
from pymor.core.pickle import dump, dumps, load, loads

from pymor.models.basic import StationaryModel, InstationaryModel
from pymor.models.iosys import LTIModel, PHLTIModel, SecondOrderModel
from pymor.models.transfer_function import TransferFunction

from pymor.discretizers.builtin import (discretize_stationary_cg, discretize_instationary_cg,
                                        discretize_stationary_fv, discretize_instationary_fv,
                                        OnedGrid, TriaGrid, RectGrid, load_gmsh)
from pymor.discretizers.builtin.domaindiscretizers.default import discretize_domain_default
from pymor.discretizers.builtin.grids.boundaryinfos import (EmptyBoundaryInfo, GenericBoundaryInfo,
                                                            AllDirichletBoundaryInfo)

from pymor.operators.constructions import (LincombOperator, ConcatenationOperator,
                                           ComponentProjectionOperator, IdentityOperator, ConstantOperator,
                                           ZeroOperator, VectorArrayOperator, VectorOperator, VectorFunctional,
                                           FixedParameterOperator, AdjointOperator, SelectionOperator, induced_norm)
from pymor.operators.ei import EmpiricalInterpolatedOperator
from pymor.operators.numpy import NumpyGenericOperator, NumpyMatrixOperator

from pymor.parallel.default import new_parallel_pool
from pymor.parallel.manager import RemoteObjectManager

from pymor.parameters.base import Parameters, Mu, ParametricObject, ParameterSpace
from pymor.parameters.functionals import (ProjectionParameterFunctional, GenericParameterFunctional,
                                          ExpressionParameterFunctional)

from pymor.reductors.basic import StationaryRBReductor, InstationaryRBReductor, LTIPGReductor, SOLTIPGReductor
from pymor.reductors.bt import BTReductor, LQGBTReductor, BRBTReductor
from pymor.reductors.coercive import CoerciveRBReductor, SimpleCoerciveRBReductor
from pymor.reductors.h2 import IRKAReductor, OneSidedIRKAReductor, TSIAReductor, TFIRKAReductor
from pymor.reductors.interpolation import LTIBHIReductor, SOBHIReductor, TFBHIReductor
from pymor.reductors.mt import MTReductor
from pymor.reductors.parabolic import ParabolicRBReductor
from pymor.reductors.sobt import (SOBTpReductor, SOBTvReductor, SOBTpvReductor, SOBTvpReductor, SOBTfvReductor,
                                  SOBTReductor)
from pymor.reductors.sor_irka import SORIRKAReductor

from pymor.tools.random import default_random_state

from pymor.vectorarrays.constructions import cat_arrays
from pymor.vectorarrays.list import ListVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace
