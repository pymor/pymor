# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.algorithms.basisextension import trivial_basis_extension, gram_schmidt_basis_extension, pod_basis_extension
from pymor.algorithms.ei import interpolate_operators, ei_greedy, deim
from pymor.algorithms.greedy import greedy

from pymor.analyticalproblems.advection import InstationaryAdvectionProblem
from pymor.analyticalproblems.burgers import BurgersProblem, Burgers2DProblem
from pymor.analyticalproblems.elliptic import EllipticProblem
from pymor.analyticalproblems.thermalblock import ThermalBlockProblem

from pymor.core.cache import clear_caches, enable_caching, disable_caching
from pymor.core.defaults import print_defaults, write_defaults_to_file, load_defaults_from_file, set_defaults
from pymor.core.logger import set_log_levels, getLogger
from pymor.core.pickle import dump, dumps, load, loads

from pymor.discretizations.basic import StationaryDiscretization, InstationaryDiscretization

from pymor.domaindescriptions.basic import RectDomain, CylindricalDomain, TorusDomain, LineDomain, CircleDomain
from pymor.domaindescriptions.boundarytypes import BoundaryType

from pymor.domaindiscretizers.default import discretize_domain_default

from pymor.discretizers.advection import discretize_nonlinear_instationary_advection_fv
from pymor.discretizers.elliptic import discretize_elliptic_cg

from pymor.functions.basic import ConstantFunction, GenericFunction, ExpressionFunction, LincombFunction

from pymor.grids.boundaryinfos import EmptyBoundaryInfo, BoundaryInfoFromIndicators, AllDirichletBoundaryInfo
from pymor.grids.oned import OnedGrid
from pymor.grids.rect import RectGrid
from pymor.grids.tria import TriaGrid

from pymor.la.basic import induced_norm, cat_arrays
from pymor.la.gram_schmidt import gram_schmidt
from pymor.la.interfaces import VectorSpace
from pymor.la.numpyvectorarray import NumpyVectorArray, NumpyVectorSpace
from pymor.la.pod import pod

from pymor.operators.numpy import NumpyGenericOperator, NumpyMatrixOperator
from pymor.operators.constructions import (LincombOperator, Concatenation, ComponentProjection, IdentityOperator,
                                           ConstantOperator, VectorArrayOperator, VectorOperator, VectorFunctional,
                                           FixedParameterOperator)
from pymor.operators.ei import EmpiricalInterpolatedOperator

from pymor.parameters.base import Parameter
from pymor.parameters.functionals import (ProjectionParameterFunctional, GenericParameterFunctional,
                                          ExpressionParameterFunctional)
from pymor.parameters.spaces import CubicParameterSpace

from pymor.reductors.basic import reduce_generic_rb, reduce_to_subbasis
from pymor.reductors.stationary import reduce_stationary_coercive

from pymor.tools.floatcmp import float_cmp, float_cmp_all
from pymor.tools.random import new_random_state
