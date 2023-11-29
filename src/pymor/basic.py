# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""This module imports some commonly used methods and classes.

You can use ``from pymor.basic import *`` in interactive session
to have the most important parts of pyMOR directly available.
"""

from pymor.algorithms.adaptivegreedy import rb_adaptive_greedy
from pymor.algorithms.basic import almost_equal, project_array, relative_error
from pymor.algorithms.dmd import dmd
from pymor.algorithms.ei import deim, ei_greedy, interpolate_function, interpolate_operators
from pymor.algorithms.error import plot_reduction_error_analysis, reduction_error_analysis
from pymor.algorithms.gram_schmidt import gram_schmidt, gram_schmidt_biorth
from pymor.algorithms.greedy import rb_greedy
from pymor.algorithms.newton import newton
from pymor.algorithms.pod import pod
from pymor.algorithms.preassemble import preassemble
from pymor.algorithms.projection import project, project_to_subbasis
from pymor.algorithms.qr import qr, rrqr
from pymor.algorithms.simplify import expand
from pymor.analyticalproblems.burgers import burgers_problem, burgers_problem_2d
from pymor.analyticalproblems.domaindescriptions import (
    CircleDomain,
    CircularSectorDomain,
    CylindricalDomain,
    DiscDomain,
    LineDomain,
    PolygonalDomain,
    RectDomain,
    TorusDomain,
)
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import (
    BitmapFunction,
    ConstantFunction,
    ExpressionFunction,
    GenericFunction,
    LincombFunction,
)
from pymor.analyticalproblems.helmholtz import helmholtz_problem
from pymor.analyticalproblems.instationary import InstationaryProblem
from pymor.analyticalproblems.text import text_problem
from pymor.analyticalproblems.thermalblock import thermal_block_problem
from pymor.core.cache import clear_caches, disable_caching, enable_caching
from pymor.core.defaults import load_defaults_from_file, print_defaults, set_defaults, write_defaults_to_file
from pymor.core.exceptions import DependencyMissingError
from pymor.core.logger import getLogger, set_log_levels
from pymor.core.pickle import dump, dumps, load, loads
from pymor.discretizers.builtin import (
    OnedGrid,
    RectGrid,
    TriaGrid,
    discretize_instationary_cg,
    discretize_instationary_fv,
    discretize_stationary_cg,
    discretize_stationary_fv,
    load_gmsh,
)
from pymor.discretizers.builtin.domaindiscretizers.default import discretize_domain_default
from pymor.discretizers.builtin.grids.boundaryinfos import (
    AllDirichletBoundaryInfo,
    EmptyBoundaryInfo,
    GenericBoundaryInfo,
)
from pymor.models.basic import InstationaryModel, StationaryModel
from pymor.models.iosys import LTIModel, PHLTIModel, SecondOrderModel
from pymor.models.transfer_function import TransferFunction
from pymor.operators.constructions import (
    AdjointOperator,
    ComponentProjectionOperator,
    ConcatenationOperator,
    ConstantOperator,
    FixedParameterOperator,
    IdentityOperator,
    LincombOperator,
    SelectionOperator,
    VectorArrayOperator,
    VectorFunctional,
    VectorOperator,
    ZeroOperator,
    induced_norm,
)
from pymor.operators.ei import EmpiricalInterpolatedOperator
from pymor.operators.numpy import NumpyGenericOperator, NumpyMatrixOperator
from pymor.parallel.default import new_parallel_pool
from pymor.parallel.manager import RemoteObjectManager
from pymor.parameters.base import Mu, Parameters, ParameterSpace, ParametricObject
from pymor.parameters.functionals import (
    ExpressionParameterFunctional,
    GenericParameterFunctional,
    ProjectionParameterFunctional,
)
from pymor.reductors.basic import InstationaryRBReductor, LTIPGReductor, SOLTIPGReductor, StationaryRBReductor
from pymor.reductors.bt import BRBTReductor, BTReductor, LQGBTReductor, PRBTReductor
from pymor.reductors.coercive import CoerciveRBReductor, SimpleCoerciveRBReductor
from pymor.reductors.h2 import IRKAReductor, OneSidedIRKAReductor, TFIRKAReductor, TSIAReductor
from pymor.reductors.interpolation import LTIBHIReductor, SOBHIReductor, TFBHIReductor
from pymor.reductors.mt import MTReductor
from pymor.reductors.parabolic import ParabolicRBReductor
from pymor.reductors.sobt import (
    SOBTfvReductor,
    SOBTpReductor,
    SOBTpvReductor,
    SOBTReductor,
    SOBTvpReductor,
    SOBTvReductor,
)
from pymor.reductors.sor_irka import SORIRKAReductor
from pymor.tools.random import get_rng, new_rng
from pymor.vectorarrays.constructions import cat_arrays
from pymor.vectorarrays.list import ListVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace

try:
    from pymor.models.interact import interact
except DependencyMissingError:
    pass
