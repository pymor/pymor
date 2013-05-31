# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.operators.interfaces import OperatorInterface, LinearOperatorInterface
from pymor.operators.basic import ConstantOperator
from pymor.operators.numpy import NumpyGenericOperator, NumpyLinearOperator
from pymor.operators.affine import LinearAffinelyDecomposedOperator
from pymor.operators.constructions import (ProjectedOperator, ProjectedLinearOperator, project_operator,
                                           rb_project_operator, LincombOperator, LinearLincombOperator, add_operators)
