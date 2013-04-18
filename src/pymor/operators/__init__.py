# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from .interfaces import OperatorInterface, LinearOperatorInterface
from .basic import NumpyGenericOperator, NumpyLinearOperator
from .affine import LinearAffinelyDecomposedOperator
from .constructions import (ProjectedOperator, ProjectedLinearOperator, project_operator,
                            LincombOperator, LinearLincombOperator, add_operators)
