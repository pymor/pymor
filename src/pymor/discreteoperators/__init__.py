from .interfaces import DiscreteOperatorInterface, LinearDiscreteOperatorInterface
from .basic import GenericOperator, GenericLinearOperator
from .affine import LinearAffinelyDecomposedOperator
from .constructions import (ProjectedOperator, ProjectedLinearOperator, project_operator,
                            SumOperator, LinearSumOperator, add_operators)
