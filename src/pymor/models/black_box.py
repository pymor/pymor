from collections.abc import Callable
from typing import Optional

import numpy as np
import numpy.typing as npt

from pymor.models.interface import Model
from pymor.parameters.base import Mu, Parameters
from pymor.vectorarrays.interface import VectorArray, VectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace


class BlackBoxModel(Model):
    def __init__(
        self,
        solution_space: VectorSpace,
        parameters: Parameters,
        solve_lambda: Callable[[Mu], VectorArray],
        *,
        visualizer=None,
        name: Optional[str] = None,
    ):
        super().__init__(visualizer=visualizer, name=name or 'BlackBoxModel')
        self.__auto_init(locals())

    def _compute(self, quantities, data, mu=None):
        if 'solution' in quantities:
            if self.solution_space.dim == 0:
                data['solution'] = self.solution_space.zeros(1)
            else:
                u = self.solve_lambda(mu)
                assert u in self.solution_space
                data['solution'] = u
            quantities.remove('solution')

        super()._compute(quantities, data, mu=mu)


class NumpyBlackBoxModel(BlackBoxModel):
    def __init__(
        self,
        solution_dim: int,
        parameters: Parameters,
        numpy_solve_lambda: Callable[[Mu], npt.ArrayLike],
        *,
        visualizer=None,
        name: Optional[str] = None,
    ):
        solution_space = NumpyVectorSpace(solution_dim)

        def solve_lambda(mu):
            u = np.array(numpy_solve_lambda(mu)).reshape(solution_space.dim, -1)
            return solution_space.from_numpy(u)

        super().__init__(
            solution_space=solution_space,
            parameters=parameters,
            solve_lambda=solve_lambda,
            visualizer=visualizer,
            name=name or 'NumpyBlackBoxModel')
        self.__auto_init(locals())
