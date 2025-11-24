from collections.abc import Callable
from numbers import Number
from typing import Optional

import numpy as np
import numpy.typing as npt

from pymor.models.interface import Model
from pymor.parameters.base import Mu, Parameters
from pymor.vectorarrays.interface import VectorArray, VectorSpace


class GenericModel(Model):

    def __init__(
        self,
        *,
        parameters: Parameters,
        computers: dict[str, tuple[VectorSpace | Number, Callable[[Mu], VectorArray | npt.ArrayLike]]],
        output_functional=None,
        visualizer=None,
        name: Optional[str] = None,
        ):
        assert isinstance(parameters, Parameters)

        assert isinstance(computers, dict)
        assert all(isinstance(k, str) for k in computers)
        assert all(isinstance(v, tuple) and len(v) == 2 for v in computers.values())
        assert all(isinstance(s, (VectorSpace, Number)) for s, _ in computers.values())
        assert all(callable(c) for _, c in computers.values())

        super().__init__(visualizer=visualizer, name=name or 'BlackBoxModel')

        if 'solution' in computers:
            self.solution_space = computers['solution'][0]
        if 'output' in computers:
            assert not isinstance(computers['output'][0], VectorSpace)
            self.dim_output = computers['output'][0]
            assert not output_functional, 'TODO: comes later'

        self.computable_quantities = set(computers.keys())

        self.__auto_init(locals())

    def _compute(self, quantities, data, mu=None):
        assert isinstance(quantities, (tuple, list, set))
        # we first compute all the data we're responsible for ...
        mu = self.parameters.parse(mu)
        for quantity in set(quantities):  # we modify quantities in place, so we need to iterate over a copy
            if quantity in self.computers:
                target_shape, computer = self.computers[quantity]
                if isinstance(target_shape, VectorSpace):
                    if target_shape.dim == 0:
                        data[quantity] = target_shape.zeros(1)
                    else:
                        computed_data = computer(mu)
                        assert computed_data in target_shape
                        data[quantity] = computed_data
                else:  # this needs to be the numpy case
                    if target_shape == 0:
                        data[quantity] = np.zeros((target_shape, 1))
                    else:
                        computed_data = computer(mu)
                        computed_data = computed_data.reshape(target_shape, -1)
                        data[quantity] = computed_data
                quantities.remove(quantity)
        # ... and delegate the rest to the parent class
        super()._compute(quantities, data, mu=mu)
