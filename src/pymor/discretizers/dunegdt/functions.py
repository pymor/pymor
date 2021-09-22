# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config

if config.HAVE_DUNEGDT:

    import numpy as np
    from numbers import Number

    from dune.xt.functions import ConstantFunction as DuneConstantFunction, GridFunction as DuneGridFunction
    from dune.xt.grid import Dim
    from dune.gdt.basic import (
            ContinuousLagrangeSpace,
            DiscontinuousLagrangeSpace,
            DiscreteFunction,
            FiniteVolumeSpace,
            default_interpolation,
            )

    from pymor.algorithms.rules import RuleTable, match_class, match_generic, RuleNotMatchingError, NoMatchingRuleError
    from pymor.core.base import BasicObject
    from pymor.analyticalproblems.functions import ConstantFunction, LincombFunction, ProductFunction, Function


    class DuneInterpolator(BasicObject):
        """Given a dune-xt-grid grid, interpolates pyMOR |Function|s by using the appropriate discrete function spaces.

        Parameters
        ----------
        grid
            A grid instance from dune.xt.grid.
        space_type
            A string identifying the target discrete function space:
            - dg (default): a non-conforming space (as in subspace of L^2) of piecewise polynomial functions
            - fv: a non-conforming space (as in subspace of L^2) of piecewise constant functions
            - cg: a conforming space (as in subspace of H^1) of piecewise polynomial functions
        order
            The local polynomial order of the elements of the target discrete function space (if not fv).
        """

        def __init__(self, grid, space_type='dg', order=1):
            assert space_type in ('dg', 'fv', 'cg')
            assert isinstance(order, Number)
            if space_type == 'cg':
                assert order >= 1
            else:
                assert order >= 0
            self._spaces = {}
            self._interpolation_points = {}
            super().__init__()
            self.__auto_init(locals())

        def interpolate(self, pymor_function, mu=None):
            assert isinstance(pymor_function, Function)
            assert pymor_function.dim_domain == self.grid.dimension
            # cache the space and interpolation points
            if pymor_function.shape_range == ():
                r = 1
            elif len(pymor_function.shape_range) == 1:
                r = pymor_function.shape_range[0]
            else:
                raise RuleNotMatchingError('Interpolation of matrix- or tensor-valued functions not implemented yet!')
            if r not in self._spaces:
                if r > 1 and self.space_type != 'dg':
                    self.logger.warn(f'Interpolation of vector-valued functions into a {space_type} space '
                                     'is not supported yet, defaulting to a dg space!')
                    self._spaces[r] = DiscontinuousLagrangeSpace(
                            self.grid, order=self.order, dim_range=Dim(r), dimwise_global_mapping=True)
                if self.space_type == 'cg':
                    self._spaces[r] = ContinuousLagrangeSpace(self.grid, order=self.order, dim_range=Dim(r))
                elif self.space_type == 'dg':
                    self._spaces[r] = DiscontinuousLagrangeSpace(self.grid, order=self.order, dim_range=Dim(r))
                elif self.space_type == 'fv':
                    self._spaces[r] = FiniteVolumeSpace(self.grid, dim_range=Dim(r))
                else:
                    assert False, 'this should not happen'
            space = self._spaces[r]
            if r not in self._interpolation_points:
                self._interpolation_points[r] = space.interpolation_points()
            interpolation_points = self._interpolation_points[r]
            # carry out the actual interpolation
            dune_function = DiscreteFunction(space)
            if r == 1:
                np_view = np.array(dune_function.dofs.vector, copy=False)
                np_view[:] = pymor_function.evaluate(interpolation_points, mu=mu)[:].ravel()
            else:
                np_view = np.array(dune_function.dofs.vector, copy=False)
                values = pymor_function.evaluate(interpolation_points)
                reshaped_values = []
                for rr in range(r):
                    reshaped_values.append(np.array(values[:, rr], copy=False))
                reshaped_values = np.hstack(reshaped_values)
                np_view[:] = reshaped_values.ravel()[:]
            return dune_function


    def to_dune_grid_function(pymor_function, dune_grid=None, dune_interpolator=None, mu=None, ensure_lincomb=False):
        """Converts |Function|s from `pymor.analyticalproblems.functions` to grid functions compatible with dune-gdt by
           interpolation, if required.

        Parameters
        ----------
        pymor_function
            The pyMOR |Function| to convert.
        dune_grid
            If not None, use to create the DuneInterpolator.
        dune_interpolator
            An instance of DuneInterpolator to cache the required discrete function spaces and interpolation points.
        mu
            If not None and `pymor_function` is parametric, returns an interpolation of the resulting non-parametric
            function.
        ensure_lincomb
            If True, always returns a tuple of functions and coefficients (even if only one function was converted)
        """

        assert dune_grid is not None or dune_interpolator is not None
        if dune_interpolator is None:
            dune_interpolator = DuneInterpolator(dune_grid)
        if dune_grid is not None:
            assert dune_grid == dune_interpolator.grid

        return ToDuneGridFunctionRules(
                mu=mu, interpolator=dune_interpolator, ensure_lincomb=ensure_lincomb).apply(pymor_function)


    def to_dune_function(pymor_function, ensure_lincomb=False):
        """Converts |Function|s from `pymor.analyticalproblems.functions` to functions compatible with dune-gdt.

        NOTE: usually not to be used directly, see `to_dune_grid_function` instead.

        Parameters
        ----------
        pymor_function
            The pyMOR |Function| to convert.
        ensure_lincomb
            If True, always returns a tuple of functions and coefficients (even if only one function was converted)
        """
        return ToDuneFunctionRules(ensure_lincomb).apply(pymor_function)


    class ToDuneGridFunctionRules(RuleTable):

        def __init__(self, mu, interpolator, ensure_lincomb):
            super().__init__()
            self.__auto_init(locals())

        @match_generic(lambda pymor_function: _is_convertible_to_dune_function(pymor_function),
                       'convertible with to_dune_function')
        def action_to_dune_function_convertible_function(self, pymor_function):
            assert pymor_function.dim_domain == self.interpolator.grid.dimension
            result = to_dune_function(pymor_function)
            if isinstance(result, tuple):
                dune_functions, coefficients = result
                dune_functions = [DuneGridFunction(self.interpolator.grid, dune_function)
                                  for dune_function in dune_functions]
                return dune_functions, coefficients
            else:
                dune_function = DuneGridFunction(self.interpolator.grid, result)
                if self.ensure_lincomb:
                    return [dune_function,], [1,]
                else:
                    return dune_function

        @match_class(LincombFunction)
        def action_LincombFunction(self, pymor_function):
            if pymor_function.parametric and self.mu is not None:
                raise RuleNotMatchingError('Delegate to directly interpolate this as a Function with fixed mu.')
            elif pymor_function.parametric:
                assert not any([func.parametric for func in pymor_function.functions])  # does not work without a mu
            dune_functions = [self.apply(func) for func in pymor_function.functions]
            return dune_functions, pymor_function.coefficients

        @match_class(Function)
        def action_Function(self, pymor_function):
            assert pymor_function.dim_domain == self.interpolator.grid.dimension
            if pymor_function.parametric:
                assert self.mu is not None
            dune_function = self.interpolator.interpolate(pymor_function, mu=self.mu)
            if self.ensure_lincomb:
                return [dune_function,], [1,]
            else:
                return dune_function


    def _is_convertible_to_dune_function(pymor_function):
        try:
            _ = to_dune_function(pymor_function)
            return True
        except NoMatchingRuleError:
            return False


    class ToDuneFunctionRules(RuleTable):

        def __init__(self, ensure_lincomb):
            super().__init__()
            self.__auto_init(locals())

        @match_class(ConstantFunction)
        def action_ConstantFunction(self, pymor_function):
            dim_domain = Dim(pymor_function.dim_domain)
            value = pymor_function.value
            if pymor_function.shape_range == ():
                dim_range = Dim(1)
                value = [value,]
            elif len(pymor_function.shape_range) == 1:
                dim_range = Dim(pymor_function.shape_range[0])
            elif len(pymor_function.shape_range) == 2:
                dim_range = (Dim(pymor_function.shape_range[0]), Dim(pymor_function.shape_range[1]))
            dune_function = DuneConstantFunction(
                    dim_domain=dim_domain, dim_range=dim_range, value=value, name=pymor_function.name)
            if self.ensure_lincomb:
                return [dune_function,], [1,]
            else:
                return dune_function

        @match_class(ProductFunction)
        def action_ProductFunction(self, pymor_function):
            if pymor_function.parametric:
                raise RuleNotMatchingError('We cannot treat products of parametric functions!')
            dune_functions = [self.apply(func) for func in pymor_function.functions]
            dune_function = dune_functions[0]
            for ii in range(1, len(dune_functions)):
                dune_function = dune_function*dune_functions[ii]
            if self.ensure_lincomb:
                return [dune_function,], [1,]
            else:
                return dune_function

        @match_class(LincombFunction)
        def action_LincombFunction(self, pymor_function):
            for func in pymor_function.functions:
                if func.parametric:
                    raise RuleNotMatchingError('We cannot treat linear combinations of parametric functions!')
            dune_functions = [self.apply(func) for func in pymor_function.functions]
            return dune_functions, pymor_function.coefficients
