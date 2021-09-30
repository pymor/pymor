# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config

if config.HAVE_DUNEGDT:

    import numpy as np
    from numbers import Number

    import dune.xt.functions
    from dune.xt.functions import (
            ConstantFunction as DuneXTConstantFunction, GridFunction as DuneXTGridFunction)
    from dune.xt.grid import Dim
    from dune.gdt.basic import (
            ContinuousLagrangeSpace,
            DiscontinuousLagrangeSpace,
            DiscreteFunction,
            FiniteVolumeSpace,
            default_interpolation,
            )

    from pymor.algorithms.rules import RuleTable, match_class, match_generic, RuleNotMatchingError, NoMatchingRuleError
    from pymor.analyticalproblems.functions import ConstantFunction, LincombFunction, ProductFunction, Function
    from pymor.core.base import BasicObject, ImmutableObject, classinstancemethod
    from pymor.parameters.base import ParametricObject


    def to_dune_grid_function(function, dune_grid=None, dune_interpolator=None, mu=None, ensure_lincomb=False):
        """Converts pyMOR |Function|s to equivalent grid functions from `dune.xt.functions`.

        Conversion is either done by creating exactly equivalent grid functions from
        `dune.xt.functions` or by approximation by interpolation.

        Parameters
        ----------
        function
            The pyMOR |Function| to convert.
        dune_grid
            A grid from `dune.xt.grid`. If not `None`, used to create the :class:`DuneInterpolator`.
        dune_interpolator
            An instance of :class:`DuneInterpolator` to cache the required discrete function spaces
            and interpolation points.
        mu
            If not `None` and `function` is parametric, returns an interpolation of the
            resulting non-parametric function.
        ensure_lincomb
            If `True`, always returns a :class:`LincombDuneGridFunction`, otherwise a
            :class:`DuneGridFunction` or :class:`LincombDuneGridFunction` (as deduced from
            `function`).
        """

        assert dune_grid is not None or dune_interpolator is not None
        if dune_interpolator is None:
            dune_interpolator = DuneInterpolator(dune_grid)
        if dune_grid is not None:
            assert dune_grid == dune_interpolator.grid

        return ToDuneGridFunctionRules(
                mu=mu, interpolator=dune_interpolator, ensure_lincomb=ensure_lincomb).apply(function)


    def to_dune_function(function, ensure_lincomb=False):
        """Converts pyMOR |Function|s to equivalent functions from `dune.xt.functions`.

        Conversion is done by creating exactly equivalent functions from `dune.xt.functions`.

        NOTE: usually not to be used directly, see `to_dune_grid_function` instead.

        Parameters
        ----------
        function
            The pyMOR |Function| to convert.
        ensure_lincomb
            If `True`, always returns a :class:`LincombDuneFunction`, otherwise a
            :class:`DuneFunction` or :class:`LincombDuneFunction` (as deduced from
            `function`).
        """
        return ToDuneFunctionRules(ensure_lincomb).apply(function)


    # collect once on first module import
    _known_dune_function_interfaces = []       # required for DuneFunction
    _known_dune_grid_function_interfaces = []  # required for DuneGridFunction
    if len(_known_dune_function_interfaces) == 0:
        for cls_nm in dune.xt.functions.__dict__.keys():
            if cls_nm.startswith('GridFunctionInterface'):
                _known_dune_grid_function_interfaces.append(dune.xt.functions.__dict__[cls_nm])
            elif cls_nm.startswith('FunctionInterface'):
                _known_dune_function_interfaces.append(dune.xt.functions.__dict__[cls_nm])


    class DuneFunction(ParametricObject):
        """Convenience wrapper class for functions from `dune.xt.functions`.

        Allows to check if a given `obj` is a function from `dune.xt.functions` by
        `assert DuneFunction.is_base_of(obj)`.

        Parameters
        ----------
        impl
            A function from `dune.xt.functions`.
        """
        def __init__(self, impl):
            assert DuneFunction.is_base_of(impl)
            self.__auto_init(locals())

        @classinstancemethod
        def is_base_of(cls, obj):
            for interface in _known_dune_function_interfaces:
                if isinstance(obj, interface):
                    return True
            return False

    class LincombDuneFunction(ParametricObject):
        """Convenience wrapper class for linear combinations of functions from `dune.xt.functions`
        """
        def __init__(self, functions, coefficients):
            assert isinstance(functions, (list, tuple))
            assert isinstance(coefficients, (list, tuple))
            assert len(functions) == len(coefficients)
            assert all(DuneFunction.is_base_of(func) for func in functions)
            functions = tuple(functions)
            coefficients = tuple(coefficients)
            self.__auto_init(locals())


    class DuneGridFunction(ParametricObject):
        """Convenience wrapper class for grid functions from `dune.xt.functions`.

        Allows to check if a given `obj` is a grid function from `dune.xt.functions` by
        `assert DuneGridFunction.is_base_of(obj)`.

        Parameters
        ----------
        impl
            A grid function from `dune.xt.functions`.
        """
        def __init__(self, impl):
            assert DuneGridFunction.is_base_of(impl)
            self.__auto_init(locals())

        @classinstancemethod
        def is_base_of(cls, obj):
            for interface in _known_dune_grid_function_interfaces:
                if isinstance(obj, interface):
                    return True
            return False

    class LincombDuneGridFunction(ParametricObject):
        """Convenience wrapper class for linear combinations of grid functions from
        `dune.xt.functions`
        """
        def __init__(self, functions, coefficients):
            assert isinstance(functions, (list, tuple))
            assert isinstance(coefficients, (list, tuple))
            assert len(functions) == len(coefficients)
            assert all(DuneGridFunction.is_base_of(func) for func in functions)
            functions = tuple(functions)
            coefficients = tuple(coefficients)
            self.__auto_init(locals())

    class DuneInterpolator(BasicObject):
        """Interpolates pyMOR |Function|s within discrete function spaces from `dune.gdt`.

        Parameters
        ----------
        grid
            A grid instance from `dune.xt.grid`.
        space_type
            A string identifying the target discrete function space:
            - dg (default): a non-conforming space (as in subspace of L^2) of piecewise polynomial
              functions
            - fv: a non-conforming space (as in subspace of L^2) of piecewise constant functions
            - cg: a conforming space (as in subspace of H^1) of piecewise polynomial functions
        order
            The local polynomial order of the elements of the target discrete function space (if
            not fv).
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
                elif self.space_type == 'cg':
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
                np_view[:] = pymor_function.evaluate(interpolation_points, mu=mu).ravel()[:]
            else:
                np_view = np.array(dune_function.dofs.vector, copy=False)
                values = pymor_function.evaluate(interpolation_points)
                reshaped_values = []
                for rr in range(r):
                    reshaped_values.append(np.array(values[:, rr], copy=False))
                reshaped_values = np.hstack(reshaped_values)
                np_view[:] = reshaped_values.ravel()[:]
            return dune_function

    class ToDuneGridFunctionRules(RuleTable):

        def __init__(self, mu, interpolator, ensure_lincomb):
            super().__init__()
            self.__auto_init(locals())

        @match_generic(lambda function: DuneGridFunction.is_base_of(function),
                       'grid function from dune.xt.functions')
        def action_dune_xt_functions_grid_function(self, function):
            if self.ensure_lincomb:
                return LincombDuneGridFunction([function,], [1,])
            else:
                return DuneGridFunction(function)

        @match_class(LincombDuneGridFunction)
        def action_LincombDuneGridFunction(self, function):
            return function

        @match_class(DuneGridFunction)
        def action_DuneGridFunction(self, function):
            if self.ensure_lincomb:
                return LincombDuneGridFunction([function.impl,], [1,])
            else:
                return function

        @match_class(LincombDuneFunction)
        def action_LincombDuneFunction(self, function):
            return LincombDuneGridFunction(
                    [DuneXTGridFunction(self.interpolator.grid, func) for func in function.functions],
                    function.coefficients)

        @match_class(DuneFunction)
        def action_DuneFunction(self, function):
            function = DuneXTGridFunction(self.interpolator.grid, function.impl)
            if self.ensure_lincomb:
                return LincombDuneGridFunction([function,], [1,])
            else:
                return DuneGridFunction(function)

        @match_generic(lambda pymor_function: _is_convertible_to_dune_function(pymor_function),
                       'convertible with to_dune_function')
        def action_to_dune_function_convertible_function(self, pymor_function):
            assert pymor_function.dim_domain == self.interpolator.grid.dimension
            dune_function = to_dune_function(pymor_function, ensure_lincomb=self.ensure_lincomb)
            return self.apply(dune_function)

        @match_class(LincombFunction)
        def action_LincombFunction(self, pymor_function):
            if pymor_function.parametric and self.mu is not None:
                raise RuleNotMatchingError('Delegate to directly interpolate this as a Function with fixed mu.')
            elif pymor_function.parametric:
                assert not any([func.parametric for func in pymor_function.functions])  # does not work without a mu
            # we know that LincombFunction is never nested, so we do not call self.apply() here in case
            # self.ensure_lincomb == True to avoid nested lists
            dune_functions = [to_dune_grid_function(
                func, dune_interpolator=self.interpolator, mu=self.mu, ensure_lincomb=False)
                              for func in pymor_function.functions]
            return LincombDuneGridFunction([func.impl for func in dune_functions], pymor_function.coefficients)

        @match_class(Function)
        def action_Function(self, pymor_function):
            assert pymor_function.dim_domain == self.interpolator.grid.dimension
            if pymor_function.parametric:
                assert self.mu is not None
            dune_function = self.interpolator.interpolate(pymor_function, mu=self.mu)
            if self.ensure_lincomb:
                return LincombDuneGridFunction([dune_function,], [1,])
            else:
                return DuneGridFunction(dune_function)


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

        @match_generic(lambda function: DuneFunction.is_base_of(function),
                       'function from dune.xt.functions')
        def action_dune_xt_functions_function(self, function):
            if self.ensure_lincomb:
                return LincombDuneFunction([function,], [1,])
            else:
                return DuneFunction(function)

        @match_class(LincombDuneFunction)
        def action_LincombDuneFunction(self, function):
            return function

        @match_class(DuneFunction)
        def action_DuneFunction(self, function):
            if self.ensure_lincomb:
                return LincombDuneFunction([function.impl,], [1,])
            else:
                return function

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
            dune_function = DuneXTConstantFunction(
                    dim_domain=dim_domain, dim_range=dim_range, value=value, name=pymor_function.name)
            if self.ensure_lincomb:
                return LincombDuneFunction([dune_function,], [1,])
            else:
                return DuneFunction(dune_function)

        @match_class(ProductFunction)
        def action_ProductFunction(self, pymor_function):
            if pymor_function.parametric:
                raise RuleNotMatchingError('We cannot treat products of parametric functions!')
            # not calling self.apply here in case ensure_lincomb == True
            dune_functions = [to_dune_function(func) for func in pymor_function.functions]
            dune_function = dune_functions[0].impl
            for ii in range(1, len(dune_functions)):
                dune_function = dune_function*dune_functions[ii].impl
            if self.ensure_lincomb:
                return LincombDuneFunction([dune_function,], [1,])
            else:
                return DuneFunction(dune_function)

        @match_class(LincombFunction)
        def action_LincombFunction(self, pymor_function):
            for func in pymor_function.functions:
                if func.parametric:
                    raise RuleNotMatchingError('We cannot treat linear combinations of parametric functions!')
            # we know that LincombFunction is never nested, so we do not call self.apply() here in case
            # self.ensure_lincomb == True to avoid nested lists
            dune_functions = [to_dune_function(func, ensure_lincomb=False) for func in pymor_function.functions]
            return LincombDuneFunction([func.impl for func in dune_functions], pymor_function.coefficients)
