# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.base import BasicObject


class BaseEstimator(BasicObject):
    """Base-class for scikit-learn-style estimators.

    This class provides the `get_params` and `set_params` methods required
    to use the hyperparameter tuning methods of scikit-learn.
    In order to make use of the functionality, an inheriting regressor only
    needs to provide a tuple of init-parameters as strings (the `_params`
    class-attribute). If the regressor also contains another object with
    hyperparameters that might be adjusted, the `_nested_object` string
    referring to that attribute can be set. Special parameter handling can
    further be implemented by overriding the `_get_extra_params`
    and `_set_extra_param` methods, respectively.
    This class cannot be used directly but serves as a base class for
    other regressors.
    """

    _params = tuple()
    _nested_object = None

    def get_params(self, deep=True):
        """Returns a dict of the init-parameters of the estimator, together with their values.

        The argument `deep=True` is required to match the scikit-learn interface.

        Parameters
        ----------
        deep
            If `True`, the parameters for this estimator and for the nested object
            (kernel, neural network, etc.) will be returned.

        Returns
        -------
        A dictionary of parameters and respective values of the estimator.
        """
        params = {name: getattr(self, name) for name in self._params}
        if deep and self._nested_object is not None:
            nested = getattr(self, self._nested_object)
            if hasattr(nested, 'get_params'):
                nested_object_params = nested.get_params(deep=True)
                for name, value in nested_object_params.items():
                    params[f'{self._nested_object}__{name}'] = value
        return params | self._get_extra_params()

    def _get_extra_params(self):
        """Allows to handle special parameters of the regressor in `get_params`."""
        return dict()

    def set_params(self, **params):
        """Set the parameters of the estimator and the nested object.

        Supports nested parameter setting for the nested object using
        the value of ``self._nested_object`` as prefix.

        Parameters
        ----------
        params
            Estimator parameters to set.

        Returns
        -------
        An instance of the estimator with the new parameters.
        """
        if self._nested_object is not None:
            nested_object_params = {}
            prefix = self._nested_object + '__'

        for key, value in params.items():
            if self._nested_object is not None and key.startswith(prefix):
                nested_object_params[key.removeprefix(prefix)] = value
            elif key in self._params:
                setattr(self, key, value)
            else:
                self._set_extra_param(key, value)

        if self._nested_object is not None and nested_object_params:
            nested = getattr(self, self._nested_object)
            if not hasattr(nested, 'set_params'):
                raise ValueError('Nested object does not support parameter setting')
            nested.set_params(**nested_object_params)

        return self

    def _set_extra_param(self, key, param):
        """Allows to handle special parameters of the regressor in `set_params`."""
        raise NotImplementedError('`_set_extra_param` not available for this regressor')
