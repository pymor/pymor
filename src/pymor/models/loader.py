# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import yaml

from pymor.models.basic import StationaryModel

# from pymor.parameters.base import Parameters

from pymor.tools.io.matrices import load_matrix

"""Loads models from yml-based configuration files.

The configuration file should be of form:

type: <subclass of pymor.models>
<operator>
parameters:
  <arguments>

With <arguments> being one or many statements of the form
<argument>:
  <value>
OR
<argument>:
  - <value>
  - <value>
  - ...

With <operator> being one or many statements of the form
operator:
  type: <subclass of pymor.operators>
  <operator>
  <coefficients>
OR
<name>:
  <matrix>
OR
<name>:
  - <matrix>
  - <matrix>
"""


def load_module(path):
    """Creates model from configuration file.

    Args:
        path (string): Path to the configuration file relative to pymor_source.

    Returns:
        model: model as loaded from the file.
    """
    with open(path, "r") as stream:
        try:
            load_dict = yaml.safe_load(stream)
            print(load_dict)
        except yaml.YAMLError as exc:
            print(exc)

    # construct the parameter object first
    parameters = None
    if ("parameters" in load_dict.keys()):
        # parameters = Parameters()
        NotImplemented

    # parse loaded dict, combine it with parameters,
    # construct the objects and write it in model_parameters dict
    model_parameters = {}
    for key, value in load_dict.items():
        if key == "type":
            model_parameters["type"] = value
        # identify operators
        elif key == "operator" or key == "products" or ".mat" in value:
            model_parameters[key] = construct_operator(parameters, **value)
        elif key == "parameters":
            continue
        else:
            raise ValueError(f"The key {key} given is not permitted.")
    # construct the model
    return construct_model(**model_parameters)


def construct_operator(parameters, param_dict):
    """Constructs the operator specified in 'type' of the param_dict with the other key-value-pairs.

    The possible operators are all subclasses of pymor.operators.
    """
    operator_parameters = {}
    if param_dict["type"] == "LincombOperator":
        # parse specific parameters of operator, should be recursive
        for key, value in param_dict.items():
            if "diffusion" in value:
                operator_parameters["diffusion"] = parameters["diffusion"]
            # and all the other parts of the parameters object
            elif ".mat" in value:
                operator_parameters[key] = load_matrix(value)

        # construct operator
    # elif all the other subclasses of operator
    else:
        raise ValueError("The type of operator given is not permitted.")


def construct_model(model_parameters):
    """Constructs the model.model.

    As specified in 'type' of the model_parameters with the other key-value-pairs.
    The possible models are all subclasses of pymor.models.
    """
    if model_parameters["type"] == "StationaryModel":
        return StationaryModel(**model_parameters)
    # elif all the other subclasses of model
    else:
        raise ValueError("The type of model given is not permitted.")
