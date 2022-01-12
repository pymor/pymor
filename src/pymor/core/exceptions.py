# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import warnings

warn = warnings.warn


class ConstError(Exception):
    """I get thrown when you try to add a new member to a locked class instance."""


class AccuracyError(Exception):
    """Is raised if the result of a computation is inaccurate."""


class ExtensionError(Exception):
    """Is raised if a (basis) extension algorithm fails.

    This will mostly happen during a basis extension when the new snapshot is already
    in the span of the basis.
    """


class InversionError(Exception):
    """Is raised if an operator inversion algorithm fails."""


class LinAlgError(Exception):
    """Is raised if a linear algebra operation fails."""


class NewtonError(Exception):
    """Is raised if the Newton algorithm fails to converge."""


class CacheKeyGenerationError(Exception):
    """Is raised when cache key generation fails due to unsupported arguments."""


class GmshMissing(Exception):
    """Is raised when a Gmsh is not found."""


class MeshioMissing(Exception):
    """Is raised when meshio is not available."""


class ImageCollectionError(Exception):
    """Is raised when a pymor.algorithms.image.estimate_image fails for given operator."""

    def __init__(self, op):
        super().__init__(f'Cannot estimate image for {op}')
        self.op = op


class NeuralNetworkTrainingFailed(Exception):
    """Is raised when training of a neural network fails."""


class DependencyMissing(ImportError):
    """Raised when optional packages are required but are not installed."""

    def __init__(self, dependency, msg=None):
        self.dependency = dependency
        super().__init__(msg or f'optional dependency {dependency} required')


class QtMissing(DependencyMissing):
    """Raise me where having importable Qt bindings is non-optional"""

    def __init__(self, msg=None):
        msg = msg or 'cannot visualize: import of Qt bindings failed'
        super().__init__('QT', msg)


class TorchMissing(DependencyMissing):
    """Raise me where having importable torch version is non-optional"""

    def __init__(self, msg=None):
        msg = msg or 'cannot use neural networks: import of torch failed'
        super().__init__('TORCH', msg)


class RuleNotMatchingError(NotImplementedError):
    pass


class NoMatchingRuleError(NotImplementedError):
    def __init__(self, obj):
        super().__init__(f'No rule could be applied to {obj}')
        self.obj = obj


class IOLibsMissing(ImportError):
    def __init__(self, msg=None):
        msg = msg or 'meshio, pyevtk, xmljson and lxml are needed for full file I/O functionality'
        super().__init__(msg)


class UnpicklableError(Exception):
    def __init__(self, cls):
        self.cls = cls

    def __str__(self):
        return f'{self.cls} cannot be pickled.'
