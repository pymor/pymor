# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import warnings

warn = warnings.warn


class ConstError(Exception):
    """I get thrown when you try to add a new member to
    a locked class instance"""
    pass


class AccuracyError(Exception):
    """Is raised if the result of a computation is inaccurate"""


class ExtensionError(Exception):
    """Is raised if a (basis) extension algorithm fails.

    This will mostly happen during a basis extension when the new snapshot is already
    in the span of the basis.
    """


class InversionError(Exception):
    """Is raised if an operator inversion algorithm fails."""


class NewtonError(Exception):
    """Is raised if the Newton algorithm fails to converge."""


class SIDGenerationError(Exception):
    """Is raised when generate_sid fails."""


class GmshError(Exception):
    """Is raised when a Gmsh related error occurs."""


class ImageCollectionError(Exception):
    """Is raised when a pymor.algorithms.image.estimate_image fails for given operator."""
    def __init__(self, op):
        super(ImageCollectionError, self).__init__('Cannot estimage image for {}'.format(op))
        self.op = op


class PySideMissing(ImportError):
    """Raise me where having importable PySide is non-optional"""
    def __init__(self, msg=None):
        msg = msg or 'cannot visualize: import of PySide failed'
        super(PySideMissing, self).__init__(msg)
