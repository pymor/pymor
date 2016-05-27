# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import os
import tempfile
import pytest

from pymor.core.interfaces import (ImmutableInterface, abstractstaticmethod, abstractclassmethod)
from pymor.core import exceptions
from pymor.core import decorators
from pymortests.base import TestInterface, runmodule, SubclassForImplemetorsOf
from pymortests.core.dummies import *   # NOQA
from pymor.grids.rect import RectGrid
from pymor.tools import timing
import pymor.core

class Test_Interface(TestInterface):

    def testLock(self):
        b = AverageImplementer()
        b.level = 43
        b.lock()
        assert b.locked
        with pytest.raises(exceptions.ConstError):
            b.new = 42
        with pytest.raises(exceptions.ConstError):
            b.level = 0
        b.lock(False)
        b.level = 1
        b.new = 43
        assert hasattr(b, 'new')
        assert b.level == 1
        assert b.new == 43

    def testImplementorlist(self):
        imps = ['StupidImplementer', 'AverageImplementer', 'FailImplementer']
        assert imps == StupidInterface.implementor_names()
        assert imps + ['DocImplementer'] == StupidInterface.implementor_names(True)
        assert ['AverageImplementer'] == BrilliantInterface.implementor_names()

    def testAbstractMethods(self):
        class ClassImplementer(BasicInterface):

            @abstractclassmethod
            def abstract_class_method(cls):
                pass

        class StaticImplementer(BasicInterface):

            @abstractstaticmethod
            def abstract_static_method():
                pass

        class CompleteImplementer(ClassImplementer, StaticImplementer):
            @classmethod
            def abstract_class_method(cls):
                return cls.__name__

            @staticmethod
            def abstract_static_method():
                return 0

        with pytest.raises(TypeError):
            FailImplementer()
        with pytest.raises(TypeError):
            ClassImplementer()
        with pytest.raises(TypeError):
            StaticImplementer()
        inst = CompleteImplementer()
        assert inst.abstract_class_method() == 'CompleteImplementer'
        assert inst.abstract_static_method() == 0

    def testDeprecated(self):
        @decorators.Deprecated('use other stuff instead')
        def deprecated_function():
            pass
        # Cause all warnings to always be triggered.
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Trigger a warning.
            deprecated_function()
            # Verify some things
            assert len(w) == 1
            assert issubclass(w[-1].category, DeprecationWarning)
            assert "DeprecationWarning" in str(w[-1].message)

    def testVersion(self):
        assert pymor.VERSION > pymor.NO_VERSION
        assert isinstance(pymor.VERSION, pymor.Version)


@SubclassForImplemetorsOf(ImmutableInterface)
class WithcopyInterface(TestInterface):

    def test_with_(self):
        self_type = self.Type
        try:
            obj = self_type()
        except Exception as e:
            self.logger.debug('WithcopyInterface: Not testing {} because its init failed: {}'.format(self_type, str(e)))
            return

        try:
            new = obj.with_()
            assert isinstance(new, self_type)
        except exceptions.ConstError:
            pass

# this needs to go into every module that wants to use dynamically generated types, ie. testcases, below the test code
from pymor.core.dynamic import *   # NOQA

if __name__ == "__main__":
    runmodule(filename=__file__)
