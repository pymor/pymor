# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import os
import tempfile

import pytest

import pymor.core
from pymor.core import exceptions
from pymor.core.dynamic import *  # NOQA
from pymor.core.interfaces import ImmutableInterface, abstractclassmethod, abstractstaticmethod
from pymor.grids.rect import RectGrid
from pymor.tools import timing
from pymortests.base import SubclassForImplemetorsOf, TestInterface, runmodule
from pymortests.core.dummies import *  # NOQA


class Test_Interface(TestInterface):

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

    def testVersion(self):
        assert 'unknown' not in pymor.__version__
        assert '?' not in pymor.__version__


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


if __name__ == "__main__":
    runmodule(filename=__file__)
