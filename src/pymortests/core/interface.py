# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import importlib
import pytest
import numpy as np

from pymor.core.interfaces import (ImmutableInterface, abstractstaticmethod, abstractclassmethod)
from pymor.core import exceptions
from pymortests.base import TestInterface, runmodule, subclassForImplemetorsOf
from pymortests.core.dummies import *   # NOQA
from pymor.grids.rect import RectGrid
from pymor.grids.tria import TriaGrid
from pymor.tools import timing
import pymor.core


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


class WithcopyInterface(TestInterface):

    def test_with_(self):
        self_type = self.Type
        try:
            obj = self_type()
        except Exception as e:
            self.logger.debug(f'WithcopyInterface: Not testing {self_type} because its init failed: {str(e)}')
            return

        try:
            new = obj.with_()
            assert isinstance(new, self_type)
        except exceptions.ConstError:
            pass


def test_withcopy_implementors():
    for TestType in subclassForImplemetorsOf(ImmutableInterface, WithcopyInterface):
        TestType().test_with_()


def test_with_newtype():
    g = RectGrid(num_intervals=(99, 99))
    g2 = g.with_(new_type=TriaGrid, domain=([0, 0], [2, 2]))

    assert isinstance(g2, TriaGrid)
    assert g2.num_intervals == (99, 99)
    assert np.all(g2.domain == ([0, 0], [2, 2]))


if __name__ == "__main__":
    runmodule(filename=__file__)
