# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function
import os
import tempfile
import pytest

from pymor.core.interfaces import (abstractstaticmethod, abstractclassmethod)
from pymor.core import exceptions
from pymor.core import decorators
from pymortests.base import TestInterface, runmodule, SubclassForImplemetorsOf
from pymortests.core.dummies import *   # NOQA
from pymor.grids import RectGrid
from pymor.tools import timing
import pymor.core


class TestTiming(TestInterface):

    def testTimingContext(self):
        with timing.Timer('busywait', self.logger.info):
            timing.busywait(1000)

    @timing.Timer('busywait_decorator', TestInterface.logger.info)
    def wait(self):
        timing.busywait(1000)

    def testTimingDecorator(self):
        self.wait()

    def testTiming(self):
        timer = timing.Timer('busywait', self.logger.info)
        timer.start()
        timing.busywait(1000)
        timer.stop()
        self.logger.info('plain timing took %s seconds', timer.dt)


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
        self.assertEqual(imps, StupidInterface.implementor_names(), '')
        self.assertEqual(imps + ['DocImplementer'], StupidInterface.implementor_names(True), '')
        self.assertEqual(['AverageImplementer'], BrilliantInterface.implementor_names(), '')

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
            def abstract_class_method(cls):
                return cls.__name__

            def abstract_static_method():
                return 0

        with pytest.raises(TypeError):
            FailImplementer()
        with pytest.raises(TypeError):
            ClassImplementer()
        with pytest.raises(TypeError):
            StaticImplementer()
        inst = CompleteImplementer()
        self.assertEqual(inst.abstract_class_method(), 'CompleteImplementer')
        self.assertEqual(inst.abstract_static_method(), 0)

    def testPickling(self):
        def picklme(obj, attribute_name):
            with tempfile.NamedTemporaryFile(mode='wb', delete=False) as dump_file:
                if hasattr(obj, 'lock'):
                    obj.lock(False)
                obj.some_attribute = 4
                pymor.core.dump(obj, dump_file)
                dump_file.close()
                f = open(dump_file.name, 'rb')
                unpickled = pymor.core.load(f)
                assert getattr(obj, attribute_name) == getattr(unpickled, attribute_name)
                os.unlink(dump_file.name)
        picklme(AverageImplementer(), 'some_attribute')
        picklme(CacheImplementer(), 'some_attribute')
        picklme(RectGrid(num_intervals=(4, 4)), 'num_intervals')

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
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))
            self.assertTrue("DeprecationWarning" in str(w[-1].message))

    def testVersion(self):
        self.assertGreater(pymor.VERSION, pymor.NO_VERSION)
        self.assertIsInstance(pymor.VERSION, pymor.Version)


@SubclassForImplemetorsOf(BasicInterface)
class WithcopyInterface(TestInterface):

    def test_with_(self):
        self_type = self.Type
        try:
            obj = self_type()
        except TypeError as e:
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
