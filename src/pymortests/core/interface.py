from __future__ import absolute_import, division, print_function
import pickle 
import os
import tempfile

from pymor.core.interfaces import (BasicInterface, abstractmethod, abstractstaticmethod, 
                                   abstractclassmethod)
from pymor.core import exceptions
from pymor.core import timing
from pymortests.base import TestBase, runmodule
from pymortests.core.dummies import *
from pymor.grids import RectGrid

class TimingTest(TestBase):
    
    def testTimingContext(self):
        with timing.Timer('busywait',self.logger.info) as timer:
            timing.busywait(1000)
            
    @timing.Timer('busywait_decorator', TestBase.logger.info)
    def wait(self):
        timing.busywait(1000)
            
    def testTimingDecorator(self):        
        self.wait()
        
    def testTiming(self):
        timer = timing.Timer('busywait',self.logger.info)
        timer.start()
        timing.busywait(1000)
        timer.stop()
        self.logger.info('plain timing took %s seconds', timer.dt)
        
class InterfaceTest(TestBase):

    def testFreeze(self):
        b = AverageImplementer()
        b.level = 43
        b.lock()
        b.level = 41
        with self.assertRaises(exceptions.ConstError):
            b.new = 42
        b.freeze()
        with self.assertRaises(exceptions.ConstError):
            b.level = 0
        b.freeze(False)
        b.level = 0
        b.lock(False)
        b.level = 0
        
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
            
        with self.assertRaisesRegexp(TypeError, "Can't instantiate abstract class.*"):
            inst = FailImplementer()
        with self.assertRaisesRegexp(TypeError, "Can't instantiate abstract class.*"):
            inst = ClassImplementer()
        with self.assertRaisesRegexp(TypeError, "Can't instantiate abstract class.*"):
            inst = StaticImplementer()
        inst = CompleteImplementer()
        self.assertEqual(inst.abstract_class_method(), 'CompleteImplementer')
        self.assertEqual(inst.abstract_static_method(), 0)        
        
    def testPickling(self):
        def picklme(obj,attribute_name):
            with tempfile.NamedTemporaryFile(delete=False) as dump_file:
                obj.some_attribute = 4
                pickle.dump(obj, dump_file)
                dump_file.close()
                f = open(dump_file.name, 'rb')
                unpickled = pickle.load(f)
                self.assert_(getattr(obj, attribute_name) == getattr(unpickled, attribute_name))
                os.unlink(dump_file.name)
        picklme(AverageImplementer(), 'some_attribute')
        picklme(CacheImplementer(), 'some_attribute')
        
        
if __name__ == "__main__":
    runmodule(name='pymortests.core.interface')
       
        