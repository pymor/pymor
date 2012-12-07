import nose
import os
import unittest
import logging
import sys

MYDIR = os.path.dirname(__file__)
sys.path.append(os.path.join(MYDIR,'src'))

class TestProgram(nose.core.TestProgram):
    pass

class TestDiscoverySelector(nose.selector.Selector):
    def wantDirectory(self, dirname):
        return 'src' in dirname

    def wantFile(self, filename):
        parts = os.path.split(filename)
        return filename.endswith('.py') and 'src/pymor/tests' in filename

    def wantModule(self, module):
        parts = module.__name__.split('.')
        return 'tests' in parts

if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    cfg = nose.config.Config(files=['setup.cfg'])
    selector = TestDiscoverySelector(cfg)
    loader = nose.loader.TestLoader()
    loader.selector = selector 
    TestProgram(argv=[__file__, '-vv' ], testLoader=loader, module='pymor.tests')