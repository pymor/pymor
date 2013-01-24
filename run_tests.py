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
        return filename.endswith('.py') and 'pymortests' in filename

    def wantModule(self, module):
        parts = module.__name__.split('.')
        
        ret = 'pymortests' in parts
        if ret:
            logging.getLogger('TEST').critical(module.__name__)
        return ret

if __name__ == '__main__':
    cfg = nose.config.Config(files=['setup.cfg'])
    selector = TestDiscoverySelector(cfg)
    loader = nose.loader.TestLoader()
    loader.selector = selector 
    TestProgram(argv=[__file__, '-vv' ].extend(sys.argv[1:]), testLoader=loader, module='pymortests')