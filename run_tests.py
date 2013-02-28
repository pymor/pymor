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
    
    def __init__(self,*args, **kwargs):
        super(TestDiscoverySelector, self).__init__(*args, **kwargs)
        self._skip_grid = 'PYMOR_NO_GRIDTESTS' in os.environ
        
    def wantDirectory(self, dirname):
        return 'src' in dirname

    def wantFile(self, filename):
        parts = os.path.split(filename)
        if self._skip_grid and 'grid' in filename:
            return False
        return filename.endswith('.py') and 'pymortests' in filename

    def wantModule(self, module):
        parts = module.__name__.split('.')
        
        ret = 'pymortests' in parts or 'pymor' in parts  
        if ret:
            logging.getLogger('TEST').critical(module.__name__)
        return ret

if __name__ == '__main__':
    pl = ('tissue', 'coverage')
    manager = nose.plugins.manager.DefaultPluginManager()
    cfg = nose.config.Config(files=['setup.cfg'], plugins=manager)
    selector = TestDiscoverySelector(config=cfg)
    loader = nose.loader.TestLoader(config=cfg, selector=selector)

    #this runs the tissue plugin on source, but no tests
    tissuerun = nose.core.run(argv=[__file__, '-vv'].extend(sys.argv[1:]), module='pymortests')
    #this runs all tests as discovered, but not the tissue plugin
    restrun = nose.core.run(argv=[__file__, '-vv'].extend(sys.argv[1:]), module='pymortests', testLoader=loader)
    #no frigging clue why