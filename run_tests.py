import nose
import os
import unittest
import logging
import sys

MYDIR = os.path.dirname(__file__)
sys.path.append(os.path.join(MYDIR,'src'))

from pymortests.base import (TestDiscoverySelector, )


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