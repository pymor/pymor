# pymor (http://www.pymor.org)
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.interfaces import BasicInterface
from pymor.core.logger import getLogger
from pymor.core.defaults import defaults

# Set default log levels
# Log levels propagte downwards, i.e. if the level of "getLogger('a.b.c')" is not set
# the log level of "getLogger('a.b')" is assumed

getLogger('pymor').setLevel('WARN')
getLogger('pymor.core').setLevel('WARN')

try:
    import cPickle as pickle
except:
    import pickle

from functools import partial

dump = partial(pickle.dump, protocol=-1)
dumps = partial(pickle.dumps, protocol=-1)
load = pickle.load
loads = pickle.loads
