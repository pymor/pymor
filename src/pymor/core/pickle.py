# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from functools import partial


try:
    import cPickle as pickle
except ImportError:
    import pickle as pickle


dump = partial(pickle.dump, protocol=-1)
dumps = partial(pickle.dumps, protocol=-1)
load = pickle.load
loads = pickle.loads
PicklingError = pickle.PicklingError
