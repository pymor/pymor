# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from pymor.core.interfaces import BasicInterface
from pymortests.base import (TestInterface, _load_all)

import pytest

def _type_list(interface_type):
    try:
        _load_all()
    except ImportError:
        pass
    return [T for T in interface_type.implementors(True) if not (T.has_interface_name() or 
                                                                 issubclass(T, TestInterface))] 

@pytest.fixture(params=_type_list(BasicInterface))
def basicinterface_subclasses(request):
    return request.param
    

if __name__ == '__main__':
    pass
