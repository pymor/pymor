pyMor - Model order reduction in Python
=======================================

Requirements
------------

-  distribute, scipy, matplotlib, numpy, pycontracts, docopt
-  optional: sympy, sphinx

Debugging
---------

-  You can globally disable caching by having PYMOR\_CACHE\_DISABLE=1 in
   the process' environment

Tests
-----

You'll need mock, nose-cov, nose, nosehtmloutput, nose-progressive and
tissue installed to run 'make test'. Having PYMOR\_NO\_GRIDTESTS=1 in
the process' environment disables all, expensive grid testing.
