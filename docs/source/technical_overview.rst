.. _technical_overview:

******************
Technical Overview
******************

Three Central Classes
---------------------

From a bird's eye perspective, pyMOR is a collection of generic algorithms
operating on objects of the following types:

|VectorArrays|
    Vector arrays are ordered collections of vectors. Each vector of the array
    must be of the same |dimension|. Subsets of vectors can be
    |copied| to a new array, |appended| to an existing array, |removed| from the
    array or |replaced| by vectors of a different array.
    Basic linear algebra operations can be performed on the vectors of the
    array: vectors can be |scaled| inplace, the BLAS |axpy| operation is
    supported and |scalar products| between vectors can be formed. Linear
    combinations of vectors can be formed using the |lincomb| method. Moreover,
    various norms can be computed and selected |components| of the vectors can
    be extracted for :mod:`empirical interpolation <pymor.algorithms.ei>`.

    Each of these methods takes optional `ind` parameters to specify the subset
    of vectors on which to operate. If the parameter is not specified, the whole
    array is selected for the operation. 

    New vector arrays can be created using the |empty| and |zeros| method. As a
    convenience, many of Python's math special methods are implemented in terms
    of the interface methods.

    Note that there is not the notion of a single vector in pyMOR. The main
    reason for this design choice is to allow vectorization of
    operations on |NumpyVectorArrays| which internally store the
    vectors as two-dimensional |NumPy| arrays. As an example, the application of
    a linear matrix based operator to an array via the |apply| method boils down
    to a call to |NumPy|'s optimized :meth:`~numpy.ndarray.dot` method. As all
    reduced computations in pyMOR are based on |NumPy| arrays, this is important
    for the performance of the reduced schemes. Given vector arrays in pyMOR
    were lists of vector objects, the above matrix-matrix multiplication would
    have to be replaced by a loop of matrix-vector multiplications. However,
    to facilitate implementation of new vector array types, a Python list based
    array is provided with |ListVectorArray|.

    .. |apply|            replace:: :meth:`~pymor.operators.interfaces.OperatorInterface.apply`
    .. |appended|         replace:: :meth:`appended <pymor.la.interfaces.VectorArrayInterface.append>`
    .. |axpy|             replace:: :meth:`~pymor.la.interfaces.VectorArrayInterface.axpy`
    .. |components|       replace:: :meth:`~pymor.la.interfaces.VectorArrayInterface.components`
    .. |copied|           replace:: :meth:`copied <pymor.la.interfaces.VectorArrayInterface.copy>`
    .. |dimension|        replace:: :attr:`dimension <pymor.la.interfaces.VectorArrayInterface.dim>`
    .. |empty|            replace:: :meth:`~pymor.la.interfaces.VectorArrayInterface.empty`
    .. |lincomb|          replace:: :meth:`~pymor.la.interfaces.VectorArrayInterface.lincomb`
    .. |removed|          replace:: :meth:`deleted <pymor.la.interfaces.VectorArrayInterface.remove>`
    .. |replaced|         replace:: :meth:`replaced <pymor.la.interfaces.VectorArrayInterface.replace>`
    .. |scalar products|  replace:: :meth:`scalar products <pymor.la.interfaces.VectorArrayInterface.dot>`
    .. |scaled|           replace:: :meth:`scaled <pymor.la.interfaces.VectorArrayInterface.scal>`
    .. |zeros|            replace:: :meth:`~pymor.la.interfaces.VectorArrayInterface.zeros`

|Operators|
    The main property of operators in pyMOR is that they can be |applied| to
    |VectorArrays| resulting in a new |VectorArray|. For this operation to be
    allowed, the operator's |type_source| must be a super class of the given
    array and its |dim_source| must agree with the array's dimension. The
    result will be a vector array of type |type_range| and dimension
    |dim_range|. An operator can be |linear| or not.
    The |apply_inverse| method provides an interface for (linear) solvers 
    
    Operators in pyMOR are also used to represent bilinear forms
    via the |apply2| method. A functional in pyMOR is simply an operator with
    |NumpyVectorArray| as |type_range| and a |dim_range| of 1. Dually, a
    vector-like operator is an operator with a |dim_source| of 1 and
    |NumpyVectorArray| as |type_source|. Such vector-like operators are used in
    pyMOR to represent |Parameter| dependent vectors such as the initial data of
    an |InstationaryDiscretization|. For linear functionals and vector-like
    operators, the |as_vector| method can be called to obtain a vector
    representation of the operator as a |VectorArray| of length 1.

    Linear combinations of operators can be formed using the |op.lincomb| method.
    While this method can be used for arbitrary types of operators,
    specializations of this method ensure that linear combinations of operators
    represented by a matrix lead to a new operator holding the linear
    combination of the matrices. The same holds true for the |projected| method,
    which is used to perform the reduced basis projection of a given operator:
    while each operator in pyMOR can be |projected|, specializations ensure
    that, if possible, the projected operator will no longer depend on
    high-dimensional data. Projected operators always have |NumpyVectorArray| as
    |type_source| and |type_range|.

    Default implementations for various types of |Operators| as well as
    |NumPy|-based operators can be found in :mod:`pymor.operators.basic`.
    Several methods for constructing new operators from existing ones are
    contained in :mod:`pymor.operators.constructions`.

    .. |applied|          replace:: :meth:`applied <pymor.operators.interfaces.OperatorsInterface.apply>`
    .. |apply2|           replace:: :meth:`~pymor.operators.interfaces.OperatorInterface.apply2`
    .. |apply_inverse|    replace:: :meth:`~pymor.operators.interfaces.OperatorInterface.apply_inverse`
    .. |as_vector|        replace:: :meth:`~pymor.operators.interfaces.OperatorInterface.as_vector`
    .. |dim_range|        replace:: :attr:`~pymor.operators.interfaces.OperatorInterface.dim_range`
    .. |dim_source|       replace:: :attr:`~pymor.operators.interfaces.OperatorInterface.dim_source`
    .. |linear|           replace:: :attr:`~pymor.operators.interfaces.OperatorInterface.linear`
    .. |op.lincomb|       replace:: :meth:`~pymor.operators.interfaces.OperatorInterface.lincomb`
    .. |projected|        replace:: :meth:`~pymor.operators.interfaces.OperatorInterface.projected`
    .. |type_range|       replace:: :attr:`~pymor.operators.interfaces.OperatorInterface.type_range`
    .. |type_source|      replace:: :attr:`~pymor.operators.interfaces.OperatorInterface.type_source`

|Discretizations|
    Discretizations in pyMOR encode the mathematical structure of a given
    discrete problem acting as container classes for operators. Each
    discretization object has |operators|, |functionals|, |vector_operators| and
    |products| dictionaries holding the |Operators| which appear in the
    formulation of the discrete problem. The keys in these dictionaries describe
    the role of the respective operator in the discrete problem.

    Apart from describing the discrete problem, discretizations also implement
    algorithms for |solving| the given problem, returning |VectorArrays| of
    type |type_solution| and dimension |dim_solution| holding the solution
    vector or trajectory of solution vectors. The solution is usually |cached|,
    s.t. subsequent solving of the problem for the same parameters reduces to
    looking up the solution in pyMOR's cache.

    While special discretization classes may be implemented which make use of
    the specific types of operators they contain (e.g. using some external
    high-dimensional solver for the problem), it is generally favourable to
    implement the solution algorithms only through the interfaces provided by
    the operators contained in the discretization, as this allows to use the
    same discretization class to solve high-dimensional and reduced problems.
    This has been done for the simple stationary and instationary
    discretizations found in :mod:`pymor.discretizations.basic`.

    Discretizations can also implement |estimate| and |visualize| methods to
    estimate the discretization error of a computed solution and create graphic
    representations of |VectorArrays| of |type_solution|.

    .. |dim_solution|     replace:: :attr:`~pymor.discretizations.interfaces.DiscretizationInterface.dim_solution`
    .. |estimate|         replace:: :meth:`~pymor.discretizations.interfaces.DiscretizationInterface.estimate`
    .. |functionals|      replace:: :attr:`~pymor.discretizations.interfaces.DiscretizationInterface.functionals`
    .. |operators|        replace:: :attr:`~pymor.discretizations.interfaces.DiscretizationInterface.operators`
    .. |products|         replace:: :attr:`~pymor.discretizations.interfaces.DiscretizationInterface.products`
    .. |solving|          replace:: :meth:`solving <pymor.discretizations.interfaces.DiscretizationInterface.solve>`
    .. |solve|            replace:: :meth:`~pymor.discretizations.interfaces.DiscretizationInterface.solve`
    .. |type_solution|    replace:: :attr:`~pymor.discretizations.interfaces.DiscretizationInterface.type_solution`
    .. |vector_operators| replace:: :attr:`~pymor.discretizations.interfaces.DiscretizationInterface.vector_operators`
    .. |visualize|        replace:: :meth:`~pymor.discretizations.interfaces.DiscretizationInterface.visualize`
    .. |cached|           replace:: :mod:`cached <pymor.core.cache>`


Base Classes
------------

While |VectorArrays| are mutable objects, both |Operators| and |Discretizations|
are immutable in pyMOR: the application of an |Operator| to the same
|VectorArray| will always lead to the same result, solving a |Discretization|
for the same parameter will always produce the same solution array. This has two
main benefits:

1. If multiple objects/algorithms hold references to the same
   |Operator| or |Discretization|, none of the objects has to worry that the
   referenced object changes without their knowledge.
2. The state of an immutable object is determined by the states of the objects
   that lead to the creation of the object. This property is used in pyMOR to
   create |state ids| for immutable objects which are used as keys in pyMOR's
   |caching| backends.

A class can be made immutable in pyMOR by deriving from |ImmutableInterface|,
which ensures that write access to the object's attributes is prohibited after
`__init__` has been executed. However, note that changes to private attributes
(attributes whose name starts with `_`) are still allowed. It lies in the
implementors responsibility to ensure, that changes of these attributes do not
affect the outcome of calls to relevant interface methods. As an example a call
to :meth:`~pymor.core.cache.CacheableInterface.enable_caching` will set the
objects private `__cache_region` attribute, which might affect the speed of a
subsequent |solve| call, but not its result.

Of course, in many situations one may wish to change properties of an immutable
object, e.g. the number of timesteps for a given discretization. This can be
easily achieved using the `~pymor.core.interfaces.BasicInterface.with_` method
every immutable object has: a call of the form `o.with_(a=x, b=y)` will return
a copy of `o` in which the attribute `a` now has the value `x` and the
attribute `b` the value `y`. It can be generally assumed that calls to
`~pymor.core.interfaces.BasicInterface.with_` are inexpensive. The set of
allowed arguments can be found in the
:attr:`~pymor.core.interfaces.BasicInterface.with_arguments` attribute.

All immutable classes in pyMOR and most other classes derive from
|BasicInterface| which, through its meta class, provides several convenience
features for pyMOR. Most notably, every subclass of |BasicInterface| obtains its
own :attr:`~pymor.core.interfaces.BasicInterface.logger` instance with a class
specific prefix.

.. |caching|        replace:: :mod:`caching <pymor.core.cache>`
.. |state id|       replace:: :ref:`state id <state id>`
.. |state ids|      replace:: :ref:`state ids <state id>`


Creating Discretizations
------------------------

pyMOR ships a small (and still quite incomplete) framework for creating finite
element or finite volume discretizations based on the `NumPy/Scipy
<http://scipy.org>`_ software stack. To end up with an appropriate
|Discretization|, one starts with instantiating an |analytical problem|, which
describes the problem we want to discretize. |analytical problems| contain
|Functions| which define the analytical data functions associated with the
problem and a |DomainDescription| that provides a geometrical definition of the
domain the problem is posed on and associates a |BoundaryType| to each part of
its boundary.

To obtain a |Discretization| from an |analytical problem| we use a
:mod:`discretizer <pymor.discretizers>`. A discretizer will first mesh the
computational domain by feeding the |DomainDescription| into a
:mod:`domaindiscretizer <pymor.domaindiscretizers>` which will return the |Grid|
along with a |BoundaryInfo| associating boundary entities with |BoundaryTypes|.
Next, the |Grid|, |BoundaryInfo| and the various data functions of the
|analytical problem| are used to instatiate :mod:`finite element
<pymor.operators.cg>` or :mod:`finite volume <pymor.operators.fv>` operators.
Finally these operators are used to instatiate one of the provided
|Discretization| classes.

In pyMOR, |analytical problems|, |Functions|, |DomainDescriptions|,
|BoundaryInfos| and |Grids| are all immutable, ensuring that the resulting
|Discretization| receives a proper |state id| to enable persistent disk
|caching| over various runs of the applications written with pyMOR.

While pyMOR's internal discretizations are useful for getting started quickly
with model reduction experiments, pyMOR's main goal is to allow the reduction of
discretizations provided by external solvers. In order to do so, all that needs
to be done is to provide |VectorArrays|, |Operators| and |Discretizations| which
interact appropriately with the solver. pyMOR makes no assumption on how the
communication with the solver is managed. For instance, communication could take
place via a network protocol or job files.  In particular it should be stressed,
that in general no communication of high-dimensional data between the solver
and pyMOR is necessary: |VectorArrays| can merely hold handles to data in the
solver's memory or some on-disk database. Where possible, we favour, however, a
deep integration of the solver with pyMOR by linking the solver code as a Python
extension module. This allows Python to directly access the solvers data
structures which can be used to quickly add features to the high-dimensional
code without any recompilation. A minimal example for such an integration using
`pybindgen <https://code.google.com/p/pybindgen>`_ can be found in the
``src/pymordemos/minimal_cpp_demo`` directory of the pyMOR repository.
The `dune-pymor <https://github.com/pymor/dune-pymor>`_ repository contains
experimental bindings for the `DUNE <http://dune-project.org>`_ software
framework.


Parameters
----------

pyMOR classes implement dependence of a parameter by deriving from the
|Parametric| mix-in class. This class gives each instance a
:attr:`~pymor.parameters.base.Parametric.parameter_type` attribute describing the
form of |Parameters| the relevant methods of the object (`apply`, `solve`,
`evaluate`, etc.) expect. A |Parameter| in pyMOR is basically a Python
:class:`dict` with strings as keys and |NumPy arrays| as values. Each such value
is called a |Parameter| component. The |ParameterType| of a |Parameter| is
simply obtained by replacing the arrays in the |Parameter| with their shape.
I.e. a |ParameterType| specifies the names of the parameter components and their
expected shapes.

The |ParameterType| of a |Parametric| object is determined by the class
implementor during `__init__` via a call to
:meth:`~pymor.parameters.base.Parametric.build_parameter_type`, which can be
used, to infer the |ParameterType| from the |ParameterTypes| of objects the
given object depends upon. I.e. an |Operator| implementing the L2-product with
some |Function| will inherit the |ParameterType| of the |Function|.

Reading the :mod:`reference documentation <pymor.parameters.base>` on pyMOR's
parameter handling facilities is strongly advised for implementors of
|Parametric| classes.


The Reduction Process
---------------------

The reduction process in pyMOR is handled by so called :mod:`~pymor.reductors`
which take arbitrary |Discretizations| and additional data (e.g. the reduced
basis) to create reduced |Discretizations| along with reconstructor classes
which allow to transform solution vectors of the reduced |Discretization| back
to vectors of the solution space of the high-dimensional |Discretization| (e.g.
by linear combination with the reduced basis). If proper offline/online
decomposition is achieved by the reductor, the reduced |Discretization| will
store no high-dimensional data. Note that there is no inherent distinction
between low- and high-dimensional |Discretizations| in pyMOR. The only
difference lies in the different types of operators, the |Discretization|
contains.

This observation comes particularly apparent in the case of the classical
reduced basis method: the operators and functionals of a given discrete problem
are projected onto the reduced basis space whereas the structure of the problem
(i.e. the type of |Discretization| containing the operators) stays the same.
pyMOR reflects this fact by offering a generic algorithm with
:func:`~pymor.reductors.basic.reduce_generic_rb`, which can
be used to RB-project any discretization available to pyMOR. It should be noted
however, that this reductor is only able to efficiently offline/online-decompose
affinely |Parameter|-dependent linear problems. Non-linear problems or such with no
affine |Parameter| dependence require additional techniques such as
:mod:`empirical interpolation <pymor.algorithms.ei>`.

If you want to further dive into the inner workings of pyMOR, we highly
recommend to study the source code of
:func:`~pymor.reductors.basic.reduce_generic_rb` and to step through calls of
this method with a Python debugger, such as `ipdb
<https://pypi.python.org/pypi/ipdb>`_.
