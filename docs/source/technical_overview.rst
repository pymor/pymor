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
    array: vectors can be |scaled| in-place, the BLAS |axpy| operation is
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
    reason for this design choice is to take advantage of vectorized
    implementations like |NumpyVectorArray| which internally store the
    vectors as two-dimensional |NumPy| arrays. As an example, the application of
    a linear matrix based operator to an array via the |apply| method boils down
    to a call to |NumPy|'s optimized :meth:`~numpy.ndarray.dot` method. If there
    were only lists of vectors in pyMOR, the above matrix-matrix multiplication
    would have to be expressed by a loop of matrix-vector multiplications. However,
    when working with external solvers, vector arrays will often be just lists
    of vectors. For this use-case we provide |ListVectorArray|, a vector array
    based on a Python list of vectors.

    Associated to each vector array is a |VectorSpace|. A Vector space in pyMOR
    is simply the combination of a |VectorArray| class and an appropriate
    |subtype|.  For |NumpyVectorArrays|, the subtype is a single integer
    denoting the dimension of the array. Subtypes for other array classes
    could, e.g., include a socket for communication with a specific PDE solver
    instance.
    
    Two arrays in pyMOR are compatible (e.g. can be added) if they are from the
    same |VectorSpace|, i.e. they are instances of the same class and share the
    same subtype. The |VectorSpace| is also precisely the information needed to
    create new arrays of null vectors using the |make_array| class method. In
    fact |empty| and |zeros| are implemented by calling |make_array| with the
    |subtype| of the |VectorArray| instance for which they have been called.
    
    .. |apply|            replace:: :meth:`~pymor.operators.interfaces.OperatorInterface.apply`
    .. |appended|         replace:: :meth:`appended <pymor.vectorarrays.interfaces.VectorArrayInterface.append>`
    .. |axpy|             replace:: :meth:`~pymor.vectorarrays.interfaces.VectorArrayInterface.axpy`
    .. |components|       replace:: :meth:`~pymor.vectorarrays.interfaces.VectorArrayInterface.components`
    .. |copied|           replace:: :meth:`copied <pymor.vectorarrays.interfaces.VectorArrayInterface.copy>`
    .. |dimension|        replace:: :attr:`dimension <pymor.vectorarrays.interfaces.VectorArrayInterface.dim>`
    .. |empty|            replace:: :meth:`~pymor.vectorarrays.interfaces.VectorArrayInterface.empty`
    .. |lincomb|          replace:: :meth:`~pymor.vectorarrays.interfaces.VectorArrayInterface.lincomb`
    .. |make_array|       replace:: :meth:`~pymor.vectorarrays.interfaces.VectorArrayInterface.make_array`
    .. |removed|          replace:: :meth:`deleted <pymor.vectorarrays.interfaces.VectorArrayInterface.remove>`
    .. |replaced|         replace:: :meth:`replaced <pymor.vectorarrays.interfaces.VectorArrayInterface.replace>`
    .. |scalar products|  replace:: :meth:`scalar products <pymor.vectorarrays.interfaces.VectorArrayInterface.dot>`
    .. |scaled|           replace:: :meth:`scaled <pymor.vectorarrays.interfaces.VectorArrayInterface.scal>`
    .. |subtype|          replace:: :attr:`~pymor.vectorarrays.interfaces.VectorSpace.subtype`
    .. |zeros|            replace:: :meth:`~pymor.vectorarrays.interfaces.VectorArrayInterface.zeros`

|Operators|
    The main property of operators in pyMOR is that they can be |applied| to
    |VectorArrays| resulting in a new |VectorArray|. For this operation to be
    allowed, the operator's |source| |VectorSpace| must be identical with the
    |VectorSpace| of the given array. The result will be a vector array from
    the |range| space. An operator can be |linear| or not.  The |apply_inverse|
    method provides an interface for (linear) solvers.
    
    Operators in pyMOR are also used to represent bilinear forms via the
    |apply2| method. A functional in pyMOR is simply an operator with
    ``VectorSpace(NumpyVectorArray, 1)`` as |range|. Dually, a vector-like
    operator is an operator with a ``VectorSpace(NumpyVectorArray, 1)`` as
    |source|. Such vector-like operators are used in pyMOR to represent
    |Parameter| dependent vectors such as the initial data of an
    |InstationaryDiscretization|. For linear functionals and vector-like
    operators, the |as_vector| method can be called to obtain a vector
    representation of the operator as a |VectorArray| of length 1.

    Linear combinations of operators can be formed using a |LincombOperator|.
    When such a linear combination is |assembled|, |assemble_lincomb|
    is called to ensure that, for instance, linear combinations of operators
    represented by a matrix lead to a new operator holding the linear
    combination of the matrices. The |projected| method is used to perform the
    reduced basis projection of a given operator.  While each operator in pyMOR
    can be |projected|, specializations of this method ensure that, if
    possible, the projected operator will no longer depend on high-dimensional
    data.

    Default implementations for many methods of the operator interface can be
    found in |OperatorBase|. Base classes for |NumPy|-based operators can be
    found in :mod:`pymor.operators.numpy`. Several methods for constructing
    new operators from existing ones are contained in
    :mod:`pymor.operators.constructions`.

    .. |applied|           replace:: :meth:`applied <pymor.operators.interfaces.OperatorInterface.apply>`
    .. |apply2|            replace:: :meth:`~pymor.operators.interfaces.OperatorInterface.apply2`
    .. |apply_inverse|     replace:: :meth:`~pymor.operators.interfaces.OperatorInterface.apply_inverse`
    .. |assembled|         replace:: :meth:`assembled <pymor.operators.interfaces.OperatorInterface.assemble>`
    .. |assemble_lincomb| replace:: :meth:`~pymor.operators.interfaces.OperatorInterface.assemble_lincomb`
    .. |as_vector|         replace:: :meth:`~pymor.operators.interfaces.OperatorInterface.as_vector`
    .. |linear|            replace:: :attr:`~pymor.operators.interfaces.OperatorInterface.linear`
    .. |projected|         replace:: :meth:`~pymor.operators.interfaces.OperatorInterface.projected`
    .. |range|             replace:: :attr:`~pymor.operators.interfaces.OperatorInterface.range`
    .. |source|            replace:: :attr:`~pymor.operators.interfaces.OperatorInterface.source`

|Discretizations|
    Discretizations in pyMOR encode the mathematical structure of a given
    discrete problem by acting as container classes for operators. Each
    discretization object has |operators|, |functionals|, |vector_operators| and
    |products| dictionaries holding the |Operators| which appear in the
    formulation of the discrete problem. The keys in these dictionaries describe
    the role of the respective operator in the discrete problem.

    Apart from describing the discrete problem, discretizations also implement
    algorithms for |solving| the given problem, returning |VectorArrays|
    with space |solution_space|. The solution is usually |cached|, s.t.
    subsequent solving of the problem for the same parameters reduces to
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
    representations of |VectorArrays| from the |solution_space|.

    .. |cached|           replace:: :mod:`cached <pymor.core.cache>`
    .. |estimate|         replace:: :meth:`~pymor.discretizations.interfaces.DiscretizationInterface.estimate`
    .. |functionals|      replace:: :attr:`~pymor.discretizations.interfaces.DiscretizationInterface.functionals`
    .. |operators|        replace:: :attr:`~pymor.discretizations.interfaces.DiscretizationInterface.operators`
    .. |products|         replace:: :attr:`~pymor.discretizations.interfaces.DiscretizationInterface.products`
    .. |solution_space|   replace:: :attr:`~pymor.discretizations.interfaces.DiscretizationInterface.solution_space`
    .. |solve|            replace:: :meth:`~pymor.discretizations.interfaces.DiscretizationInterface.solve`
    .. |solving|          replace:: :meth:`solving <pymor.discretizations.interfaces.DiscretizationInterface.solve>`
    .. |vector_operators| replace:: :attr:`~pymor.discretizations.interfaces.DiscretizationInterface.vector_operators`
    .. |visualize|        replace:: :meth:`~pymor.discretizations.interfaces.DiscretizationInterface.visualize`


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
2. It becomes affordable to generate persisten keys for |caching| of computation
   results by generating |state ids| which uniquely identify the object's state.
   Since the state cannot change, these ids have to be computed only once for the
   lifetime of the object.

A class can be made immutable in pyMOR by deriving from |ImmutableInterface|,
which ensures that write access to the object's attributes is prohibited after
`__init__` has been executed. However, note that changes to private attributes
(attributes whose name starts with `_`) are still allowed. It lies in the
implementors responsibility to ensure that changes to these attributes do not
affect the outcome of calls to relevant interface methods. As an example, a call
to :meth:`~pymor.core.cache.CacheableInterface.enable_caching` will set the
objects private `__cache_region` attribute, which might affect the speed of a
subsequent |solve| call, but not its result.

Of course, in many situations one may wish to change properties of an immutable
object, e.g. the number of timesteps for a given discretization. This can be
easily achieved using the `~pymor.core.interfaces.BasicInterface.with_` method
every immutable object has: a call of the form ``o.with_(a=x, b=y)`` will return
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


Creating Discretizations
------------------------

pyMOR ships a small (and still quite incomplete) framework for creating finite
element or finite volume discretizations based on the `NumPy/Scipy
<http://scipy.org>`_ software stack. To end up with an appropriate
|Discretization|, one starts by instantiating an |analytical problem| which
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
|BoundaryInfos| and |Grids| are all immutable, enabling efficient 
disk |caching| for the resulting |Discretizations|, persistent over various
runs of the applications written with pyMOR.

While pyMOR's internal discretizations are useful for getting started quickly
with model reduction experiments, pyMOR's main goal is to allow the reduction of
discretizations provided by external solvers. In order to do so, all that needs
to be done is to provide |VectorArrays|, |Operators| and |Discretizations| which
interact appropriately with the solver. pyMOR makes no assumption on how the
communication with the solver is managed. For instance, communication could take
place via a network protocol or job files.  In particular it should be stressed
that in general no communication of high-dimensional data between the solver
and pyMOR is necessary: |VectorArrays| can merely hold handles to data in the
solver's memory or some on-disk database. Where possible, we favour, however, a
deep integration of the solver with pyMOR by linking the solver code as a Python
extension module. This allows Python to directly access the solver's data
structures which can be used to quickly add features to the high-dimensional
code without any recompilation. A minimal example for such an integration using
`pybindgen <https://code.google.com/p/pybindgen>`_ can be found in the
``src/pymordemos/minimal_cpp_demo`` directory of the pyMOR repository.
The `dune-pymor <https://github.com/pymor/dune-pymor>`_ repository contains
experimental bindings for the `DUNE <http://dune-project.org>`_ software
framework.


Parameters
----------

pyMOR classes implement dependence on a parameter by deriving from the
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


Defaults
--------

pyMOR offers a convenient mechanism for handling default values such as solver
tolerances, cache sizes, log levels, etc. Each default in pyMOR is the default
value of an optional argument of some function. Such an argument is made a
default by decorating the function with the :func:`~pymor.core.defaults.defaults`
decorator::

    @defaults('tolerance')
    def some_algorithm(x, y, tolerance=1e-5)
        ...

Default values can be changed by calling :func:`~pymor.core.defaults.set_defaults`.
A configuration file with all defaults defined in pyMOR can be obtained with
:func:`~pymor.core.defaults.write_defaults_to_file`. This file can then be loaded,
either programmatically or automatically by setting the ``PYMOR_DEFAULTS`` environment
variable.

As an additional feature, if ``None`` is passed as value for a function argument
which is a default, its default value is used instead of ``None``. This allows
writing code of the following form::

    def method_called_by_user(U, V, tolerance_for_algorithm=None):
        ...
        algorithm(U, V, tolerance=tolerance_for_algorithm)
        ...

See the :mod:`~pymor.core.defaults` module for more information.


The Reduction Process
---------------------

The reduction process in pyMOR is handled by so called :mod:`~pymor.reductors`
which take arbitrary |Discretizations| and additional data (e.g. the reduced
basis) to create reduced |Discretizations| along with reconstructor classes
which allow to transform solution vectors of the reduced |Discretization| back
to vectors of the solution space of the high-dimensional |Discretization| (e.g.
by linear combination with the reduced basis). If proper offline/online
decomposition is achieved by the reductor, the reduced |Discretization| will
not store any high-dimensional data. Note that there is no inherent distinction
between low- and high-dimensional |Discretizations| in pyMOR. The only
difference lies in the different types of operators, the |Discretization|
contains.

This observation is particularly apparent in the case of the classical
reduced basis method: the operators and functionals of a given discrete problem
are projected onto the reduced basis space whereas the structure of the problem
(i.e. the type of |Discretization| containing the operators) stays the same.
pyMOR reflects this fact by offering with
:func:`~pymor.reductors.basic.reduce_generic_rb` a generic algorithm which can
be used to RB-project any discretization available to pyMOR. It should be noted
however that this reductor is only able to efficiently
offline/online-decompose affinely |Parameter|-dependent linear problems.
Non-linear problems or such with no affine |Parameter| dependence require
additional techniques such as :mod:`empirical interpolation <pymor.algorithms.ei>`.

If you want to further dive into the inner workings of pyMOR, we highly
recommend to study the source code of
:func:`~pymor.reductors.basic.reduce_generic_rb` and to step through calls of
this method with a Python debugger, such as `ipdb <https://pypi.python.org/pypi/ipdb>`_.
