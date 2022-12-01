---
jupytext:
  text_representation:
   format_name: myst
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: ipynb,myst
    main_language: python
    text_representation:
      format_name: myst
      extension: .md
      format_version: '1.3'
      jupytext_version: 1.11.2
kernelspec:
  display_name: Python 3
  name: python3
---

```{try_on_binder}
```

```{code-cell}
:load: myst_code_init.py
:tags: [remove-cell]


```

# Tutorial: Binding an external PDE solver to pyMOR

One of pyMOR's main features is easy integration of external solvers that implement the full-order model. In this tutorial
we will do this step-by-step for a custom toy solver written in C++.
If you use the [FEniCS](<https://fenicsproject.org>) or [NGSovle](<https://ngsolve.org>) PDE solver libraries,
you can find ready-to-use pyMOR bindings in the {mod}`~pymor.bindings` package. pyMOR support for
[deal.II](<https://dealii.org>) can be found in a [separate repository](<https://github.com/pymor/pymor-deal.II>).

## Defining the PDE solver

Our solver discretizes the one-dimensional Laplace equation {math}`u''(x)=0` on the interval {math}`[\text{left},\text{right}]`
using a central differences scheme with {math}`h=\frac{|\text{right}-\text{left}|}{n}`.
First, we need a class to store our data in and with some basic linear algebra operations
declared on it.

```{literalinclude} minimal_cpp_demo/model.hh
:language: cpp
:lines: 4-19


```

Next, we need the operator that discretizes our PDE.

```{literalinclude} minimal_cpp_demo/model.hh
:language: cpp
:lines: 22-32


```

Together with some header guards, these two snippets make up our {download}`model.hh <minimal_cpp_demo/model.hh>`.

The definitions for the `Vector` class are pretty straightforward:

```{literalinclude} minimal_cpp_demo/model.cc
:language: cpp
:lines: 7-35

```

Just like the diffusion operator that computes a central differences stencil:

```{literalinclude} minimal_cpp_demo/model.cc
:language: cpp
:lines: 39-49

```

This completes all the C++ code needed for the toy solver itself. Next, we will make this code usable from Python.
We utilize the [pybind11](<https://github.com/pybind/pybind11>) library to create a Python
[extension module](<https://docs.python.org/3/extending/extending.html>) named `model`, that allows us to manipulate
instances of the C++ `Vector` and `DiffusionOperator` classes.

Compiling the PDE solver as a shared library and creating Python bindings for it using
[pybind11](<https://github.com/pybind/pybind11>), [Cython](<https://cython.org>) or
[ctypes](<https://docs.python.org/3/library/ctypes.html>) is the preferred way of integrating
external solvers, as it offers maximal flexibility and performance. For instance, in this
example we will actually completely implement the {{ Model }} in Python using a
{mod}`time stepper <pymor.algorithms.timestepping>` from pyMOR to
{meth}`~pymor.models.interface.Model.solve` the {{ Model }}.

When this is not an option,
[RPC](<https://en.wikipedia.org/wiki/Remote_procedure_call>)-based approaches are
possible as well. For small to medium-sized linear problems, another option it to import
system matrices and snapshot data into pyMOR via file exchange and to use NumPy-based
{mod}`Operators <pymor.operators.numpy>` and {mod}`VectorArrays <pymor.vectorarrays.numpy>`
to represent the full-order model.

## Binding the solver to Python

All of the C++ code related to the extension module is defined inside a scope started with

```{literalinclude} minimal_cpp_demo/model.cc
:language: cpp
:lines: 56-57

```

This tells pybind11 to make the contained symbols accessible in the module instance `m` that will be importable by the
name `model`. Now we create a new pybind11 `class\_` object that wraps the `DiffusionOperator`. Note that the module
instance is passed to the constructor alongside a name for the Python class and a docstring. The second
line shows how to define an init function for the Python object by using the special `py:init` object to
forward arguments to the C++ constructor.

```{literalinclude} minimal_cpp_demo/model.cc
:lines: 60-61
:language: cpp

```

Next, we define read-only properties on the Python side named after and delegated to the members of the C++ class.

```{literalinclude} minimal_cpp_demo/model.cc
:lines: 62-63
:language: cpp

```

The last `DiffusionOperator`-related line exposes the function call to apply in the same way:

```{literalinclude} minimal_cpp_demo/model.cc
:lines: 64
:language: cpp

```

This is everything that is needed to expose the operator to Python. We will now do the same for the `Vector`,
with a few more advanced techniques added.

```{literalinclude} minimal_cpp_demo/model.cc
:lines: 66-68
:language: cpp

```

Again we define a `py:class\_` with appropiate name and docstring, but now we also indicate to pybind11
that this class will implement the [buffer protocol](<https://docs.python.org/3/c-api/buffer.html>), which basically
exposes direct access to the chunk of memory associated with a `Vector` instance to Python. We also see how we can dispatch multiple init functions
by using `py:init` objects with C++ lambda functions.
Note that direct memory access to the vector data from Python is not required to integrate a solver with pyMOR.
It is, however, useful for debugging and quickly modifying or extending the solver from within Python. For instance,
in our toy example we will use the direct memory access to quickly define a visualization of the solutions and to
construct the right-hand side vector for our problem.

```{literalinclude} minimal_cpp_demo/model.cc
:lines: 70-74
:language: cpp

```

```{literalinclude} minimal_cpp_demo/model.cc
:lines: 76-80
:language: cpp

```

This completes the {download}`model.cc <minimal_cpp_demo/model.cc>`.

This extension module needs to be compiled to a shared object that the Python interpreter can import.
We use a minimal [CMake](<https://cmake.org/>) project that generates makefiles for us to achieve this.

First we make sure pybind11 can be used:

```{literalinclude} minimal_cpp_demo/CMakeLists.txt
:lines: 1-6
:language: cmake

```

Next, we define a new library with our `model.cc` as the single source file and let pybind11 set the proper compile
flags.

```{literalinclude} minimal_cpp_demo/CMakeLists.txt
:lines: 9-12
:language: cmake

```

That is all that is needed for {download}`CMakeLists.txt <minimal_cpp_demo/CMakeLists.txt>`.
In the next step, we will switch to a bash terminal and actually compile this module.

After creating a build directory for the module, we let cmake initialize the build and call make to execute the
compilation.

```{code-cell}
:tags: [raises-exception]

%%bash
mkdir -p minimal_cpp_demo/build
cmake -B minimal_cpp_demo/build -S minimal_cpp_demo
make -C minimal_cpp_demo/build
```

To be able to use this extension module we need to insert the build directory into the path where the Python
interpreter looks for things to import. Afterwards we can import the module and create and use the exported classes.

```{code-cell}
import sys
sys.path.insert(0, 'minimal_cpp_demo/build')

import model
mymodel = model.DiffusionOperator(10, 0, 1)
myvector = model.Vector(10, 0)
mymodel.apply(myvector, myvector)
dir(model)
```

## Using the exported Python classes with pyMOR

All of pyMOR's algorithms operate on {{ VectorArray }} and {{ Operator }} objects that all share the same programming interface. To be able to use
our Python `model.Vector` and `model.DiffusionOperator` in pyMOR, we have to provide implementations of
{{ VectorArray }}, {{ VectorSpace }} and {{ Operator }} that wrap the classes defined in the extension module
and translate calls to the interface methods into operations on `model.Vector` and `model.DiffusionOperator`.

Instead of writing a full implementaion of a {{ VectorArray }} that manages multiple `model.Vector`
instances, we can instead implement a wrapper `WrappedVector` for a single `model.Vector` instance based on
{class}`~pymor.vectorarrays.list.CopyOnWriteVector` which will be used to create
{{ ListVectorArrays }} via a {class}`~pymor.vectorarrays.list.ListVectorSpace`-based `WrappedVectorSpace`.

The {class}`~pymor.vectorarrays.list.CopyOnWriteVector` base class manages a reference count for
us and automatically copies data when necessary in methods {meth}`~pymor.vectorarrays.list.CopyOnWriteVector.scal`
and {meth}`~pymor.vectorarrays.list.CopyOnWriteVector.axpy`. To use this, we need to implement
{meth}`~pymor.vectorarrays.list.CopyOnWriteVector._scal`
and {meth}`~pymor.vectorarrays.list.CopyOnWriteVector._axpy` in addition to all the abstract
methods from  {class}`~pymor.vectorarrays.list.CopyOnWriteVector`. We can get away
with using just a stub that raises an {class}`~NotImplementedError` in some methods that are not actually called in our example.

```{code-cell}
from pymor.operators.interface import Operator
from pymor.vectorarrays.list import CopyOnWriteVector, ListVectorSpace

import numpy as np
import math
from model import Vector, DiffusionOperator


class WrappedVector(CopyOnWriteVector):

    def __init__(self, vector):
        assert isinstance(vector, Vector)
        self._impl = vector

    @classmethod
    def from_instance(cls, instance):
        return cls(instance._impl)

    def to_numpy(self, ensure_copy=False):
        # Note how this uses the buffer protocol setup to allow efficient
        # data access as a Numpy Vector
        result = np.frombuffer(self._impl, dtype=np.float64)
        if ensure_copy:
            result = result.copy()
        return result

    def _copy_data(self):
        # This uses the second exposed 'init' signature to delegate to the C++ copy constructor
        self._impl = Vector(self._impl)

    def _scal(self, alpha):
        self._impl.scal(alpha)

    def _axpy(self, alpha, x):
        self._impl.axpy(alpha, x._impl)

    def inner(self, other):
        return self._impl.inner(other._impl)

    def norm(self):
        return math.sqrt(self.inner(self))

    def norm2(self):
        return self.inner(self)

    def sup_norm(self):
        raise NotImplementedError

    def dofs(self, dof_indices):
        raise NotImplementedError

    def amax(self):
        raise NotImplementedError
```

The implementation of the `WrappedVectorSpace` is very short as most of the necessary methods
of {{ VectorSpace }} are implemented in {class}`~pymor.vectorarrays.list.ListVectorSpace`.

```{code-cell}
class WrappedVectorSpace(ListVectorSpace):

    def __init__(self, dim):
        self.dim = dim

    def zero_vector(self):
        return WrappedVector(Vector(self.dim, 0))

    def make_vector(self, obj):
        # obj is a `model.Vector` instance
        return WrappedVector(obj)

    def __eq__(self, other):
        return type(other) is WrappedVectorSpace and self.dim == other.dim
```

Wrapping the `model.DiffusionOperator` is straightforward as well. We just need to attach
suitable {{ VectorSpaces }} to the class and implement the application of the operator on a {{ VectorArray }}
as a sequence of applications on single vectors.

```{code-cell}
class WrappedDiffusionOperator(Operator):
    def __init__(self, op):
        assert isinstance(op, DiffusionOperator)
        self.op = op
        self.source = WrappedVectorSpace(op.dim_source)
        self.range = WrappedVectorSpace(op.dim_range)
        self.linear = True

    @classmethod
    def create(cls, n, left, right):
        return cls(DiffusionOperator(n, left, right))

    def apply(self, U, mu=None):
        assert U in self.source

        def apply_one_vector(u):
            v = Vector(self.range.dim, 0)
            self.op.apply(u._impl, v)
            return v

        return self.range.make_array([apply_one_vector(u) for u in U.vectors])
```

## Putting it all together

As a demonstration, we will use our toy Laplace solver to compute an approximation for
the transient diffusion equation

```{math}
\frac{\partial u}{\partial t} = {\alpha_\mu} \frac{\partial^2 u}{\partial x^2},
```

with explicit timestepping provided by pyMOR, with a parameterized, block-wise defined,  diffusion
coefficient  {math}`\alpha_\mu`.

First up, we implement a `discretize` function that uses the `WrappedDiffusionOperator` and `WrappedVectorSpace`
to assemble an {{ InstationaryModel }}.

```{code-cell}
from pymor.algorithms.pod import pod
from pymor.algorithms.timestepping import ExplicitEulerTimeStepper
from pymor.discretizers.builtin.gui.visualizers import OnedVisualizer
from pymor.models.basic import InstationaryModel
from pymor.discretizers.builtin import OnedGrid
from pymor.operators.constructions import VectorOperator, LincombOperator
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.reductors.basic import InstationaryRBReductor


def discretize(n, nt, blocks):
    h = 1. / blocks
    ops = [WrappedDiffusionOperator.create(n, h * i, h * (i + 1)) for i in range(blocks)]
    pfs = [ProjectionParameterFunctional('diffusion_coefficients', blocks, i) for i in range(blocks)]
    operator = LincombOperator(ops, pfs)

    initial_data = operator.source.zeros()

    rhs_vec = operator.range.zeros()
    rhs_data = rhs_vec.vectors[0].to_numpy()
    rhs_data[:] = np.ones(len(rhs_data))
    rhs_data[0] = 0
    rhs_data[len(rhs_data) - 1] = 0
    rhs = VectorOperator(rhs_vec)

    # we can re-use pyMOR's builtin grid and visualizer for our demonstration
    grid = OnedGrid(domain=(0, 1), num_intervals=n)
    visualizer = OnedVisualizer(grid)

    time_stepper = ExplicitEulerTimeStepper(nt)

    fom = InstationaryModel(T=1, operator=operator, rhs=rhs, initial_data=initial_data,
                            time_stepper=time_stepper, num_values=20,
                            visualizer=visualizer, name='C++-Model')
    return fom
```

Now we can build a reduced basis for our model. Note that this code is not specific to our wrapped classes.
Those wrapped classes are only directly used in the `discretize` call.

```{code-cell}
%matplotlib inline
# discretize
fom = discretize(50, 10000, 4)
parameter_space = fom.parameters.space(0.1, 1)

# generate solution snapshots
snapshots = fom.solution_space.empty()
for mu in parameter_space.sample_uniformly(2):
    snapshots.append(fom.solve(mu))

# apply POD
reduced_basis = pod(snapshots, modes=4)[0]

# reduce the model
reductor = InstationaryRBReductor(fom, reduced_basis, check_orthonormality=True)
rom = reductor.reduce()

# stochastic error estimation
mu_max = None
err_max = -1.
for mu in parameter_space.sample_randomly(10):
    U_RB = reductor.reconstruct(rom.solve(mu))
    U = fom.solve(mu)
    err = np.max((U_RB-U).norm())
    if err > err_max:
        err_max = err
        mu_max = mu

# visualize maximum error solution
U_RB = (reductor.reconstruct(rom.solve(mu_max)))
U = fom.solve(mu_max)
fom.visualize((U_RB, U), title=f'mu = {mu}', legend=('reduced', 'detailed'))
```

As you can see in this comparison, we get a good approximation of the full-order model here and
the error plot confirms it:

```{code-cell}
fom.visualize((U-U_RB), title=f'mu = {mu}', legend=('error'))
```

You can download this demonstration plus the wrapper definitions as a
notebook {nb-download}`tutorial_external_solver.ipynb` or
as Markdown text {download}`tutorial_external_solver.md`.
