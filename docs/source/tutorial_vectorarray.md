---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{try_on_binder}
```

```{code-cell} ipython3
:load: myst_code_init.py
:tags: [remove-cell]


```

# Tutorial: VectorArray Basics

## What are VectorArrays and what are they used for in pyMOR?

While pyMOR uses {{ NumPy }} arrays to represent extrinsic quantities like
inputs or outputs of a {{ Model }}, the Model's internal state is always represented
by {{ VectorArrays }}, ordered collections of abstract vectors of the same
dimension.
The reason for this is that when pyMOR is used in conjunction with an external
solver that implements the FOM, state space vectors must be compatible with the
solver's internal data structures, for instance when some {{ Operator }}
appearing in the FOM shall be applied to the vector.
Instead of constantly converting between NumPy arrays and the solver's data structures
-- which might even be impossible in some cases -- pyMOR uses {{ VectorArrays }} to access
these vectors through a unified interface.
So as soon as you get in touch with a model's state-space data, e.g., by calling its
{meth}`~pymor.models.interface.Model.solve` method, you will work with {{ VectorArrays }}.
Note that this also the case for ROMs. pyMOR does not distinguish between FOMs and ROMs,
so even though no external solver is involved, the ROM state is still represented by
{{ VectorArrays }}.

## VectorSpaces / How to create VectorArrays

{{ VectorArrays }} are never instantiated directly but are always created by the
{{ VectorSpace }} they belong to.
You will find {{ VectorSpaces }} as the {attr}`~pymor.operators.interface.Operator.source`
and {attr}`~pymor.operators.interface.Operator.range` space of {{ Operators }} or as the
{attr}`~pymor.models.interface.Model.solution_space` of {{ Models }}.
In the following, we will work with {{ NumpyVectorArrays }} that internally store the
vectors as two-dimensional NumPy arrays.
The corresponding {{ VectorSpace }} is called {{ NumpyVectorSpace }}.
To make a {{ NumpyVectorSpace }}, we need to specify the dimension of the contained vectors:

```{code-cell} ipython3
from pymor.basic import NumpyVectorSpace
import numpy as np

space = NumpyVectorSpace(7)
```

The dimension can be accessed via the {attr}`~pymor.vectorarrays.interface.VectorSpace.dim`
attribute:

```{code-cell} ipython3
space.dim
```

Our first {{ VectorArray }} will only contain three zero vectors:

```{code-cell} ipython3
U = space.zeros(3)
```

To check that we indeed have created three zero vectors of dimension 7, we can do the following:

```{code-cell} ipython3
print(len(U))
print(U.dim)
print(U.norm())
```

Just zero vectors are boring.
To create vectors with different entries, we can use the following methods:

```{code-cell} ipython3
print(space.ones(1))
print(space.full(42, count=4))
print(space.random(2))
```

Sometimes, for instance when accumulating vectors in a loop, we want to initialize a
{{ VectorArray }} with no vectors in it.
We can use the {meth}`~pymor.vectorarrays.interface.VectorSpace.empty` method for that:

```{code-cell} ipython3
U.empty()
```

You might wonder how to create {{ VectorArrays }} with more interesting data.
When implementing {{ Operators }} or, for instance, the
{meth}`~pymor.models.interface.Model.solve` method, you will get in touch with data
from the actual solver backend you are using, that needs to be wrapped into a
corresponding {{ VectorArray }}.
For that, every {{ VectorSpace }} has a
{meth}`~pymor.vectorarrays.interface.VectorSpace.make_array` method that takes care of the
wrapping for you.
In case of {{ NumpyVectorSpace }}, the backend is NumPy and the data is given as NumPy arrays:

```{code-cell} ipython3
space.make_array_TP(np.arange(0, 14).reshape((7, 2)))
```

## Converting NumPy arrays to VectorArrays

Some but not all {{ VectorArrays }} can be initialized from NumPy arrays.
For these arrays, the corresponding {{ VectorSpace }} implements the
{meth}`~pymor.vectorarrays.interface.VectorSpace.from_numpy_TP` method:

```{code-cell} ipython3
space = NumpyVectorSpace(4)
numpy_array = np.linspace(0, 4, 8).reshape((4, 2))
print(numpy_array)

vector_array = space.from_numpy_TP(numpy_array)
vector_array
```

```{warning}
The generated {{ VectorArray }} might take ownership of the provided data.
In particular, this is the case for {{ NumpyVectorArrays }}.
So if you change the {{ VectorArray }}, the original array will change and vice versa:
```

```{code-cell} ipython3
numpy_array[0, 0] = 99
vector_array
```

To avoid this problem, you can set `ensure_copy` to `True`:

```{code-cell} ipython3
numpy_array = np.linspace(0, 4, 8).reshape((4, 2))
vector_array = space.from_numpy_TP(numpy_array, ensure_copy=True)

numpy_array[0, 0] = 99
print(numpy_array)
vector_array
```

If you quickly want to create a {{ NumpyVectorArray }}, you can also use
{meth}`~pymor.vectorarrays.interface.VectorSpace.from_numpy_TP`
as a class method of {{ NumpyVectorSpace }}, which will infer the dimension of the
space from the data:

```{code-cell} ipython3
vector_array = NumpyVectorSpace.from_numpy_TP(numpy_array)
vector_array
```

```{warning}
Do not use {meth}`~pymor.vectorarrays.interface.VectorSpace.from_numpy_TP`
in generic code that is supposed to work with arbitrary external solvers that might
not support converting data from NumPy or only at high costs.
Using it is fine, however, when you know that you are dealing with {{ NumpyVectorArrays }},
for instance when building ROMs.
```

## Converting VectorArrays to NumPy arrays

Some {{ VectorArrays }} also implement a
{meth}`~pymor.vectorarrays.interface.VectorArray.to_numpy_TP` method that returns the
internal data as a NumPy array:

```{code-cell} ipython3
space = NumpyVectorSpace(4)
vector_array = space.random(3)
vector_array
```

```{code-cell} ipython3
array = vector_array.to_numpy_TP()
array
```

```{warning}
Again, the returned NumPy array might share data with the {{ VectorArray }}:
```

```{code-cell} ipython3
array[:] = 0
vector_array
```

As with {meth}`~pymor.vectorarrays.interface.VectorSpace.from_numpy_TP`
we can use the `ensure_copy` parameter to make sure we get a copy:

```{code-cell} ipython3
vector_array = space.random(3)
array = vector_array.to_numpy_TP(ensure_copy=True)
array[:] = 0
vector_array
```

## Rows and columns

First time pyMOR users coming from numerical linear algebra often say that
pyMOR uses row vectors instead column vectors.
However, it is more useful to think of {{ VectorArrays }} as simple lists of vectors,
that do not have any notion of being a row or column vector.
When a matrix {{ Operator }} is applied to a {{ VectorArray }}, think of a `for`-loop,
where the corresponding linear {{ Operator }} is applied individually to each vector in
the array, not of a matrix-matrix product.
What is true, however, is that
{meth}`~pymor.vectorarrays.interface.VectorSpace.from_numpy_TP` /
{meth}`~pymor.vectorarrays.interface.VectorArray.to_numpy_TP`
interpret {{ VectorArrays }} as matrices of row vectors.
The reason for that is that NumPy prefers a C-like memory layout for matrices, where the
individual rows are stored consecutively in memory.
(In contrast, Matlab uses a Fortran-like memory layout, where the columns are
stored consecutively in memory.)

## Basic operations

```{code-cell} ipython3
space = NumpyVectorSpace(5)

U = space.ones(2)
V = space.full(2)
W = space.random(1)
print(U)
print(V)
print(W)
```

We can {meth}`~pymor.vectorarrays.interface.VectorArray.append` the vectors from one array
to another:

```{code-cell} ipython3
V.append(space.full(3))
print(V)
```

The original array is left unchanged.
If we do not want to copy the vectors but remove them from the original array, we can use the
`remove_from_other` argument.

We can add {{ VectorArrays }} and multiply by a scalar:

```{code-cell} ipython3
print(U + V)
print(U * 2)
```

All operations are vectorized as they are for NumPy arrays.
Addition and other operations also allow 'broadcasting' a single vector to the entire array:

```{code-cell} ipython3
print(V + W)
```

{{ VectorArrays }} can be scaled in place using the
{meth}`~pymor.vectorarrays.interface.VectorArray.scal` method:

```{code-cell} ipython3
U.scal(3)
print(U)
```

Adding a scalar multiple can be achieved using
{meth}`~pymor.vectorarrays.interface.VectorArray.axpy`:

```{code-cell} ipython3
U.axpy(2, V)
print(U)
```

The same could be achieved with:

```{code-cell} ipython3
U = space.ones(2)
U += 2 * V
print(U)
```

However, this would create an unnecessary temporary for `2 * V`.
{meth}`~pymor.vectorarrays.interface.VectorArray.axpy` also allows broadcasting
or specifying multiple coefficients:

```{code-cell} ipython3
U = space.ones(2)
U.axpy(2, W)
print(U)

U = space.ones(2)
U.axpy(np.array([1, -1]), V)
print(U)
```

Often, a {{ VectorArray }} represents a basis, and we want to form linear combinations
w.r.t. this basis.
For this, we can use the {meth}`~pymor.vectorarrays.interface.VectorArray.lincomb_TP` method:

```{code-cell} ipython3
V.lincomb_TP(np.array([2,3]))
```

This can also be vectorized:

```{code-cell} ipython3
V.lincomb_TP(np.array([[2,1,0],
                       [3,0,1]]))
```

Inner products can be computed using the {meth}`~pymor.vectorarrays.interface.VectorArray.inner`
method:

```{code-cell} ipython3
print(U)
print(V)
U.inner(V)
```

`inner` returns the matrix of all possible inner products between vectors in `U` and `V`.
{meth}`~pymor.vectorarrays.interface.VectorArray.pairwise_inner` only returns the inner product
between the i-th vector in `U` and the i-th vector in `V`:

```{code-cell} ipython3
U.pairwise_inner(V)
```

Norms can be conveniently computed using using the
{meth}`~pymor.vectorarrays.interface.VectorArray.norm` or
{meth}`~pymor.vectorarrays.interface.VectorArray.norm2` methods:

```{code-cell} ipython3
print(U.norm())
print(np.sqrt(U.norm2()))
print(np.sqrt(U.pairwise_inner(U)))
```

The `norm` and `inner` methods have an optional `product` argument that can be used to compute
norms and inner products w.r.t. the specified product {{ Operator }}:

```{code-cell} ipython3
from pymor.operators.numpy import NumpyMatrixOperator
mat = np.diag(np.arange(5) + 1.)
print(mat)
prod = NumpyMatrixOperator(mat)
print(U)
print(U.norm2(product=prod))
```

Finally, we can use the {meth}`~pymor.vectorarrays.interface.VectorArray.gramian` method as a
shorthand to compute the matrix of all inner products of vectors in `U`:

```{code-cell} ipython3
print(U.gramian())
```

## Copies

Remember that assignment using `=` never copies data in Python, but only assigns a new name
to an existing object:

```{code-cell} ipython3
space = NumpyVectorSpace(3)
U = space.ones()
V = U
U *= 0
print(U)
print(V)
```

To get a new array we use the {meth}`~pymor.vectorarrays.interface.VectorArray.copy` method:

```{code-cell} ipython3
U = space.ones()
V = U.copy()
U *= 0
print(U)
print(V)
```

{{ VectorArrays }} use copy-on-write semantics.
This means that just calling {meth}`~pymor.vectorarrays.interface.VectorArray.copy` will not
copy any actual data.
Only when one of the arrays is modified, the data is copied.
Sometimes, this behavior is not desired. Then `deep=True` can be specified to force an immediate
copy of the data.

## Indexing and views

Like NumPy arrays, {{ VectorArrays }} can be indexes with positive or negative integers,
slices or lists of integers.
This *always* results in a new {{ VectorArray }} that is a view onto the data in the original
array.

```{code-cell} ipython3
space = NumpyVectorSpace(4)
U = space.ones(5)
print(U)
U[1].scal(2)
U[-1].scal(42)
print(U)
print(U[:3].gramian())
print(U[[1,-1]].lincomb_TP(np.array([1,1])))
```

```{note}
Indexing with lists of indices creates a view as well.
This is different from NumPy where 'advanced indexing' always yields a copy.
```

```{code-cell} ipython3
V = U[[3,0]]
print(V.is_view)
print(V.base is U)
V.scal(0)
print(U)
```

## DOF access

All interface methods of {{ VectorArray }} we have covered so far operate on abstract vectors
and either return `None`, new {{ VectorArrays }} or scalar quantities based on inner products.
In general, there is no way to extract the actual (potentially high-dimensional) data stored
in a {{ VectorArray }}.
This is a deliberate design decision in pyMOR as the computationally heavy operations on this
data should be handled by the respective backend.
In particular, note again that
{meth}`~pymor.vectorarrays.interface.VectorSpace.from_numpy_TP` /
{meth}`~pymor.vectorarrays.interface.VectorArray.to_numpy_TP`
should only be used for {{ NumpyVectorArrays }}, debugging or other special cases.
Usually, algorithms should not rely on these methods.

However, sometimes it is necessary to access selected degrees of freedom (entries) of the vectors
stored in the {{ VectorArray }}, in particular for techniques such as empirical interpolation.
For that reason, most {{ VectorArrays }} implement the
{meth}`~pymor.vectorarrays.interface.VectorArray.dofs` method, which allows extracting the
values of certain degrees of freedom:

```{code-cell} ipython3
space = NumpyVectorSpace(6)
U = space.from_numpy_TP(np.arange(6) * 2.)
U.append(space.full(-1))
print(U)
print(U.dofs_TP(np.array([3, 5])))
```

Related methods are {meth}`~pymor.vectorarrays.interface.VectorArray.sup_norm` and
{meth}`~pymor.vectorarrays.interface.VectorArray.amax`:

```{code-cell} ipython3
print(U.sup_norm())
dofs, values = U.amax()
print(dofs, values)
for i in range(len(U)):
    print(np.abs(U[i].dofs_TP([dofs[i]])) == values[i])
```

By speaking of degrees of freedom, we assume that our vectors are coefficient vectors w.r.t.
some basis and that `dofs` returns those coefficients.
We further assume that, if defined,
{meth}`~pymor.vectorarrays.interface.VectorSpace.from_numpy_TP` returns the NumPy array of the
values at all degrees of freedom and that {meth}`~pymor.vectorarrays.interface.VectorArray.inner`
is just the Euclidean inner product of these coefficient vectors:

```{code-cell} ipython3
numpy_array = U.dofs_TP(np.arange(U.dim))
print(numpy_array)
print(numpy_array == U.to_numpy_TP())
print(numpy_array.T @ numpy_array == U.inner(U))
```

```{warning}
Theoretically, {meth}`~pymor.vectorarrays.interface.VectorArray.dofs` allows to extract all
data from a {{ VectorArray }} by calling `U.dofs_TP(np.arange(U.dim))` as above.
However, doing so is strongly discouraged and might lead to bad performance as
{meth}`~pymor.vectorarrays.interface.VectorArray.dofs` is
designed to be used to only extract a small amount of degrees of freedom.
```

## Complex Numbers

Like NumPy, {{ VectorArrays }} transparently handle complex entries and {{ VectorSpaces }}
do not distinguish between being complex or not:

```{code-cell} ipython3
space = NumpyVectorSpace(3)
U = space.ones(2)
U[1].scal(1j)
print(U)
```

This also works for external solvers that do not support complex numbers themselves.
In that case, pyMOR automatically manages pairs of real vectors to represent the real and
imaginary parts and translates all operations on these 'complexified' vectors to the
corresponding operations on the real and complex parts.
For any {{ VectorArray }}, those can be accessed through the
{attr}`~pymor.vectorarrays.interface.VectorArray.real` and
{attr}`~pymor.vectorarrays.interface.VectorArray.imag`
attributes:

```{code-cell} ipython3
U.real, U.imag
```

Even when the array is real,
{attr}`~pymor.vectorarrays.interface.VectorArray.real` will alwasy return a copy of the original
array.
When it comes to inner products, the convention in pyMOR is that inner products are anti-linear
in the first argument:

```{code-cell} ipython3
print((U * 1j).inner(U) == U.inner(U) * (-1j))
print(U.inner(U * 1j) == U.inner(U) * 1j)
```

## Internals / other types of arrays

Finally, we want to take brief look at the internals of {{ VectorArrays }} and also try one other type of
array. Let's first build another {{ NumpyVectorArray }}.

```{code-cell} ipython3
space = NumpyVectorSpace(4)
U = space.random(3)
```

So where does `U` actually store its data?
{{ NumpyVectorArray }} is a subclass of the {{ VectorArray }} interface class, which takes care of
arguments checking and indexing.
The actual work, however, is performed by an implementation object, which also holds the data:

```{code-cell} ipython3
type(U.impl)
```

In the case of {class}`~pymor.vectorarrays.numpy.NumpyVectorArrayImpl`, the data is stored in the
private `_array` attribute:

```{code-cell} ipython3
U.impl._array
```

However, you should never access this attribute directly and use
{meth}`~pymor.vectorarrays.interface.VectorSpace.from_numpy_TP` /
{meth}`~pymor.vectorarrays.interface.VectorArray.to_numpy_TP`
instead, as `_array` may not be what you expect:
the array creation methods
{meth}`~pymor.vectorarrays.interface.VectorSpace.empty`,
{meth}`~pymor.vectorarrays.interface.VectorSpace.zeros`,
{meth}`~pymor.vectorarrays.interface.VectorSpace.ones`,
{meth}`~pymor.vectorarrays.interface.VectorSpace.full` have an additional `reserve` argument,
which allows the user to give the {{ VectorArray }} implementation a hint how large the array
might grow.+
The implementation can use this to over-allocate memory to improve performance:

```{code-cell} ipython3
V = space.ones(reserve=5)
V.impl._array
```

Here, `_array` is large enough to store 5 vectors, even though the array only contains one vector:

```{code-cell} ipython3
print(len(V))
print(V.to_numpy_TP())
```

If we append one vector, the second row in the array is overwritten:

```{code-cell} ipython3
V.append(space.full(2))
V.impl._array
```

As NumPy arrays store their data consecutively in memory, this can significantly improve
performance of iterative algorithms where new vectors are repeatedly appended to a given array.
However, this is only useful for implementations which do store the data consecutively in memory.

We will now look at another type of {{ VectorArray }}, which uses Python lists of one-dimensional
NumPy arrays to store the data as a list of vectors:

```{code-cell} ipython3
from pymor.vectorarrays.list import NumpyListVectorSpace
space = NumpyListVectorSpace(4)
W = space.random(3)
```

`W` behaves as any other array.
It also implements `to_numpy_TP` so that we can look at the data directly:

```{code-cell} ipython3
W.to_numpy_TP()
```

However, the implementation class now holds a list of vector objects:

```{code-cell} ipython3
print(type(W))
print(type(W.impl))
print(W.impl._list)
print(type(W.impl._list))
```

Each of these {class}`~pymor.vectorarrays.list.NumpyVector` objects now stores one of the vectors
in the array:

```{code-cell} ipython3
W.impl._list[1]._array
```

Ending this tutorial, we want to take a brief look how one of the interface methods is actually
implemented by the different array types.
We choose {meth}`~pymor.vectorarrays.interface.VectorArray.pairwise_inner` as the implementation is
particularly simple.
Here is the implementation for {{ NumpyVectorArray }}:

```{code-cell} ipython3
from pymor.tools.formatsrc import print_source
print_source(U.impl.pairwise_inner)
```

We see the actual call to NumPy that does all computations in the last line.
The implementation has to take into account that one of the arrays might be a view, in which case the
selected indices are given by `ind`/`oind`.
Also, `reserve` might have been used when the array was created, making `_array` larger than it
actually is.
The true length is stored in the `_len` attribute, so that the first `_len` vectors from the array
are selected.

Here is the code for the `list`-based array:

```{code-cell} ipython3
print_source(W.impl.pairwise_inner)
```

`W.impl` is a generic {class}`~pymor.vectorarrays.list.ListVectorArrayImpl` object, which provides a
common implementation for all {{ VectorArrays }} based on Python lists.
The implementation just loops over all pairs `a, b` of vector objects in both arrays and calls `a`'s
{meth}`~pymor.vectorarrays.interface.VectorArray.inner` method, which computes the actual inner product.
In case of {class}`~pymor.vectorarrays.list.NumpyVector`, the code looks like this:

```{code-cell} ipython3
print_source(W.impl._list[0].inner)
```

As most PDE solver libraries only know of single vectors and have no notion of an array of vectors,
{{ VectorArrays }} for external solvers are typically based on {{ ListVectorArray }} /
{class}`~pymor.vectorarrays.list.ListVectorArrayImpl`.
In that case, only the interface methods for a single {class}`~pymor.vectorarrays.list.Vector` object
have to be implemented.
Note that for NumPy-based data, {{ NumpyVectorArray }} is the default {{ VectorArray }} type used throughout
pyMOR, as it can benefit from vectorized operations on the underlying NumPy array.
{class}`~pymor.vectorarrays.list.NumpyListVectorSpace` is only implemented in pyMOR for testing purposes.
