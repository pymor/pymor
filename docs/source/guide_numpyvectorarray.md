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

```{code-cell}
:load: myst_code_init.py
:tags: [remove-cell]

```

# How to Convert NumPy Arrays to pyMOR VectorArrays?

You may have a 2D NumPy array `X`,
where rows correspond to high-dimensional snapshots,
and you want to convert it to a {{VectorArray}}
to be able to pass it to pyMOR's algorithms.

As an example, let us use

```{code-cell}
import numpy as np
X = np.ones((5, 100))
X.shape
```

Then use the following to convert it to a pyMOR {{NumpyVectorArray}}:

```{code-cell}
from pymor.vectorarrays.numpy import NumpyVectorSpace
V = NumpyVectorSpace.from_numpy(X)
len(V), V.dim
```

In particular, note that {{VectorSpace}} subclasses
(in this case, {{NumpyVectorSpace}})
should be used to build {{VectorArrays}}.
