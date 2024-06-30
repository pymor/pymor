# How to Run Demo Scripts?

pyMOR ships several example scripts that showcase various features of the library.
While many features are also covered in our {doc}`tutorials`, the demos are more extensive
and often have various command-line flags which allow to run the script for different
parameters or problems.
All demos can be found in the [src/pymordemos](https://github.com/pymor/pymor/tree/main/src/pymordemos)
directory of the source repository.

The demo scripts can be launched directly from the source tree:

```
./thermalblock.py --plot-err --plot-solutions 3 2 3 32
```

or by using the `pymor-demo` script that is installed with pyMOR:

```
pymor-demo thermalblock --plot-err --plot-solutions 3 2 3 32
```
