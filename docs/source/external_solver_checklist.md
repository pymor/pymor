# Checklist for Using an External Solver in pyMOR

This checklist guides you through the process of using an external solver (already bound to pyMOR) for solving PDEs.  
**Reference Tutorial**: [Binding an external PDE solver to pyMOR](./tutorial_external_solver.md).

---

## 1. **Prerequisites**

- [ ] Install pyMOR and its dependencies.
- [ ] Install/compile the external solver (e.g., the C++ code with pybind11 bindings).
- [ ] Verify compatibility between pyMOR and the solver (Python/C++ versions, dependencies).

---

## 2. **Set Up the Solver Bindings**

- [ ] Compile the solver’s Python module (e.g., `model.so`):
  ```bash
  mkdir -p minimal_cpp_demo/build
  cmake -B minimal_cpp_demo/build -S minimal_cpp_demo
  make -C minimal_cpp_demo/build
  ```
- [ ] Add the compiled module to Python’s import path:
  ```python
  import sys
  sys.path.insert(0, 'minimal_cpp_demo/build')
  ```

---

## 3. **Wrap the Solver for pyMOR**

- [ ] Create a `WrappedVector` class inheriting from `CopyOnWriteVector`:
  - Implement `to_numpy`, `_scal`, `_axpy`, and other required methods.
  - Reference: [WrappedVector implementation](../../src/pymordemos/minimal_cpp_demo/wrapper.py).
- [ ] Define a `WrappedVectorSpace` inheriting from `ListVectorSpace`:
  - Implement `zero_vector`, `make_vector`, and `__eq__`.
- [ ] Wrap the solver’s operator in a `WrappedDiffusionOperator` class:
  - Inherit from `pymor.operators.interface.Operator`.
  - Implement `apply` to delegate calls to the external solver.

---

## 4. **Build the Full-Order Model (FOM)**

- [ ] Use the wrapped classes to define the discretization:
  ```python
  def discretize(n, nt, blocks):
      ops = [WrappedDiffusionOperator.create(...)]
      operator = LincombOperator(ops, ...)
      # ... (see tutorial for details)
      return InstationaryModel(...)
  ```
- [ ] Attach visualization tools (e.g., `OnedVisualizer` for 1D problems).

---

## 5. **Solve and Analyze**

- [ ] Solve the FOM for specific parameters:
  ```python
  fom = discretize(n=50, nt=10000, blocks=4)
  U = fom.solve(mu)
  ```
- [ ] Visualize results:
  ```python
  fom.visualize(U)
  ```

---

## 6. **Model Order Reduction**

- [ ] Generate snapshots for training:
  ```python
  snapshots = fom.solution_space.empty()
  for mu in parameter_space.sample_uniformly(2):
      snapshots.append(fom.solve(mu))
  ```
- [ ] Compute a reduced basis (e.g., via POD):
  ```python
  reduced_basis = pod(snapshots, modes=4)[0]
  ```
- [ ] Reduce the model:
  ```python
  reductor = InstationaryRBReductor(fom, reduced_basis)
  rom = reductor.reduce()
  ```
- [ ] Validate the reduced model:
  ```python
  U_RB = reductor.reconstruct(rom.solve(mu))
  err = np.max((U_RB - U).norm())
  ```

---

## 7. **Troubleshooting**
- [ ] **Module Not Found**: Verify `sys.path` and compilation steps.
- [ ] **Version Conflicts**: Check pyMOR/solver compatibility.
- [ ] **Memory Access Issues**: Ensure `WrappedVector` correctly implements the buffer protocol.
- [ ] **Numerical Instability**: Adjust solver parameters (e.g., time-stepping scheme, grid resolution).

---