#include "model.hh"

#include <assert.h>
#include <iostream>
#include <cmath>

Vector::Vector(int dim, double value) : _data(dim, value), dim(dim) {}

Vector::Vector(const Vector& other) : _data(other._data), dim(other.dim) {}

void Vector::scal(double val) {
  for (int i = 0; i < dim; i++) {
    _data[i] *= val;
  }
}

void Vector::axpy(double a, const Vector& x) {
  assert(x.dim == dim);
  for (int i = 0; i < dim; i++) {
    _data[i] += a * x._data[i];
  }
}

double Vector::inner(const Vector& other) const {
  assert(other.dim == dim);
  double result = 0;
  for (int i = 0; i < dim; i++) {
    result += _data[i] * other._data[i];
  }
  return result;
}

double* Vector::data() {
  return _data.data();
}

// -------------------------------------------------------

DiffusionOperator::DiffusionOperator(int n, double left, double right)
  : dim_source(n+1), dim_range(n+1), h(1./n), left(left), right(right) {}

void DiffusionOperator::apply(const Vector& U, Vector& R) const {
  const double one_over_h2 = 1. / (h * h);
  for (int i = 1; i < dim_range - 1; i++) {
    if ((i * h > left) && (i * h <= right)) {
      R._data[i] = -(U._data[i-1] - 2*U._data[i] + U._data[i+1]) * one_over_h2;
    }
  }
}

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

PYBIND11_MODULE(model, m)
{
    namespace py = pybind11;

    py::class_<DiffusionOperator> op(m, "DiffusionOperator", "DiffusionOperator Docstring");
    op.def(py::init<int, double, double>());
    op.def_readonly("dim_source", &DiffusionOperator::dim_source);
    op.def_readonly("dim_range", &DiffusionOperator::dim_range);
    op.def("apply", &DiffusionOperator::apply);

    py::class_<Vector> vec(m, "Vector", "Vector Docstring", py::buffer_protocol());
    vec.def(py::init([](int size, double value) { return std::make_unique<Vector>(size, value); }));
    vec.def(py::init([](const Vector& other) { return std::make_unique<Vector>(other); }));

    vec.def_readonly("dim", &Vector::dim);
    vec.def("scal", &Vector::scal);
    vec.def("axpy", &Vector::axpy);
    vec.def("inner", &Vector::inner);
    vec.def("data", &Vector::data);

    vec.def_buffer([](Vector& vec) -> py::buffer_info {
        return py::buffer_info(
            vec.data(), sizeof(double), py::format_descriptor<double>::format(), 1, {vec.dim}, {sizeof(double)});
    });
} // end PYBIND11_MODULE(model, m)
