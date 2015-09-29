#include "discretization.hh"

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

double Vector::dot(const Vector& other) const {
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
