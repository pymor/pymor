#ifndef DISCRETIZATION_HH
#define DISCRETIZATION_HH

#include <vector>


class Vector {
  friend class DiffusionOperator;
public:
  Vector(int dim, double value);
  Vector(const Vector& other);
  const int dim;
  void scal(double val);
  void axpy(double a, const Vector& x);
  double inner(const Vector& other) const;
  double* data();
private:
  std::vector<double> _data;
};


class DiffusionOperator {
public:
  DiffusionOperator(int n, double left, double right);
  const int dim_source;
  const int dim_range;
  void apply(const Vector& U, Vector& R) const;
private:
  const double h;
  const double left;
  const double right;
};


#endif // DISCRETIZATION_HH
