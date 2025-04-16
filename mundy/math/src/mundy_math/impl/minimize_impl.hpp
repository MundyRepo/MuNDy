// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                              Copyright 2024 Bryce Palmer
// 
// Developed under support from the NSF Graduate Research Fellowship Program.
//
// Mundy is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
//
// Mundy is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with Mundy. If not, see
// <https://www.gnu.org/licenses/>.
//
// **********************************************************************************************************************
// @HEADER

#ifndef MUNDY_MATH_IMPL_MINIMIZE_IMPL_HPP_
#define MUNDY_MATH_IMPL_MINIMIZE_IMPL_HPP_

// External
#include <Kokkos_Core.hpp>

// C++ core
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>

// Mundy
#include <mundy_math/Vector.hpp>  // for mundy::math::Vector

namespace mundy {

namespace math {

namespace impl {

template <typename T1, typename T2, typename T3>
KOKKOS_INLINE_FUNCTION T3 put_in_range(T1 min_val, T2 max_val, T3 val) {
  return (val < min_val) ? min_val : (val > max_val) ? max_val : val;
}

template <typename T>
KOKKOS_INLINE_FUNCTION bool is_subnormal(T value) {
  return (value != 0) && (Kokkos::abs(value) < Kokkos::Experimental::norm_min_v<T>);
}

KOKKOS_INLINE_FUNCTION double poly_min_extrap(const double f0, const double d0, const double f1, const double d1,
                                              const double limit = 1) {
  const double n = 3 * (f1 - f0) - 2 * d0 - d1;
  const double e = d0 + d1 - 2 * (f1 - f0);

  // find the minimum of the derivative of the polynomial

  const double temp_sqr = Kokkos::max(n * n - 3 * e * d0, 0.0);
  if (temp_sqr < 0) {
    return 0.5;
  }
  if (Kokkos::abs(e) <= Kokkos::Experimental::epsilon_v<double>) {
    return 0.5;
  }

  // figure out the two possible min values
  const double temp = Kokkos::sqrt(temp_sqr);
  const double x1 = (temp - n) / (3 * e);
  const double x2 = -(temp + n) / (3 * e);

  // compute the value of the interpolating polynomial at these two points
  const double y1 = f0 + d0 * x1 + n * x1 * x1 + e * x1 * x1 * x1;
  const double y2 = f0 + d0 * x2 + n * x2 * x2 + e * x2 * x2 * x2;

  // pick the best point
  const double x = (y1 < y2) ? x1 : x2;

  // now make sure the minimum is within the allowed range of [0,limit]
  return put_in_range(0, limit, x);
}

KOKKOS_INLINE_FUNCTION double poly_min_extrap(const double f0, const double d0, const double f1) {
  const double temp = 2 * (f1 - f0 - d0);
  if (Kokkos::abs(temp) <= d0 * Kokkos::Experimental::epsilon_v<double>) {
    return 0.5;
  }

  const double alpha = -d0 / temp;

  // now make sure the minimum is within the allowed range of (0,1)
  return put_in_range(0, 1, alpha);
}

KOKKOS_INLINE_FUNCTION double poly_min_extrap(const double f0, const double d0, const double x1, const double f_x1,
                                              const double x2, const double f_x2) {
  // The contents of this function follow the equations described on page 58 of the
  // book Numerical Optimization by Nocedal and Wright, second edition.
  const double aa2 = x2 * x2;
  const double aa1 = x1 * x1;
  const double m11 = aa2;
  const double m12 = -aa1;
  const double m21 = -aa2 * x2;
  const double m22 = aa1 * x1;
  const double v1 = f_x1 - f0 - d0 * x1;
  const double v2 = f_x2 - f0 - d0 * x2;

  const double temp1 = aa2 * aa1 * (x1 - x2);

  // just take a guess if this happens
  if (temp1 == 0 || is_subnormal(temp1)) {
    return x1 / 2.0;
  }

  const double inv_temp1 = 1.0 / temp1;
  const double a = (m11 * v1 + m12 * v2) * inv_temp1;
  const double b = (m21 * v1 + m22 * v2) * inv_temp1;
  const double temp2 = b * b - 3 * a * d0;
  if (temp2 < 0 || a == 0) {
    // This is probably a line so just pick the lowest point
    if (f0 < f_x2)
      return 0;
    else
      return x2;
  }
  const double temp3 = (-b + Kokkos::sqrt(temp2)) / (3 * a);
  return put_in_range(0, x2, temp3);
}

template <size_t N, typename CostFunctionType>
class line_search_funct {
 public:
  KOKKOS_FUNCTION
  line_search_funct(const CostFunctionType& const_func, const Vector<double, N>& start,
                    const Vector<double, N>& direction)
      : const_func_(const_func), start_(start), direction_(direction) {
  }

  KOKKOS_FUNCTION
  double operator()(const double& x) const {
    return const_func_(start_ + x * direction_);
  }

 private:
  const CostFunctionType& const_func_;
  const Vector<double, N>& start_;
  const Vector<double, N>& direction_;
};

class objective_delta_stop_strategy {
 public:
  KOKKOS_FUNCTION
  explicit objective_delta_stop_strategy(const double min_delta = 1e-7)
      : been_used_(false), min_delta_(min_delta), max_iter_(0), cur_iter_(0), prev_funct_value_(0) {
  }

  KOKKOS_FUNCTION
  objective_delta_stop_strategy(double min_delta, size_t max_iter)
      : been_used_(false), min_delta_(min_delta), max_iter_(max_iter), cur_iter_(0), prev_funct_value_(0) {
  }

  template <size_t N>
  KOKKOS_FUNCTION bool should_continue_search(const Vector<double, N>&, const double funct_value,
                                              const Vector<double, N>&) {
    ++cur_iter_;
    if (been_used_) {
      // Check if we have hit the max allowable number of iterations.  (but only
      // check if max_iter_ is enabled (i.e. not 0)).
      if (max_iter_ != 0 && cur_iter_ > max_iter_) {
        return false;
      }

      // check if the function change was too small
      if (Kokkos::abs(funct_value - prev_funct_value_) < min_delta_) {
        return false;
      }
    }

    been_used_ = true;
    prev_funct_value_ = funct_value;
    return true;
  }

 private:
  bool been_used_;
  double min_delta_;
  size_t max_iter_;
  size_t cur_iter_;
  double prev_funct_value_;
};

template <typename CostFunctionType>
class central_differences {
 public:
  KOKKOS_FUNCTION
  explicit central_differences(const CostFunctionType& const_func, double eps = 1e-7)
      : const_func_(const_func), eps_(eps) {
  }

  template <size_t N>
  KOKKOS_FUNCTION Vector<double, N> operator()(const Vector<double, N>& x) const {
    Vector<double, N> der;
    Vector<double, N> e = x;
    for (size_t i = 0; i < N; ++i) {
      const double old_val = e[i];

      e[i] += eps_;
      const double delta_plus = const_func_(e);
      e[i] = old_val - eps_;
      const double delta_minus = const_func_(e);

      der[i] = (delta_plus - delta_minus) / ((old_val + eps_) - (old_val - eps_));

      // and finally restore the old value of this element
      e[i] = old_val;
    }

    return der;
  }

  KOKKOS_FUNCTION
  double operator()(const double& x) const {
    return (const_func_(x + eps_) - const_func_(x - eps_)) / ((x + eps_) - (x - eps_));
  }

 private:
  const CostFunctionType& const_func_;
  const double eps_;
};

template <typename CostFunctionType, typename CostFunctionDerType>
KOKKOS_FUNCTION double line_search(const CostFunctionType& f, const double f0, const CostFunctionDerType& der,
                                   const double d0, const double rho, const double sigma, const double min_f,
                                   const size_t max_iter) {
  // The bracketing phase of this function is implemented according to block 2.6.2 from
  // the book Practical Methods of Optimization by R. Fletcher.   The sectioning
  // phase is an implementation of 2.6.4 from the same book.

  // 1 <= tau1a < tau1b. Controls the alpha jump size during the bracketing phase of
  // the search.
  const double tau1a = 1.4;
  const double tau1b = 9;

  // it must be the case that 0 < tau2 < tau3 <= 1/2 for the algorithm to function
  // correctly but the specific values of tau2 and tau3 aren't super important.
  const double tau2 = 1.0 / 10.0;
  const double tau3 = 1.0 / 2.0;

  // Stop right away and return a step size of 0 if the gradient is 0 at the starting point
  if (Kokkos::abs(d0) <= Kokkos::abs(f0) * Kokkos::Experimental::epsilon_v<double>) {
    return 0;
  }

  // Stop right away if the current value is good enough according to min_f
  if (f0 <= min_f) {
    return 0;
  }

  // Figure out a reasonable upper bound on how large alpha can get.
  const double mu = (min_f - f0) / (rho * d0);
  double alpha = 1;
  if (mu < 0) {
    alpha = -alpha;
  }
  alpha = put_in_range(0, 0.65 * mu, alpha);

  double last_alpha = 0;
  double last_val = f0;
  double last_val_der = d0;

  // The bracketing stage will find a range of points [a,b]
  // that contains a reasonable solution to the line search
  double a, b;

  // These variables will hold the values and derivatives of f(a) and f(b)
  double a_val, b_val, a_val_der, b_val_der;

  // This thresh value represents the Wolfe curvature condition
  const double thresh = Kokkos::abs(sigma * d0);

  size_t itr = 0;
  // do the bracketing stage to find the bracket range [a,b]
  while (true) {
    ++itr;
    const double val = f(alpha);
    const double val_der = der(alpha);

    // we are done with the line search since we found a value smaller
    // than the minimum f value
    if (val <= min_f) {
      return alpha;
    }

    if (val > f0 + rho * alpha * d0 || val >= last_val) {
      a_val = last_val;
      a_val_der = last_val_der;
      b_val = val;
      b_val_der = val_der;

      a = last_alpha;
      b = alpha;
      break;
    }

    if (Kokkos::abs(val_der) <= thresh) {
      return alpha;
    }

    // if we are stuck not making progress then quit with the current alpha
    if (last_alpha == alpha || itr >= max_iter) {
      return alpha;
    }

    if (val_der >= 0) {
      a_val = val;
      a_val_der = val_der;
      b_val = last_val;
      b_val_der = last_val_der;

      a = alpha;
      b = last_alpha;
      break;
    }

    const double temp = alpha;
    // Pick a larger range [first, last].  We will pick the next alpha in that
    // range.
    double first, last;
    if (mu > 0) {
      first = Kokkos::min(mu, alpha + tau1a * (alpha - last_alpha));
      last = Kokkos::min(mu, alpha + tau1b * (alpha - last_alpha));
    } else {
      first = Kokkos::max(mu, alpha + tau1a * (alpha - last_alpha));
      last = Kokkos::max(mu, alpha + tau1b * (alpha - last_alpha));
    }

    // pick a point between first and last by doing some kind of interpolation
    if (last_alpha < alpha) {
      alpha = last_alpha + (alpha - last_alpha) * poly_min_extrap(last_val, last_val_der, val, val_der, 1e10);
    } else {
      alpha = alpha + (last_alpha - alpha) * poly_min_extrap(val, val_der, last_val, last_val_der, 1e10);
    }

    alpha = put_in_range(first, last, alpha);

    last_alpha = temp;

    last_val = val;
    last_val_der = val_der;
  }

  // Now do the sectioning phase from 2.6.4
  while (true) {
    ++itr;
    double first = a + tau2 * (b - a);
    double last = b - tau3 * (b - a);

    // use interpolation to pick alpha between first and last
    alpha = a + (b - a) * poly_min_extrap(a_val, a_val_der, b_val, b_val_der);
    alpha = put_in_range(first, last, alpha);

    const double val = f(alpha);
    const double val_der = der(alpha);

    // we are done with the line search since we found a value smaller
    // than the minimum f value or we ran out of iterations.
    if (val <= min_f || itr >= max_iter) {
      return alpha;
    }

    // stop if the interval gets so small that it isn't shrinking any more due to rounding error
    if (a == first || b == last) {
      return b;
    }

    // If alpha has basically become zero then just stop.  Think of it like this,
    // if we take the largest possible alpha step will the objective function
    // change at all?  If not then there isn't any point looking for a better
    // alpha.
    const double max_possible_alpha = Kokkos::max(Kokkos::abs(a), Kokkos::abs(b));
    if (Kokkos::abs(max_possible_alpha * d0) <= Kokkos::abs(f0) * Kokkos::Experimental::epsilon_v<double>) {
      return alpha;
    }

    if (val > f0 + rho * alpha * d0 || val >= a_val) {
      b = alpha;
      b_val = val;
      b_val_der = val_der;
    } else {
      if (Kokkos::abs(val_der) <= thresh) {
        return alpha;
      }

      if ((b - a) * val_der >= 0) {
        b = a;
        b_val = a_val;
        b_val_der = a_val_der;
      }

      a = alpha;
      a_val = val;
      a_val_der = val_der;
    }
  }
}

template <size_t max_size, size_t N>
class lbfgs_search_strategy {
 public:
  KOKKOS_FUNCTION
  lbfgs_search_strategy()
      : data(),
        alpha(0.0),
        been_used(false),
        current_size(0),
        prev_x(),
        prev_derivative(),
        prev_direction(),
        dh_temp() {
    static_assert(max_size > 0, "max_size must be greater than 0");
  }

  KOKKOS_FUNCTION
  lbfgs_search_strategy(const lbfgs_search_strategy& item)
      : data(item.data),
        alpha(item.alpha),
        been_used(item.been_used),
        current_size(item.current_size),
        prev_x(item.prev_x),
        prev_derivative(item.prev_derivative),
        prev_direction(item.prev_direction),
        dh_temp(item.dh_temp) {
  }

  KOKKOS_FUNCTION
  static constexpr double get_wolfe_rho() {
    return 0.01;
  }

  KOKKOS_FUNCTION
  static constexpr double get_wolfe_sigma() {
    return 0.9;
  }

  KOKKOS_FUNCTION
  static constexpr size_t get_max_line_search_iterations() {
    return 100;
  }

  KOKKOS_FUNCTION
  const Vector<double, N>& get_next_direction(const Vector<double, N>& x, const double,
                                              const Vector<double, N>& funct_derivative) {
    prev_direction = -funct_derivative;

    if (!been_used) {
      been_used = true;
    } else {
      // add an element into the stored data sequence
      dh_temp.s = x - prev_x;
      dh_temp.y = funct_derivative - prev_derivative;
      double temp = dot(dh_temp.s, dh_temp.y);
      // only accept this bit of data if temp isn't zero
      if (Kokkos::abs(temp) > Kokkos::Experimental::epsilon_v<double>) {
        dh_temp.rho = 1.0 / temp;
        if (current_size < max_size) {
          data[current_size++] = dh_temp;
        } else {
          // remove the oldest element and add the new one at the end
          rotate_data();
          data[max_size - 1] = dh_temp;
        }
      } else {
        current_size = 0;
      }

      if (current_size > 0) {
        // This block of code is from algorithm 7.4 in the Nocedal book.
        alpha.fill(0.0);
        for (size_t i = current_size - 1; i < current_size; --i) {
          alpha[i] = data[i].rho * dot(data[i].s, prev_direction);
          prev_direction = prev_direction - alpha[i] * data[i].y;
        }

        // Take a guess at what the first H matrix should be.
        double H_0 = 1.0 / data[current_size - 1].rho / dot(data[current_size - 1].y, data[current_size - 1].y);
        H_0 = put_in_range(0.001, 1000.0, H_0);
        prev_direction = H_0 * prev_direction;

        for (size_t i = 0; i < current_size; ++i) {
          double beta = data[i].rho * dot(data[i].y, prev_direction);
          prev_direction = prev_direction + (alpha[i] - beta) * data[i].s;
        }
      }
    }

    prev_x = x;
    prev_derivative = funct_derivative;
    return prev_direction;
  }

 private:
  struct data_helper {
    Vector<double, N> s;
    Vector<double, N> y;
    double rho;

    /// \brief Default constructor
    KOKKOS_FUNCTION
    data_helper() : s(), y(), rho(0.0) {
    }

    /// \brief Deep copy constructor
    KOKKOS_FUNCTION
    data_helper(const data_helper& item) : s(item.s), y(item.y), rho(item.rho) {
    }

    /// \brief Deep move constructor
    KOKKOS_FUNCTION
    data_helper(data_helper&& item) : s(std::move(item.s)), y(std::move(item.y)), rho(item.rho) {
    }

    /// \brief Deep copy assignment operator
    KOKKOS_FUNCTION
    data_helper& operator=(const data_helper& item) {
      s = item.s;
      y = item.y;
      rho = item.rho;
      return *this;
    }

    /// \brief Deep move assignment operator
    KOKKOS_FUNCTION
    data_helper& operator=(data_helper&& item) {
      s = std::move(item.s);
      y = std::move(item.y);
      rho = item.rho;
      return *this;
    }

    /// \brief Destructor
    KOKKOS_FUNCTION
    ~data_helper() {
    }
  };

  Array<data_helper, max_size> data;
  Vector<double, max_size> alpha;

  bool been_used;
  size_t current_size;
  Vector<double, N> prev_x;
  Vector<double, N> prev_derivative;
  Vector<double, N> prev_direction;
  data_helper dh_temp;

  KOKKOS_FUNCTION
  void rotate_data() {
    data_helper& first = data[0];
    for (size_t i = 1; i < max_size; ++i) {
      data[i - 1] = data[i];
    }
    data[max_size - 1] = first;
  }
};

template <size_t max_size, size_t N, typename search_strategy_type, typename stop_strategy_type,
          typename CostFunctionType>
KOKKOS_FUNCTION double find_min_using_approximate_derivatives(search_strategy_type search_strategy,
                                                              stop_strategy_type stop_strategy,
                                                              const CostFunctionType& cost_func, Vector<double, N>& x,
                                                              const double min_alowable_cost,
                                                              const double derivative_eps = 1e-7) {
  Vector<double, N> g;
  Vector<double, N> s;

  double cost = cost_func(x);
  const auto derivative_func = central_differences(cost_func, derivative_eps);
  g = derivative_func(x);
  while (stop_strategy.should_continue_search(x, cost, g) && cost > min_alowable_cost) {
    s = search_strategy.get_next_direction(x, cost, g);

    double alpha =
        line_search(line_search_funct(cost_func, x, s), cost,
                    central_differences(line_search_funct(cost_func, x, s), derivative_eps),
                    dot(g, s),  // Sometimes the following line is a better way of determining the initial gradient.
                    search_strategy.get_wolfe_rho(), search_strategy.get_wolfe_sigma(), min_alowable_cost,
                    search_strategy.get_max_line_search_iterations());

    // Take the search step indicated by the above line search
    x = alpha * s + x;
    g = derivative_func(x);
    cost = cost_func(x);
  }

  return cost;
}

}  // namespace impl

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_IMPL_MINIMIZE_IMPL_HPP_
