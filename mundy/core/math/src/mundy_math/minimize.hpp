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

#ifndef MUNDY_MATH_MINIMIZE_HPP_
#define MUNDY_MATH_MINIMIZE_HPP_

/// \file minimize.hpp
/// \defgroup MundyMathMinimize mundy::minimize(...)
/// \brief Fixed-size, allocation-free L-BFGS minimization helpers for Kokkos code.

// External
#include <Kokkos_Core.hpp>

// C++ core
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>

// Mundy
#include <mundy_math/Vector.hpp>  // for mundy::Vector
#include <mundy_math/impl/minimize_impl.hpp>

namespace mundy {

template <size_t max_size, size_t N, typename CostFunctionType>
KOKKOS_FUNCTION double find_min_using_approximate_derivatives(
    const CostFunctionType& cost_func, Vector<double, N>& x,
    const double min_alowable_cost = -Kokkos::Experimental::infinity_v<double>, const double min_objective_delta = 1e-7,
    const double derivative_eps = 1e-7) {
  auto stop_strategy = impl::objective_delta_stop_strategy(min_objective_delta);
  auto search_strategy = impl::lbfgs_search_strategy<max_size, N>();
  return impl::find_min_using_approximate_derivatives<max_size, N>(search_strategy, stop_strategy, cost_func, x,
                                                                   min_alowable_cost, derivative_eps);
}

/// \brief L-BFGS minimization using a caller-supplied analytical gradient.
///
/// Prefer this over \c find_min_using_approximate_derivatives whenever the exact gradient is
/// available.  The line search uses exact directional derivatives, so each outer L-BFGS
/// iteration costs O(line_search) evaluations of \p cost_func instead of O(2N + line_search).
///
/// \tparam max_size  L-BFGS history depth.
/// \tparam N         Dimensionality of the parameter space.
/// \param cost_func  Callable: \c double(const Vector<double,N>&).
/// \param der_func   Callable: \c Vector<double,N>(const Vector<double,N>&) — exact gradient.
/// \param x          Starting point; overwritten with the minimizer on return.
/// \param min_alowable_cost  Early-exit threshold (stop if cost drops below this value).
/// \param min_objective_delta  Convergence tolerance on the change in objective value.
/// \brief L-BFGS minimization using a single combined cost-and-gradient (FDF) callable.
///
/// Prefer this over \c find_min_with_derivatives when the cost and gradient can be computed
/// from the same intermediate values (e.g.\ foot-points on ellipsoid surfaces).  The combined
/// callable is invoked exactly once per evaluation point in both the outer loop and the line
/// search, eliminating all redundant forward passes.
///
/// \tparam max_size  L-BFGS history depth.
/// \tparam N         Dimensionality of the parameter space.
/// \param fdf        Callable: \c double(const Vector<double,N>&, Vector<double,N>&).
///                   Must fill the second argument with \c ∇f and return \c f(x).
/// \param x          Starting point; overwritten with the minimizer on return.
/// \param min_alowable_cost  Early-exit threshold.
/// \param min_objective_delta  Convergence tolerance on the change in objective value.
template <size_t max_size, size_t N, typename FDFType>
KOKKOS_FUNCTION double find_min_with_fdf(const FDFType& fdf, Vector<double, N>& x,
                         const double min_alowable_cost = -Kokkos::Experimental::infinity_v<double>,
                         const double min_objective_delta = 1e-7) {
  auto stop_strategy = impl::objective_delta_stop_strategy(min_objective_delta);
  auto search_strategy = impl::lbfgs_search_strategy<max_size, N>();
  return impl::find_min_with_fdf<max_size, N>(search_strategy, stop_strategy, fdf, x, min_alowable_cost);
}

template <size_t max_size, size_t N, typename CostFunctionType, typename DerivativeFunctionType>
KOKKOS_FUNCTION double find_min_with_derivatives(
    const CostFunctionType& cost_func, const DerivativeFunctionType& der_func, Vector<double, N>& x,
    const double min_alowable_cost = -Kokkos::Experimental::infinity_v<double>,
    const double min_objective_delta = 1e-7) {
  auto stop_strategy = impl::objective_delta_stop_strategy(min_objective_delta);
  auto search_strategy = impl::lbfgs_search_strategy<max_size, N>();
  return impl::find_min_with_derivatives<max_size, N>(search_strategy, stop_strategy, cost_func, der_func, x,
                                                      min_alowable_cost);
}

}  // namespace mundy

#endif  // MUNDY_MATH_MINIMIZE_HPP_
