// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                           Copyright 2024 Flatiron Institute
//                                                 Author: Bryce Palmer
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
#include <mundy_math/impl/minimize_impl.hpp>

namespace mundy {

namespace math {

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

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_MINIMIZE_HPP_
