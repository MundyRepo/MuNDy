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

#ifndef MUNDY_MATH_CONVEX_SPACES_HPP_
#define MUNDY_MATH_CONVEX_SPACES_HPP_

// Kokkos:
#include <Kokkos_Core.hpp>

// C++ core:
#include <concepts>
#include <type_traits>

// Mundy
#include <mundy_math/cmath.hpp>  // for mundy::max, mundy::min

namespace mundy {

//! \name Convex spaces
//@{
// 1d convex spaces, applied elementwise to a vector under the assumption of a separable convex space. Each is a
// projection Proj: R -> R onto the feasible set, exposed both as project(x) and operator()(x).

/// \brief Proj(x) = x for all x in R
template <typename Scalar>
struct UnconstrainedSpace {
  using value_type = Scalar;

  KOKKOS_INLINE_FUNCTION
  constexpr value_type operator()(const value_type& x) const {
    return project(x);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr value_type project(const value_type& x) const {
    return x;
  }
};

/// \brief Proj(x) = max(x, lower_bound) for all x in R
template <typename Scalar>
struct LowerBoundSpace {
  using value_type = Scalar;

  value_type lower_bound;

  KOKKOS_INLINE_FUNCTION
  constexpr value_type operator()(const value_type& x) const {
    return project(x);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr value_type project(const value_type& x) const {
    return max(x, lower_bound);
  }
};

/// \brief Proj(x) = min(x, upper_bound) for all x in R
template <typename Scalar>
struct UpperBoundSpace {
  using value_type = Scalar;

  value_type upper_bound;

  KOKKOS_INLINE_FUNCTION
  constexpr value_type operator()(const value_type& x) const {
    return project(x);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr value_type project(const value_type& x) const {
    return min(x, upper_bound);
  }
};

/// \brief Proj(x) = min(max(x, lower_bound), upper_bound) for all x in R
template <typename Scalar>
struct BoundedSpace {
  using value_type = Scalar;

  value_type lower_bound;
  value_type upper_bound;

  KOKKOS_INLINE_FUNCTION
  constexpr value_type operator()(const value_type& x) const {
    return project(x);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr value_type project(const value_type& x) const {
    return min(max(x, lower_bound), upper_bound);
  }
};

/// \brief Concept for a valid convex space: a value_type plus project(x) and operator()(x) that both return it.
template <class Space>
concept ValidConvexSpace = requires {
  typename std::remove_cvref_t<Space>::value_type;
} && requires(const std::remove_cvref_t<Space>& s, typename std::remove_cvref_t<Space>::value_type x) {
  { s.project(x) } -> std::same_as<typename std::remove_cvref_t<Space>::value_type>;
  { s(x) } -> std::same_as<typename std::remove_cvref_t<Space>::value_type>;
};

static_assert(ValidConvexSpace<UnconstrainedSpace<double>>, "UnconstrainedSpace<double> is not a ValidConvexSpace");
static_assert(ValidConvexSpace<LowerBoundSpace<double>>, "LowerBoundSpace<double> is not a ValidConvexSpace");
static_assert(ValidConvexSpace<UpperBoundSpace<double>>, "UpperBoundSpace<double> is not a ValidConvexSpace");
static_assert(ValidConvexSpace<BoundedSpace<double>>, "BoundedSpace<double> is not a ValidConvexSpace");
//@}

}  // namespace mundy

#endif  // MUNDY_MATH_CONVEX_SPACES_HPP_
