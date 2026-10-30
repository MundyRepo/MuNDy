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

#ifndef MUNDY_MATH_IMPL_CONVEX_SPACES_IMPL_HPP_
#define MUNDY_MATH_IMPL_CONVEX_SPACES_IMPL_HPP_

// Kokkos:
#include <Kokkos_Core.hpp>

// C++ core:
#include <concepts>

// Mundy
#include <mundy_math/NumTraits.hpp>  // for mundy::NumTraits
#include <mundy_math/cmath.hpp>      // for mundy::abs

namespace mundy {

namespace impl {

/// \brief Whether Space gives its geometry one entry at a time: project, projected_gradient, free_part, and
/// step_to_boundary on its value_type.
template <class Space>
concept HasEntryGeometry = requires(const Space& space, typename Space::value_type v) {
  { project(space, v) } -> std::convertible_to<typename Space::value_type>;
  { projected_gradient(space, v, v) } -> std::convertible_to<typename Space::value_type>;
  { free_part(space, v, v) } -> std::convertible_to<typename Space::value_type>;
  { step_to_boundary(space, v, v) } -> std::convertible_to<typename Space::value_type>;
};

/// \brief Whether x lies on the bound b, to within epsilon.
template <class Scalar>
KOKKOS_INLINE_FUNCTION constexpr bool on_bound(Scalar x, Scalar b) {
  return abs(x - b) <= NumTraits<Scalar>::epsilon();
}

}  // namespace impl

}  // namespace mundy

#endif  // MUNDY_MATH_IMPL_CONVEX_SPACES_IMPL_HPP_
