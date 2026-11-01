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

#ifndef MUNDY_MATH_IMPL_LINEAR_OPS_IMPL_HPP_
#define MUNDY_MATH_IMPL_LINEAR_OPS_IMPL_HPP_

// Kokkos:
#include <Kokkos_Core.hpp>

// C++ core:
#include <cstddef>
#include <utility>

// Mundy
#include <mundy_utils/storage.hpp>  // for mundy::storage, mundy::store
#include <mundy_utils/tuple.hpp>    // for mundy::tuple

namespace mundy {

namespace impl {

/// \brief The workspaces of a composite operator's sub-operators.
///
/// A composite operator's Workspace derives from this for its nested per-sub-operator workspaces, so it only needs to
/// declare its own scratch-vector storage.
template <class... ChildWorkspace>
class ChildWorkspaces {
 public:
  KOKKOS_INLINE_FUNCTION explicit ChildWorkspaces(ChildWorkspace&&... children)
      : children_(::mundy::store(std::forward<ChildWorkspace>(children))...) {
  }

 protected:
  template <size_t I>
  KOKKOS_INLINE_FUNCTION auto& child() {
    return children_.template get<I>().get();
  }
  template <size_t I>
  KOKKOS_INLINE_FUNCTION const auto& child() const {
    return children_.template get<I>().get();
  }

 private:
  ::mundy::tuple<::mundy::storage<ChildWorkspace>...> children_;
};

}  // namespace impl

}  // namespace mundy

#endif  // MUNDY_MATH_IMPL_LINEAR_OPS_IMPL_HPP_
