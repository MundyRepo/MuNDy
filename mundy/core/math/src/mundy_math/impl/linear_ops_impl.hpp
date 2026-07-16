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
#include <mundy_math/impl/solver_backends_impl.hpp>  // for workspace_commit/invalidate/is_committed
#include <mundy_utils/storage.hpp>                   // for mundy::storage, mundy::store
#include <mundy_utils/tuple.hpp>                     // for mundy::tuple, mundy::for_each, mundy::all_of

namespace mundy {

namespace impl {

/// \brief Aggregates commit()/invalidate()/is_committed() over a pack of child workspace storages.
///
/// A composite operator's Workspace derives from this for its nested per-sub-operator workspaces, so it only
/// needs to declare its own scratch-vector storage; the commit/invalidate/is_committed contract is inherited.
template <class... ChildWorkspace>
class CommitGroup {
 public:
  // For an empty ChildWorkspace... pack this is a zero-argument constructor (committed defaults to false).
  KOKKOS_INLINE_FUNCTION explicit CommitGroup(ChildWorkspace&&... children, bool committed = false)
      : children_(::mundy::store(std::forward<ChildWorkspace>(children))...), committed_(committed) {
  }

  KOKKOS_FUNCTION void commit() {
    committed_ = true;
    ::mundy::for_each(children_, [](auto& c) { workspace_commit(c.get()); });
  }

  KOKKOS_FUNCTION void invalidate() {
    committed_ = false;
    ::mundy::for_each(children_, [](auto& c) { workspace_invalidate(c.get()); });
  }

  KOKKOS_FUNCTION bool is_committed() const {
    return committed_ && ::mundy::all_of(children_, [](const auto& c) { return workspace_is_committed(c.get()); });
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
  bool committed_{false};
};

}  // namespace impl

}  // namespace mundy

#endif  // MUNDY_MATH_IMPL_LINEAR_OPS_IMPL_HPP_
