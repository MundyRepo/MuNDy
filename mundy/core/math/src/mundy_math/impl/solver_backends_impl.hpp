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

#ifndef MUNDY_MATH_IMPL_SOLVER_BACKENDS_IMPL_HPP_
#define MUNDY_MATH_IMPL_SOLVER_BACKENDS_IMPL_HPP_

// Kokkos:
#include <Kokkos_Core.hpp>

// C++ core:
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

// KokkosKernels
#include <MundyMath_config.hpp>  // for HAVE_MUNDYMATH_*
#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
#include <KokkosSparse_CrsMatrix.hpp>  // for KokkosSparse::is_crs_matrix_v
#endif

namespace mundy {

namespace impl {

/// \brief A dense matrix in Kokkos::View<double**> or Kokkos::View<const double**> form.
template <class Op>
concept DenseMatView = Kokkos::is_view_v<Op> && requires {
  // These are checked in constraint context:
  { std::remove_reference_t<Op>::rank } -> std::convertible_to<int>;
  typename std::remove_reference_t<Op>::non_const_value_type;
} && (std::remove_reference_t<Op>::rank == 2);

/// \brief A sparse matrix in KokkosSparse::CrsMatrix (compressed row storage) form.
template <class Op>
#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
concept SparseMatView = KokkosSparse::is_crs_matrix_v<std::remove_cvref_t<Op>>;
#else
concept SparseMatView = false;
#endif

/// \brief A dense or sparse matrix view, as opposed to an operator that applies itself.
template <class Op>
concept MatView = DenseMatView<Op> || SparseMatView<Op>;

/// \brief The number of rows of a matrix view.
template <MatView Matrix>
KOKKOS_INLINE_FUNCTION size_t mat_view_num_rows(const Matrix& A) {
  if constexpr (DenseMatView<Matrix>) {
    return A.extent(0);
  } else {
    return static_cast<size_t>(A.numRows());
  }
}

/// \brief The number of columns of a matrix view.
template <MatView Matrix>
KOKKOS_INLINE_FUNCTION size_t mat_view_num_cols(const Matrix& A) {
  if constexpr (DenseMatView<Matrix>) {
    return A.extent(1);
  } else {
    return static_cast<size_t>(A.numCols());
  }
}

template <class Op, typename X, typename Y>
concept HasApplyMember = requires(const Op& op, const X& x, Y& y) {
  { op.apply(x, y) } -> std::same_as<void>;
};

template <class Op, typename X, typename Y, typename Workspace>
concept HasApplyMemberWithWorkspace = requires(const Op& op, const X& x, Y& y, Workspace& workspace) {
  { op.apply(x, y, workspace) } -> std::same_as<void>;
};

template <class Op>
concept HasDomainSizeMember = requires(const Op& op) {
  { op.domain_size() } -> std::convertible_to<size_t>;
};

template <class Op>
concept HasRangeSizeMember = requires(const Op& op) {
  { op.range_size() } -> std::convertible_to<size_t>;
};

template <class Op>
concept HasMakeDomainVectorMember = requires(const Op& op) {
  { op.make_domain_vector() };
};

template <class Op>
concept HasMakeRangeVectorMember = requires(const Op& op) {
  { op.make_range_vector() };
};

template <class Workspace>
concept HasCommitMember = requires(Workspace& workspace) {
  { workspace.commit() } -> std::same_as<void>;
};

template <class Workspace>
concept HasInvalidateMember = requires(Workspace& workspace) {
  { workspace.invalidate() } -> std::same_as<void>;
};

template <class Workspace>
concept HasIsCommittedMember = requires(const Workspace& workspace) {
  { workspace.is_committed() } -> std::convertible_to<bool>;
};

template <class Workspace>
KOKKOS_INLINE_FUNCTION void workspace_commit(Workspace& workspace) {
  if constexpr (HasCommitMember<Workspace>) {
    workspace.commit();
  }
}

template <class Workspace>
KOKKOS_INLINE_FUNCTION void workspace_invalidate(Workspace& workspace) {
  if constexpr (HasInvalidateMember<Workspace>) {
    workspace.invalidate();
  }
}

template <class Workspace>
KOKKOS_INLINE_FUNCTION bool workspace_is_committed(const Workspace& workspace) {
  if constexpr (HasIsCommittedMember<Workspace>) {
    return workspace.is_committed();
  } else {
    return true;
  }
}

/// \brief The workspace for an operator with no scratch state of its own: commit()/invalidate() just track a
/// single committed flag, with no children to propagate to.
struct NoWorkspace {
  KOKKOS_INLINE_FUNCTION NoWorkspace() = default;

  // clang-format off
  KOKKOS_INLINE_FUNCTION void commit() { committed_ = true; }
  KOKKOS_INLINE_FUNCTION void invalidate() { committed_ = false; }
  KOKKOS_INLINE_FUNCTION bool is_committed() const { return committed_; }
  // clang-format on

 private:
  bool committed_{false};
};

template <class Op>
concept HasMakeWorkspaceNoArgMember = requires(const Op& op) {
  { op.make_workspace() };
};

/// \brief The workspace for op: op.make_workspace() if op provides one, else NoWorkspace.
template <class LinearOp>
KOKKOS_INLINE_FUNCTION auto make_workspace(LinearOp&& op) {
  if constexpr (HasMakeWorkspaceNoArgMember<LinearOp>) {
    return std::forward<LinearOp>(op).make_workspace();
  } else {
    return NoWorkspace();
  }
}

template <class LinearOp>
using workspace_for_t = decltype(make_workspace(std::declval<LinearOp>()));

template <class Vector>
using vector_value_type =
    std::remove_cvref_t<decltype(std::declval<const std::remove_reference_t<Vector>&>()(size_t{}))>;

}  // namespace impl

}  // namespace mundy

#endif  // MUNDY_MATH_IMPL_SOLVER_BACKENDS_IMPL_HPP_
