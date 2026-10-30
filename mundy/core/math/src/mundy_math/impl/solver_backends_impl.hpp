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
#include <stdexcept>
#include <type_traits>
#include <utility>

// KokkosKernels
#include <MundyMath_config.hpp>  // for HAVE_MUNDYMATH_*
#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
#include <KokkosSparse_CrsMatrix.hpp>  // for KokkosSparse::is_crs_matrix_v
#endif

// Mundy
#include <mundy_math/NumTraits.hpp>  // for mundy::ValidScalarType
#include <mundy_utils/requires.hpp>
#include <mundy_utils/throw_assert.hpp>
#include <mundy_utils/tuple.hpp>  // for mundy::tuple

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

//! \name Vector expression machinery
//@{

/// \brief Whether E carries the vector expression marker.
template <class E>
concept HasVectorExprMarker = requires { requires std::remove_cvref_t<E>::is_vector_expression; };

/// \brief Whether Vector's entries can be read as x(i).
template <class Vector>
concept HasIndexedEntries = requires(const std::remove_cvref_t<Vector>& x, size_t i) { x(i); };

/// \brief Whether E's size is a compile-time constant.
template <class E>
concept HasStaticSize = requires {
  { std::remove_cvref_t<E>::static_size() } -> std::convertible_to<size_t>;
};

/// \brief A scalar applied to every entry of an expression.
template <class S>
concept EntrywiseScalar = !HasVectorExprMarker<S> && ValidScalarType<std::remove_cvref_t<S>>;

/// \brief The operands of an entrywise binary operation: two expressions, or an expression and a scalar.
template <class A, class B>
concept EntrywiseOperands = (HasVectorExprMarker<A> && (HasVectorExprMarker<B> || EntrywiseScalar<B>)) ||
                            (HasVectorExprMarker<B> && EntrywiseScalar<A>);

/// \brief Entry i of an operand: e(i) for an expression, the scalar itself otherwise.
template <class Operand>
KOKKOS_INLINE_FUNCTION auto operand_entry(const Operand& operand, size_t i) {
  if constexpr (HasVectorExprMarker<Operand>) {
    return operand(i);
  } else {
    return operand;
  }
}

template <class Operand>
using operand_value_t = decltype(operand_entry(std::declval<const Operand&>(), size_t{}));

/// \brief The operand of an entrywise unary expression.
template <class A>
class UnaryOperand {
 public:
  KOKKOS_INLINE_FUNCTION explicit UnaryOperand(const A& a) : a_(a) {
  }

  KOKKOS_INLINE_FUNCTION size_t size() const {
    return a_.size();
  }

  KOKKOS_INLINE_FUNCTION static constexpr size_t static_size() MUNDY_REQUIRES(HasStaticSize<A>) {
    return A::static_size();
  }

 protected:
  KOKKOS_INLINE_FUNCTION auto operand(size_t i) const {
    return a_(i);
  }

 private:
  A a_;
};

/// \brief The operands of an entrywise binary expression: two expressions of one size, or an expression and a scalar.
template <class A, class B>
class BinaryOperands {
  using sized_t = std::conditional_t<HasVectorExprMarker<A>, A, B>;

 public:
  KOKKOS_INLINE_FUNCTION BinaryOperands(const A& a, const B& b) : a_(a), b_(b) {
    if constexpr (HasStaticSize<A> && HasStaticSize<B>) {
      static_assert(A::static_size() == B::static_size(), "Entrywise operands must have the same size.");
    } else if constexpr (HasVectorExprMarker<A> && HasVectorExprMarker<B>) {
      MUNDY_THROW_ASSERT(a.size() == b.size(), std::invalid_argument, "Entrywise operands must have the same size.");
    }
  }

  KOKKOS_INLINE_FUNCTION size_t size() const {
    if constexpr (HasVectorExprMarker<A>) {
      return a_.size();
    } else {
      return b_.size();
    }
  }

  KOKKOS_INLINE_FUNCTION static constexpr size_t static_size() MUNDY_REQUIRES(HasStaticSize<sized_t>) {
    return sized_t::static_size();
  }

 protected:
  KOKKOS_INLINE_FUNCTION auto left(size_t i) const {
    return operand_entry(a_, i);
  }

  KOKKOS_INLINE_FUNCTION auto right(size_t i) const {
    return operand_entry(b_, i);
  }

 private:
  A a_;
  B b_;
};

/// \brief A source of an assignment or reduction as an expression: a scalar or an expression as is, a vector's
/// entries otherwise.
template <class Backend, class Source>
KOKKOS_INLINE_FUNCTION auto as_source(const Source& source) {
  if constexpr (HasVectorExprMarker<Source> || EntrywiseScalar<Source>) {
    return source;
  } else {
    return Backend::entries(source);
  }
}

/// \brief Whether an argument of an assignment or reduction has n entries; a scalar fits any n.
template <class Backend, class Arg>
KOKKOS_INLINE_FUNCTION bool has_size(const Arg& arg, size_t n) {
  if constexpr (HasVectorExprMarker<Arg>) {
    return arg.size() == n;
  } else if constexpr (EntrywiseScalar<Arg>) {
    return true;
  } else {
    return Backend::size(arg) == n;
  }
}

/// \brief Entry i of each (target, source) pair: every source is evaluated before any target is written.
template <class Backend>
KOKKOS_INLINE_FUNCTION void assign_entry(size_t) {
}
//
template <class Backend, class Target, class Source, class... TargetsAndSources>
KOKKOS_INLINE_FUNCTION void assign_entry(size_t i, Target& target, const Source& source,
                                         TargetsAndSources&... targets_and_sources) {
  const auto value = operand_entry(as_source<Backend>(source), i);
  assign_entry<Backend>(i, targets_and_sources...);
  Backend::vector_data(target, i) = value;
}

/// \brief assign_entry at every index of a parallel loop, over (target, source) pairs held in order.
template <class Backend, class TargetsAndSources>
struct AssignFunctor {
  TargetsAndSources targets_and_sources;

  KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
    assign_at(i, std::make_index_sequence<TargetsAndSources::size()>{});
  }

 private:
  template <size_t... K>
  KOKKOS_INLINE_FUNCTION void assign_at(size_t i, std::index_sequence<K...>) const {
    assign_entry<Backend>(i, get<K>(targets_and_sources)...);
  }
};

/// \brief Joins entry i of each source into its reducer's partial value, in a parallel reduction.
template <class Reducers, class Sources>
struct ReduceFunctor;
//
template <class... Reducers, class... Sources>
struct ReduceFunctor<::mundy::tuple<Reducers...>, ::mundy::tuple<Sources...>> {
  ::mundy::tuple<Reducers...> reducers;
  ::mundy::tuple<Sources...> sources;

  KOKKOS_INLINE_FUNCTION void operator()(const size_t i, typename Reducers::value_type&... values) const {
    join_at(i, std::index_sequence_for<Sources...>{}, values...);
  }

 private:
  template <size_t... K, class... Values>
  KOKKOS_INLINE_FUNCTION void join_at(size_t i, std::index_sequence<K...>, Values&... values) const {
    (get<K>(reducers).join(values, get<K>(sources)(i)), ...);
  }
};

/// \brief Each (reducer, source) pair reduced over the source's compile-time index set, into the reducer's result.
template <class Backend>
KOKKOS_INLINE_FUNCTION void static_reduce() {
}
//
template <class Reducer, class Expression, size_t... Is>
KOKKOS_INLINE_FUNCTION void static_join(const Reducer& reducer, typename Reducer::value_type& value,
                                        const Expression& expression, std::index_sequence<Is...>) {
  (reducer.join(value, expression(Is)), ...);
}
//
template <class Backend, class Reducer, class Source, class... ReducersAndSources>
KOKKOS_INLINE_FUNCTION void static_reduce(const Reducer& reducer, const Source& source,
                                          const ReducersAndSources&... reducers_and_sources) {
  const auto expression = as_source<Backend>(source);
  typename Reducer::value_type value;
  reducer.init(value);
  static_join(reducer, value, expression,
              std::make_index_sequence<std::remove_cvref_t<decltype(expression)>::static_size()>{});
  reducer.reference() = value;
  static_reduce<Backend>(reducers_and_sources...);
}
//@}

}  // namespace impl

}  // namespace mundy

#endif  // MUNDY_MATH_IMPL_SOLVER_BACKENDS_IMPL_HPP_
