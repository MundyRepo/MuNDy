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

#ifndef MUNDY_MATH_CONVEX_HPP_
#define MUNDY_MATH_CONVEX_HPP_

// Kokkos:
#include <Kokkos_Core.hpp>

// C++ core:
#include <functional>
#include <ostream>
#include <type_traits>
#include <utility>

// KokkosKernels
#include <MundyMath_config.hpp>  // for HAVE_MUNDYMATH_*
#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
#include <KokkosBlas.hpp>
#include <KokkosBlas_gesv.hpp>
#endif

// Mundy core:
#include <mundy_core/throw_assert.hpp>

// Mundy math:
#include <mundy_math/Tolerance.hpp>  // for mundy::math::get_zero_tolerance<T>
#include <mundy_math/Vector.hpp>     // for mundy::math::Vector

namespace mundy {

namespace math {

namespace convex {

namespace space {

// These are 1d convex spaces, which will be applied to each element of a vector assuming a separable convex space

/// \brief Proj(x) = x for all x in R
template <typename Scalar>
struct Unconstrained {
  using scalar_t = Scalar;

  KOKKOS_INLINE_FUNCTION
  constexpr scalar_t operator()(const scalar_t& x) const {
    return project(x);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr scalar_t project(const scalar_t& x) const {
    return x;
  }
};

/// \brief Proj(x) = max(x, lower_bound) for all x in R
template <typename Scalar>
struct LowerBound {
  using scalar_t = Scalar;

  scalar_t lower_bound;

  KOKKOS_INLINE_FUNCTION
  constexpr scalar_t operator()(const scalar_t& x) const {
    return project(x);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr scalar_t project(const scalar_t& x) const {
    return Kokkos::max(x, lower_bound);
  }
};

/// \brief Proj(x) = min(x, upper_bound) for all x in R
template <typename Scalar>
struct UpperBound {
  using scalar_t = Scalar;

  scalar_t upper_bound;

  KOKKOS_INLINE_FUNCTION
  constexpr scalar_t operator()(const scalar_t& x) const {
    return project(x);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr scalar_t project(const scalar_t& x) const {
    return Kokkos::min(x, upper_bound);
  }
};

/// \brief Proj(x) = min(max(x, lower_bound), upper_bound) for all x in R
template <typename Scalar>
struct Bounded {
  using scalar_t = Scalar;

  scalar_t lower_bound;
  scalar_t upper_bound;

  KOKKOS_INLINE_FUNCTION
  constexpr scalar_t operator()(const scalar_t& x) const {
    return project(x);
  }

  KOKKOS_INLINE_FUNCTION
  constexpr scalar_t project(const scalar_t& x) const {
    return Kokkos::min(Kokkos::max(x, lower_bound), upper_bound);
  }
};

}  // namespace space

//! \name Backends
//@{

namespace impl {

template <class Op>
concept DenseMatView = Kokkos::is_view_v<Op> && requires {
  // These are checked in constraint context:
  { std::remove_reference_t<Op>::rank } -> std::convertible_to<int>;
  typename std::remove_reference_t<Op>::non_const_value_type;
} && (std::remove_reference_t<Op>::rank == 2);

template <class Op, typename X, typename Y>
concept HasApplyMember = requires(const Op& op, const X& x, Y& y) {
  { op.apply(x, y) } -> std::same_as<void>;
};

template <class Op, typename X, typename Y, typename Workspace>
concept HasApplyMemberWithWorkspace = requires(const Op& op, const X& x, Y& y, Workspace& workspace) {
  { op.apply(x, y, workspace) } -> std::same_as<void>;
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

template <class Op, class Vector>
concept HasMakeWorkspaceMember = requires(const Op& op, const Vector& q) {
  { op.make_workspace(q) };
};

template <class LinearOp, class Vector>
auto make_workspace(const LinearOp& op, const Vector& q) {
  if constexpr (HasMakeWorkspaceMember<LinearOp, Vector>) {
    return op.make_workspace(q);
  } else {
    return NoWorkspace();
  }
}

template <class LinearOp, class Vector>
using workspace_for_t = decltype(make_workspace(std::declval<const LinearOp&>(), std::declval<const Vector&>()));

template <class Vector>
using vector_scalar_t = std::remove_cvref_t<decltype(std::declval<const std::remove_reference_t<Vector>&>()(size_t{}))>;

template <class T>
struct is_reference_wrapper : std::false_type {};

template <class U>
struct is_reference_wrapper<std::reference_wrapper<U>> : std::true_type {};

template <class T>
inline constexpr bool is_reference_wrapper_v = is_reference_wrapper<std::remove_cvref_t<T>>::value;

template <class Storage>
KOKKOS_INLINE_FUNCTION decltype(auto) unwrap(Storage& value) {
  if constexpr (is_reference_wrapper_v<Storage>) {
    return value.get();
  } else {
    return (value);
  }
}

template <class Storage>
KOKKOS_INLINE_FUNCTION decltype(auto) unwrap(const Storage& value) {
  if constexpr (is_reference_wrapper_v<Storage>) {
    return value.get();
  } else {
    return (value);
  }
}

template <class Storage>
using unwrapped_storage_t = std::remove_cvref_t<decltype(unwrap(std::declval<const Storage&>()))>;

}  // namespace impl

template <class Backend, class LinearOpDT, class LinearOpM, class LinearOpD>
class QuadraticFormOp {
 public:
  using backend_t = Backend;

  template <class FStorage, class UStorage>
  struct Workspace {
    KOKKOS_INLINE_FUNCTION Workspace(FStorage f, UStorage u, bool committed = false)
        : f_(f), u_(u), committed_(committed) {
    }

    // clang-format off
    KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
    KOKKOS_INLINE_FUNCTION       auto& f()       { return impl::unwrap(f_); }
    KOKKOS_INLINE_FUNCTION const auto& f() const { return impl::unwrap(f_); }
    KOKKOS_INLINE_FUNCTION       auto& u()       { return impl::unwrap(u_); }
    KOKKOS_INLINE_FUNCTION const auto& u() const { return impl::unwrap(u_); }
    KOKKOS_INLINE_FUNCTION void commit() { committed_ = true; }
    KOKKOS_INLINE_FUNCTION void invalidate() { committed_ = false; }
    KOKKOS_INLINE_FUNCTION bool is_committed() const { return committed_; }
    // clang-format on

   private:
    FStorage f_;
    UStorage u_;
    bool committed_{false};
  };

  KOKKOS_INLINE_FUNCTION
  QuadraticFormOp(backend_t, const LinearOpDT& DT, const LinearOpM& M, const LinearOpD& D) : DT_(DT), M_(M), D_(D) {
  }

  KOKKOS_INLINE_FUNCTION Backend backend() const {
    return Backend{};
  }

  template <class QVector>
  auto make_workspace(const QVector& q) const {
    using f_t = decltype(Backend::make_vector_like(q));
    using u_t = decltype(Backend::make_vector_like(q));
    return Workspace<f_t, u_t>(Backend::make_vector_like(q), Backend::make_vector_like(q), false);
  }

  template <class FVector, class UVector>
  auto make_workspace(FVector& f, UVector& u, bool committed = false) const {
    using f_t = std::reference_wrapper<FVector>;
    using u_t = std::reference_wrapper<UVector>;
    return Workspace<f_t, u_t>(std::ref(f), std::ref(u), committed);
  }

  template <class XVector, class YVector, class WorkspaceType>
  KOKKOS_INLINE_FUNCTION void apply(const XVector& x, YVector& y, WorkspaceType& workspace) const {
    impl::workspace_invalidate(workspace);
    Backend::apply(D_, x, workspace.f());
    Backend::apply(M_, workspace.f(), workspace.u());
    Backend::apply(DT_, workspace.u(), y);
  }

 private:
  const LinearOpDT& DT_;
  const LinearOpM& M_;
  const LinearOpD& D_;
};

/// \brief Backend for Kokkos single process execution
template <typename ExecSpace>
struct KokkosBackend {
 public:
  using exec_space = ExecSpace;

 public:
  template <class Vector>
  static auto make_vector_like(const Vector& q) {
    return Vector(q.extent(0));
  }

  template <class Vector>
  KOKKOS_INLINE_FUNCTION static size_t vector_size(const Vector& x) {
    return x.extent(0);
  }

  template <class Vector>
  KOKKOS_INLINE_FUNCTION static decltype(auto) vector_data(const Vector& x, size_t i) {
    return x(i);
  }

  template <class Vector>
  KOKKOS_INLINE_FUNCTION static decltype(auto) vector_data(Vector& x, size_t i) {
    return x(i);
  }

  template <class DestVector, class SrcVector>
  static void deep_copy(DestVector& dest, const SrcVector& src) {
    Kokkos::deep_copy(dest, src);
  }

#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
  // Path 1: If op is a dense 2D Kokkos::View, call BLAS gemv.
  // y = A*x
  template <class LinearOp, class XVector, class YVector>
    requires(impl::DenseMatView<LinearOp>)
  static void apply(const LinearOp& A, const XVector& x, YVector& y) {
    using scalar_t = impl::vector_scalar_t<YVector>;
    MUNDY_THROW_ASSERT(A.extent(1) == x.extent(0), std::invalid_argument, "gemv: dimension mismatch A(:,1) vs x");
    MUNDY_THROW_ASSERT(A.extent(0) == y.extent(0), std::invalid_argument, "gemv: dimension mismatch A(0,:) vs y");
    KokkosBlas::gemv(exec_space{}, "N", scalar_t(1), A, x, scalar_t(0), y);
  }

  template <class LinearOp, class XVector, class YVector, class Workspace>
    requires(impl::DenseMatView<LinearOp>)
  static void apply(const LinearOp& A, const XVector& x, YVector& y, Workspace&) {
    apply(A, x, y);
  }

  // Path 1: If op is a dense 2D Kokkos::View, call BLAS gemv.
  // y = alpha * A * x + beta * y
  template <class Scalar, class LinearOp, class XVector, class YVector>
    requires(impl::DenseMatView<LinearOp>)
  static void apply(Scalar alpha, const LinearOp& A, const XVector& x, Scalar beta, YVector& y) {
    MUNDY_THROW_ASSERT(A.extent(1) == x.extent(0), std::invalid_argument, "gemv: dimension mismatch A(:,1) vs x");
    MUNDY_THROW_ASSERT(A.extent(0) == y.extent(0), std::invalid_argument, "gemv: dimension mismatch A(0,:) vs y");
    KokkosBlas::gemv(exec_space{}, "N", alpha, A, x, beta, y);
  }
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS

  // Path 2: If op has member `apply(x,y)`
  template <class LinearOp, class XVector, class YVector>
    requires(!impl::DenseMatView<LinearOp> && impl::HasApplyMember<LinearOp, XVector, YVector>)
  static void apply(const LinearOp& op, const XVector& x, YVector& y) {
    op.apply(x, y);
  }

  template <class LinearOp, class XVector, class YVector, class Workspace>
    requires(!impl::DenseMatView<LinearOp> && impl::HasApplyMemberWithWorkspace<LinearOp, XVector, YVector, Workspace>)
  static void apply(const LinearOp& op, const XVector& x, YVector& y, Workspace& workspace) {
    op.apply(x, y, workspace);
  }

  template <class LinearOp, class XVector, class YVector, class Workspace>
    requires(!impl::DenseMatView<LinearOp> &&
             !impl::HasApplyMemberWithWorkspace<LinearOp, XVector, YVector, Workspace> &&
             impl::HasApplyMember<LinearOp, XVector, YVector>)
  static void apply(const LinearOp& op, const XVector& x, YVector& y, Workspace&) {
    op.apply(x, y);
  }

  // Path 3: Otherwise, runtime error.
  template <typename LinearOp, class XVector, class YVector>
    requires(!impl::DenseMatView<LinearOp> && !impl::HasApplyMember<LinearOp, XVector, YVector>)
  static void apply(const LinearOp& op, const XVector& x, YVector& y) {
    MUNDY_THROW_REQUIRE(false, std::logic_error,
                        "KokkosBackend::apply: op must be a rank-2 Kokkos::View or provide void apply(x,y).");
  }

  template <typename LinearOp, class XVector, class YVector, class Workspace>
    requires(!impl::DenseMatView<LinearOp> &&
             !impl::HasApplyMemberWithWorkspace<LinearOp, XVector, YVector, Workspace> &&
             !impl::HasApplyMember<LinearOp, XVector, YVector>)
  static void apply(const LinearOp&, const XVector&, YVector&, Workspace&) {
    MUNDY_THROW_REQUIRE(
        false, std::logic_error,
        "KokkosBackend::apply: op must be a rank-2 Kokkos::View or provide void apply(x,y[,workspace]).");
  }

  template <class Scalar, class XVector, class YVector>
  static void axpby(const Scalar alpha, const XVector& x, const Scalar beta, YVector& y) {
    MUNDY_THROW_ASSERT(x.extent(0) == y.extent(0), std::invalid_argument, "x and y must have the same size.");
    const bool alpha_is_zero = Kokkos::abs(alpha) < get_zero_tolerance<Scalar>();
    const bool beta_is_zero = Kokkos::abs(beta) < get_zero_tolerance<Scalar>();

    if (!alpha_is_zero && !beta_is_zero) {
      Kokkos::parallel_for(
          "axpby", Kokkos::RangePolicy<exec_space>(0, x.extent(0)),
          KOKKOS_LAMBDA(const int i) { y(i) = alpha * x(i) + beta * y(i); });
    } else if (alpha_is_zero && !beta_is_zero) {
      Kokkos::parallel_for(
          "axpby", Kokkos::RangePolicy<exec_space>(0, x.extent(0)), KOKKOS_LAMBDA(const int i) { y(i) *= beta; });
    } else if (!alpha_is_zero && beta_is_zero) {
      Kokkos::parallel_for(
          "axpby", Kokkos::RangePolicy<exec_space>(0, x.extent(0)),
          KOKKOS_LAMBDA(const int i) { y(i) = alpha * x(i); });
    } else {
      // Both alpha and beta are zero. Set y to zero.
      Kokkos::deep_copy(y, Scalar(0));
    }
  }

  template <typename Wrapper, class Scalar, class XVector, class YVector, class ZVector>
  static void wrapped_axpbyz(const Scalar alpha, const XVector& x, const Scalar beta, const YVector& y, ZVector& z,
                             const Wrapper& wrapper) {
    MUNDY_THROW_ASSERT(x.extent(0) == y.extent(0) && x.extent(0) == z.extent(0), std::invalid_argument,
                       "x, y, and z must have the same size.");
    const bool alpha_is_zero = Kokkos::abs(alpha) < get_zero_tolerance<Scalar>();
    const bool beta_is_zero = Kokkos::abs(beta) < get_zero_tolerance<Scalar>();

    if (!alpha_is_zero && !beta_is_zero) {
      Kokkos::parallel_for(
          "wrapped_axpbyz", Kokkos::RangePolicy<exec_space>(0, x.extent(0)),
          KOKKOS_LAMBDA(const int i) { z(i) = wrapper(alpha * x(i) + beta * y(i)); });
    } else if (alpha_is_zero && !beta_is_zero) {
      Kokkos::parallel_for(
          "wrapped_axpbyz", Kokkos::RangePolicy<exec_space>(0, x.extent(0)),
          KOKKOS_LAMBDA(const int i) { z(i) = wrapper(beta * y(i)); });
    } else if (!alpha_is_zero && beta_is_zero) {
      Kokkos::parallel_for(
          "wrapped_axpbyz", Kokkos::RangePolicy<exec_space>(0, x.extent(0)),
          KOKKOS_LAMBDA(const int i) { z(i) = wrapper(alpha * x(i)); });
    } else {
      Kokkos::parallel_for(
          "wrapped_axpbyz", Kokkos::RangePolicy<exec_space>(0, x.extent(0)),
          KOKKOS_LAMBDA(const int i) { z(i) = wrapper(Scalar(0)); });
    }
  }

  template <class ReductionScalar, class XVector, class YVector>
  static ReductionScalar diff_dot(const XVector& x, const YVector& y) {
    MUNDY_THROW_ASSERT(x.extent(0) == y.extent(0), std::invalid_argument, "x and y must have the same size.");
    ReductionScalar result = 0;
    Kokkos::parallel_reduce(
        "diff_dot", Kokkos::RangePolicy<exec_space>(0, x.extent(0)),
        KOKKOS_LAMBDA(const int i, ReductionScalar& sum) {
          ReductionScalar diff = x(i) - y(i);
          sum += diff * diff;
        },
        result);
    return result;
  }

  template <class ReductionScalar, class X1Vector, class X2Vector, class Y1Vector, class Y2Vector>
  static ReductionScalar diff_dot(const X1Vector& x1, const X2Vector& x2,  //
                                  const Y1Vector& y1, const Y2Vector& y2) {
    MUNDY_THROW_ASSERT(x1.extent(0) == x2.extent(0) && x1.extent(0) == y1.extent(0) && x1.extent(0) == y2.extent(0),
                       std::invalid_argument, "x1, x2, y1, and y2 must have the same size.");
    ReductionScalar result = 0;
    Kokkos::parallel_reduce(
        "diff_dot", Kokkos::RangePolicy<exec_space>(0, x1.extent(0)),
        KOKKOS_LAMBDA(const int i, ReductionScalar& sum) {
          ReductionScalar x_diff = x1(i) - x2(i);
          ReductionScalar y_diff = y1(i) - y2(i);
          sum += x_diff * y_diff;
        },
        result);
    return result;
  }

  template <typename ReductionScalar, typename Functor>
  static void reduce_max(size_t n, const Functor& func, ReductionScalar& result) {
    Kokkos::parallel_reduce(
        "reduce_max", Kokkos::RangePolicy<exec_space>(0, n),
        KOKKOS_LAMBDA(const int i, ReductionScalar& max_val) { func(i, max_val); },
        Kokkos::Max<ReductionScalar>(result));
  }

  template <typename ReductionScalar, typename Functor, class Vector>
  static void reduce_max(const Vector&, size_t n, const Functor& func, ReductionScalar& result) {
    reduce_max<ReductionScalar>(n, func, result);
  }
};  // KokkosBackend

/// \brief Backend for Mundy math within a kernel
struct MundyMathBackend {
  template <class Vector>
  static auto make_vector_like(const Vector&) {
    return Vector();
  }

  template <class Vector>
  KOKKOS_INLINE_FUNCTION static size_t vector_size(const Vector& /*x*/) {
    return std::remove_reference_t<Vector>::size;
  }

  template <class Vector>
  KOKKOS_INLINE_FUNCTION static decltype(auto) vector_data(const Vector& x, size_t i) {
    return x[i];
  }

  template <class Vector>
  KOKKOS_INLINE_FUNCTION static decltype(auto) vector_data(Vector& x, size_t i) {
    return x[i];
  }

  template <class DestVector, class SrcVector>
  KOKKOS_INLINE_FUNCTION static void deep_copy(DestVector& dest, const SrcVector& src) {
    dest = src;
  }

  template <typename LinearOp, class XVector, class YVector>
  KOKKOS_INLINE_FUNCTION static void apply(const LinearOp& op, const XVector& x, YVector& y) {
    y = op * x;
  }

  template <typename LinearOp, class XVector, class YVector, typename Workspace>
  KOKKOS_INLINE_FUNCTION static void apply(const LinearOp& op, const XVector& x, YVector& y, Workspace& workspace) {
    if constexpr (impl::HasApplyMemberWithWorkspace<LinearOp, XVector, YVector, Workspace>) {
      op.apply(x, y, workspace);
    } else if constexpr (impl::HasApplyMember<LinearOp, XVector, YVector>) {
      op.apply(x, y);
    } else {
      y = op * x;
    }
  }

  template <class Scalar, class XVector, class YVector>
  KOKKOS_INLINE_FUNCTION static void axpby(const Scalar alpha, const XVector& x, const Scalar beta, YVector& y) {
    y = alpha * x + beta * y;
  }

  template <typename Wrapper, class Scalar, class XVector, class YVector, class ZVector>
  KOKKOS_INLINE_FUNCTION static void wrapped_axpbyz(const Scalar alpha, const XVector& x, const Scalar beta,
                                                    const YVector& y, ZVector& z, const Wrapper& wrapper) {
    z = ::mundy::math::apply(wrapper, alpha * x + beta * y);
  }

  template <class ReductionScalar, class XVector, class YVector>
  KOKKOS_INLINE_FUNCTION static ReductionScalar diff_dot(const XVector& x, const YVector& y) {
    auto diff = x - y;
    return static_cast<ReductionScalar>(dot(diff, diff));
  }

  template <class ReductionScalar, class X1Vector, class X2Vector, class Y1Vector, class Y2Vector>
  KOKKOS_INLINE_FUNCTION static ReductionScalar diff_dot(const X1Vector& x1, const X2Vector& x2,  //
                                                         const Y1Vector& y1, const Y2Vector& y2) {
    auto x_diff = x1 - x2;
    auto y_diff = y1 - y2;
    return static_cast<ReductionScalar>(dot(x_diff, y_diff));
  }

  template <class ReductionScalar, class Functor, class Vector>
  KOKKOS_INLINE_FUNCTION static void reduce_max(const Vector&, size_t n, const Functor& func, ReductionScalar& result) {
    constexpr size_t N = std::remove_reference_t<Vector>::size;
    MUNDY_THROW_ASSERT(n == N, std::invalid_argument, "reduce_max: n must match the size of the vector.");
    reduce_max_impl(std::make_index_sequence<N>{}, func, result);
  }

 private:
  template <size_t... Is, class Functor, class ReductionScalar>
  KOKKOS_INLINE_FUNCTION static void reduce_max_impl(std::index_sequence<Is...>, const Functor& func,
                                                     ReductionScalar& result) {
    ReductionScalar max_val = -Kokkos::Experimental::infinity_v<ReductionScalar>;
    ((func(Is, max_val)), ...);
    result = max_val;
  }
};  // MundyMathBackend
//@}

//! \name Problems + state
//@{

/// \brief Constrained quadratic programming problem (CQPP) formulation
///
/// This is for a constrained quadratic programming problem of the form:
///   x^* = argmin_{x in Omega} 0.5 x^T A x + q^T x
/// where A is a symmetric positive semi-definite matrix, q is a vector, and Omega is a convex space.
///
/// \tparam Backend The backend to use for operations (e.g., KokkosBackend, MundyMathBackend)
template <typename Backend, typename LinearOpStorage, typename QVector, typename ConvexSpace,
      typename Workspace = impl::workspace_for_t<impl::unwrapped_storage_t<LinearOpStorage>, QVector>>
class CQPPProblem {
 public:
  using backend_t = Backend;
  using linear_op_t = LinearOpStorage;
  using vector_t = QVector;
  using space_t = ConvexSpace;
  using workspace_t = Workspace;
  using scalar_t = impl::vector_scalar_t<vector_t>;

  CQPPProblem(Backend, linear_op_t A, const vector_t& q, const space_t& space)
    : A_(std::move(A)), q_(q), space_(space), workspace_(impl::make_workspace(impl::unwrap(A_), q)) {
  }

  CQPPProblem(Backend, linear_op_t A, const vector_t& q, const space_t& space, const workspace_t& workspace)
    : A_(std::move(A)), q_(q), space_(space), workspace_(workspace) {
  }

  // Accessors — all const to preserve the problem definition
  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  KOKKOS_INLINE_FUNCTION const auto& A() const { return impl::unwrap(A_); }
  KOKKOS_INLINE_FUNCTION const vector_t& q() const { return q_; }
  KOKKOS_INLINE_FUNCTION const space_t& space() const { return space_; }
  KOKKOS_INLINE_FUNCTION workspace_t& workspace() const { return workspace_; }
  // clang-format on

 private:
  linear_op_t A_;
  const vector_t& q_;
  const space_t& space_;
  mutable workspace_t workspace_;
};

/// \brief Mixed constrained quadratic programming problem (MCQPP) formulation
///
/// This is for a mixed constrained convex quadratic programming problem such as:
///   x^*, y^* = argmin_{x in Omega_x, y in R^m} q^T x + b^T y + 0.5 (Dx + By)^T M (Dx + By) + 0.5 y^T K^{-1} y
/// where M and K^{-1} are symmetric positive definite matrices, D and B are linear operators,
/// q and b are vectors, and Omega_x is a convex space.
///
/// This can be mapped onto a reduced CQPP in x alone via the Schur complement.
/// Define:
///   S^{-1} := B^T M B + K^{-1} (symmetric positive definite)
///   H := D^T M D - D^T M B S B^T M D
///   g := q - D^T M B S b
/// Then the reduced CQPP is:
///   x^* = argmin_{x in Omega_x} 0.5 x^T H x + g^T x
///   y^* = -S^{-1}(b + B^T M D x^*)
///
/// In more general terms, the mixed problem exists because of an affine operator mapping x -> f:
///   f = (I + L M) D x + f_b,
/// where f_b = f(x = 0) and L is an spsd linear operator
///
/// In the less-general problem above: L = B S B^T and f_b = B S b. In this form, the reduced CQPP is:
///   x^* = argmin_{x in Omega_x} 0.5 x^T H x + g^T x
///   H := D^T M (D - L M D)
///   g := q - D^T M f_b
/// y^*, f^*, and u^* are all intermediates that may or may not be explicitly formed during the solution
/// process depending on the structure of L. We refer to these variables as your "workspace" variables. They are updated
/// by the linear operators during the solution process, and their final values are "committed" at the end of the solve
/// to indicate that they are the correct values corresponding to the optimal x^*.
///
/// We may also have the simpler problem where D = I, in which case the reduced CQPP is:
///   x^* = argmin_{x in Omega_x} 0.5 x^T H x + g^T x
///   H := A (I - L A)
///   g := q - A f_b
///
/// Just like the CQPP, we will either accept the simple case of (A, q, L, f_b) or the more general case of (DT, M, D, q, BT, S, B, b).
///
/// 	param Backend The backend to use for operations (e.g., KokkosBackend, MundyMathBackend)
template <typename Backend, typename LinearOpStorage, typename QVector, typename ConvexSpace,
      typename Workspace = impl::workspace_for_t<impl::unwrapped_storage_t<LinearOpStorage>, QVector>>
class MCQPPProblem {
 public:
  using backend_t = Backend;
  using linear_op_t = LinearOpStorage;
  using vector_t = QVector;
  using space_t = ConvexSpace;
  using workspace_t = Workspace;
  using scalar_t = impl::vector_scalar_t<vector_t>;

  MCQPPProblem(Backend, linear_op_t A, const vector_t& q, const space_t& space)
    : A_(std::move(A)), q_(q), space_(space), workspace_(impl::make_workspace(impl::unwrap(A_), q)) {
  }

  MCQPPProblem(Backend, linear_op_t A, const vector_t& q, const space_t& space, const workspace_t& workspace)
    : A_(std::move(A)), q_(q), space_(space), workspace_(workspace) {
  }

  // Accessors — all const to preserve the problem definition
  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  KOKKOS_INLINE_FUNCTION const auto& A() const { return impl::unwrap(A_); }
  KOKKOS_INLINE_FUNCTION const vector_t& q() const { return q_; }
  KOKKOS_INLINE_FUNCTION const space_t& space() const { return space_; }
  KOKKOS_INLINE_FUNCTION workspace_t& workspace() const { return workspace_; }
  // clang-format on

 private:
  linear_op_t A_;
  const vector_t& q_;
  const space_t& space_;
  mutable workspace_t workspace_;
};

/// \brief Linear complementarity problem (LCP) formulation
///
/// This is for a linear complementarity problem of the form:
///   0 <= A x + q _|_ x >= 0
/// where A is a symmetric positive semi-definite matrix, q is a vector, and x is the solution vector.
///
/// This is equivalent to solving the following constrained quadratic programming problem:
///   x^* = argmin 0.5 x^T A x + q^T x
///          s.t  x in R^n, x >= 0
///
/// 	param Backend The backend to use for operations (e.g., KokkosBackend, MundyMathBackend)
template <typename Backend, typename LinearOpStorage, typename QVector,
      typename Workspace = impl::workspace_for_t<impl::unwrapped_storage_t<LinearOpStorage>, QVector>>
class LCPProblem {
 public:
  using backend_t = Backend;
  using linear_op_t = LinearOpStorage;
  using vector_t = QVector;
  using workspace_t = Workspace;
  using scalar_t = impl::vector_scalar_t<vector_t>;

  LCPProblem(Backend, linear_op_t A, const vector_t& q)
    : A_(std::move(A)), q_(q), workspace_(impl::make_workspace(impl::unwrap(A_), q)) {
  }

  LCPProblem(Backend, linear_op_t A, const vector_t& q, const workspace_t& workspace)
    : A_(std::move(A)), q_(q), workspace_(workspace) {
  }

  // clang-format off
  KOKKOS_INLINE_FUNCTION Backend backend() const { return Backend{}; }
  KOKKOS_INLINE_FUNCTION const auto& A() const { return impl::unwrap(A_); }
  KOKKOS_INLINE_FUNCTION const vector_t& q() const { return q_; }
  KOKKOS_INLINE_FUNCTION workspace_t& workspace() const { return workspace_; }
  // clang-format on

 private:
  linear_op_t A_;
  const vector_t& q_;
  mutable workspace_t workspace_;
};

template <class Backend, class LinearOp, class QVector>
KOKKOS_INLINE_FUNCTION auto to_cqpp(const LCPProblem<Backend, LinearOp, QVector>& P) {
  static constexpr space::LowerBound Rn_plus{static_cast<typename LCPProblem<Backend, LinearOp, QVector>::scalar_t>(0)};
  return CQPPProblem(P.backend(), P.A(), P.q(), Rn_plus, P.workspace());
}
//@}

//! \name Policies
//@{

struct LinfNormProjectedGradientResidual {  // LCP only
  template <typename Backend, typename XVector, typename GradVector, typename ConvexSpace,
            typename ReductionScalar = impl::vector_scalar_t<GradVector>>
  KOKKOS_INLINE_FUNCTION ReductionScalar operator()([[maybe_unused]] const Backend& backend,  //
                                                    const XVector& x,                         //
                                                    const GradVector& grad,                   //
                                                    const ConvexSpace& convex_space) const {
    using scalar_t = ReductionScalar;

    size_t n = Backend::vector_size(x);
    scalar_t largest_abs_gradient;
    Backend::template reduce_max<scalar_t>(
        x, n,
        KOKKOS_LAMBDA(const int i, scalar_t& max_val) {
          // perform the projection EQ 2.2 of Dai & Fletcher 2005
          scalar_t x_i = Backend::vector_data(x, i);
          scalar_t grad_i = Backend::vector_data(grad, i);

          scalar_t abs_projected_grad;
          if (x_i < get_zero_tolerance<scalar_t>()) {
            abs_projected_grad = Kokkos::max(scalar_t(0), grad_i);
          } else {
            abs_projected_grad = Kokkos::abs(grad_i);
          }

          if (abs_projected_grad > max_val) {
            max_val = abs_projected_grad;
          }
        },
        largest_abs_gradient);

    return largest_abs_gradient;
  }
};

struct LinfNormProjectedDiffResidual {
  template <typename Backend, typename XVector, typename GradVector, typename ConvexSpace,
            typename ReductionScalar = impl::vector_scalar_t<GradVector>>
  KOKKOS_INLINE_FUNCTION ReductionScalar operator()([[maybe_unused]] const Backend& backend,  //
                                                    const XVector& x,                         //
                                                    const GradVector& grad,                   //
                                                    const ConvexSpace& convex_space) const {
    using scalar_t = ReductionScalar;

    // This res comes from line 17 and Eq 25 of Mazhar 2015
    // res =  1.0 / (3 * num_unknowns * gd) * norm_inf(xk - proj(xk - gd * gk))
    size_t num_unknowns = Backend::vector_size(x);
    constexpr scalar_t small_step_size = static_cast<scalar_t>(1e-6);
    scalar_t largest_abs_diff;
    Backend::template reduce_max<scalar_t>(
        x, num_unknowns,
        KOKKOS_LAMBDA(const int i, scalar_t& max_val) {
          scalar_t x_i = Backend::vector_data(x, i);
          scalar_t grad_i = Backend::vector_data(grad, i);
          scalar_t x_i_proj = convex_space.project(x_i - small_step_size * grad_i);
          scalar_t abs_diff = Kokkos::abs(x_i - x_i_proj);
          if (abs_diff > max_val) {
            max_val = abs_diff;
          }
        },
        largest_abs_diff);

    return largest_abs_diff / small_step_size;
  }
};

struct BBStepStrategy {
  template <typename Backend, typename XOldVector, typename GradOldVector, typename XVector, typename GradVector,
            typename ReductionScalar = impl::vector_scalar_t<XVector>>
  KOKKOS_INLINE_FUNCTION ReductionScalar operator()([[maybe_unused]] const Backend& backend,  //
                                                    const XOldVector& x_old,
                                                    const GradOldVector& grad_old,  //
                                                    const XVector& x, const GradVector& grad) const {
    using scalar_t = ReductionScalar;

    scalar_t num = Backend::template diff_dot<scalar_t>(x, x_old);  // (x - x_old) dot (x - x_old)
    scalar_t denom =
        Backend::template diff_dot<scalar_t>(x, x_old, grad, grad_old);  // (x - x_old) dot (grad - grad_old)

    // Avoid division by zero
    constexpr scalar_t eps = get_zero_tolerance<scalar_t>() * static_cast<scalar_t>(10);
    denom += eps * (Kokkos::abs(denom) < eps);

    return num / denom;
  }
};  // BBStepStrategy
//@}

template <typename Scalar>
struct PGDConfig {
  using scalar_t = Scalar;

  unsigned max_iters{1000};
  Scalar tol{get_relaxed_zero_tolerance<Scalar>()};
};

template <class Scalar>
struct SolveResult {
  using scalar_t = Scalar;

  unsigned num_iters{0};
  Scalar residual{0};
  bool converged{false};
};

/// \brief Write SolveResult to an ostream
template <class Scalar>
std::ostream& operator<<(std::ostream& os, const SolveResult<Scalar> result) {
  os << "num_iters: " << result.num_iters << ", residual: " << result.residual << ", converged?: " << result.converged;
  return os;
}

template <class Scalar, class XVector, class GradVector, class XTmpVector, class GradTmpVector>
class PGDState {
 public:
  using scalar_t = Scalar;
  using x_vector_t = XVector;
  using grad_vector_t = GradVector;
  using x_tmp_vector_t = XTmpVector;
  using grad_tmp_vector_t = GradTmpVector;

  KOKKOS_INLINE_FUNCTION
  PGDState(x_vector_t& x, grad_vector_t& g, x_tmp_vector_t& x_tmp, grad_tmp_vector_t& g_tmp)
      : x_(x), g_(g), x_tmp_(x_tmp), g_tmp_(g_tmp) {
  }

  // Accessors (const/non-const as needed)
  // clang-format off
  KOKKOS_INLINE_FUNCTION       x_vector_t& x()      { return x_; }
  KOKKOS_INLINE_FUNCTION const x_vector_t& x() const{ return x_; }
  KOKKOS_INLINE_FUNCTION       grad_vector_t& grad()      { return g_; }
  KOKKOS_INLINE_FUNCTION const grad_vector_t& grad() const{ return g_; }
  KOKKOS_INLINE_FUNCTION       x_tmp_vector_t& x_tmp()      { return x_tmp_; }
  KOKKOS_INLINE_FUNCTION const x_tmp_vector_t& x_tmp() const{ return x_tmp_; }
  KOKKOS_INLINE_FUNCTION       grad_tmp_vector_t& grad_tmp()      { return g_tmp_; }
  KOKKOS_INLINE_FUNCTION const grad_tmp_vector_t& grad_tmp() const{ return g_tmp_; }
  // clang-format on

  // Iteration locals with accessors
  // clang-format off
  KOKKOS_INLINE_FUNCTION unsigned& iter()         { return iter_; }
  KOKKOS_INLINE_FUNCTION bool&     converged()    { return converged_; }
  KOKKOS_INLINE_FUNCTION scalar_t& residual()     { return residual_; }
  KOKKOS_INLINE_FUNCTION scalar_t& step_size()        { return step_size_; }

  KOKKOS_INLINE_FUNCTION unsigned  iter()    const { return iter_; }
  KOKKOS_INLINE_FUNCTION bool      converged() const { return converged_; }
  KOKKOS_INLINE_FUNCTION scalar_t  residual() const  { return residual_; }
  KOKKOS_INLINE_FUNCTION scalar_t  step_size()    const  { return step_size_; }
  // clang-format on

 private:
  x_vector_t& x_;
  grad_vector_t& g_;
  x_tmp_vector_t& x_tmp_;
  grad_tmp_vector_t& g_tmp_;
  unsigned iter_{0};
  bool converged_{false};
  scalar_t residual_{0};
  scalar_t step_size_{1};
};

template <class StepPolicy, class ResidualPolicy, class Config>
class PGDStrategy {
 public:
  using scalar_t = typename Config::scalar_t;

  using step_policy_t = StepPolicy;
  using residual_policy_t = ResidualPolicy;
  using config_t = Config;
  using result_t = SolveResult<scalar_t>;

  KOKKOS_INLINE_FUNCTION
  PGDStrategy(step_policy_t step, residual_policy_t resid, config_t cfg = {}) : step_(step), resid_(resid), cfg_(cfg) {
  }

  template <class Problem, class State>
  KOKKOS_INLINE_FUNCTION void initialize(const Problem& prob, State& state) const {
    auto backend = prob.backend();
    using backend_t = decltype(backend);

    constexpr scalar_t one = static_cast<scalar_t>(1);
    auto& workspace = prob.workspace();

    // x_tmp = x
    backend_t::deep_copy(state.x_tmp(), state.x());

    // grad_tmp = A x_tmp + q
    impl::workspace_invalidate(workspace);
    backend_t::apply(prob.A(), state.x_tmp(), state.grad_tmp(), workspace);
    backend_t::axpby(one, prob.q(), one, state.grad_tmp());

    // Dai-Fletcher Sec. 5 initial step
    state.residual() = resid_(backend, state.x_tmp(), state.grad_tmp(), prob.space());

    // Initialize iteration state (allow for early exit)
    state.iter() = 0;
    state.converged() = (state.residual() <= static_cast<scalar_t>(cfg_.tol));
    if (state.converged()) {
      state.step_size() = one;
      // If already converged, copy grad_tmp to grad
      backend_t::deep_copy(state.grad(), state.grad_tmp());
      impl::workspace_commit(workspace);
    } else {
      state.step_size() = one / state.residual();
    }
  }

  template <class Problem, class State>
  KOKKOS_INLINE_FUNCTION bool iterate(const Problem& prob, State& state) const {
    auto backend = prob.backend();
    using backend_t = decltype(backend);

    constexpr scalar_t one = static_cast<scalar_t>(1);
    auto& workspace = prob.workspace();

    if (state.converged() || state.iter() >= cfg_.max_iters) {
      return state.converged();
    }

    // x = Proj(x_tmp - step_size * grad_tmp)
    backend_t::wrapped_axpbyz(one, state.x_tmp(), -state.step_size(), state.grad_tmp(), state.x(),
                                      prob.space());

    // grad = A x + q
    impl::workspace_invalidate(workspace);
    backend_t::apply(prob.A(), state.x(), state.grad(), workspace);
    backend_t::axpby(one, prob.q(), one, state.grad());

    // residual & test
    state.residual() = resid_(backend, state.x(), state.grad(), prob.space());
    if (state.residual() <= static_cast<scalar_t>(cfg_.tol)) {
      state.converged() = true;
      impl::workspace_commit(workspace);
      return true;
    }

    // update step size and roll x_tmp/grad_tmp forward
    state.step_size() = step_(backend, state.x_tmp(), state.grad_tmp(), state.x(), state.grad());
    backend_t::deep_copy(state.x_tmp(), state.x());
    backend_t::deep_copy(state.grad_tmp(), state.grad());
    ++state.iter();
    return false;
  }

  template <class State>
  KOKKOS_INLINE_FUNCTION bool done(const State& state) const {
    return state.converged() || state.iter() >= cfg_.max_iters;
  }

  template <class State>
  KOKKOS_INLINE_FUNCTION result_t result(const State& state) const {
    return {state.iter(), state.residual(), state.converged()};
  }

 private:
  step_policy_t step_;
  residual_policy_t resid_;
  config_t cfg_;
};

template <class Strategy, class Problem, class State>
concept CQPPSolverStrategy = requires(const Strategy& s, const Problem& prob, State& state) {
  { s.initialize(prob, state) } -> std::same_as<void>;
  { s.iterate(prob, state) } -> std::same_as<bool>;
  { s.done(state) } -> std::same_as<bool>;
  s.result(state);
};

//! \name Deduction guides
//@{

/// \brief Deduction guide for CQPPProblem
template <typename Backend, typename LinearOp, typename QVector, typename ConvexSpace>
CQPPProblem(Backend, const LinearOp&, const QVector&, const ConvexSpace&)
    -> CQPPProblem<Backend, LinearOp, QVector, ConvexSpace>;

template <typename Backend, typename LinearOp, typename QVector, typename ConvexSpace, typename Workspace>
CQPPProblem(Backend, const LinearOp&, const QVector&, const ConvexSpace&, const Workspace&)
    -> CQPPProblem<Backend, LinearOp, QVector, ConvexSpace, Workspace>;

/// \brief Deduction guide for LCPProblem
template <typename Backend, typename LinearOp, typename QVector>
LCPProblem(Backend, const LinearOp&, const QVector&) -> LCPProblem<Backend, LinearOp, QVector>;

template <typename Backend, typename LinearOp, typename QVector, typename Workspace>
LCPProblem(Backend, const LinearOp&, const QVector&, const Workspace&)
    -> LCPProblem<Backend, LinearOp, QVector, Workspace>;

/// \brief Deduction guide for PGDConfig
template <typename Scalar>
PGDConfig(unsigned, Scalar) -> PGDConfig<Scalar>;

/// \brief Deduction guide for PGDState
template <class XVector, class GradVector, class XTmpVector, class GradTmpVector>
PGDState(XVector&, GradVector&, XTmpVector&, GradTmpVector&)
    -> PGDState<impl::vector_scalar_t<XVector>, XVector, GradVector, XTmpVector, GradTmpVector>;

/// \brief Deduction guide for PGDStrategy
template <class StepPolicy, class ResidualPolicy, class Config>
PGDStrategy(StepPolicy, ResidualPolicy, Config = {}) -> PGDStrategy<StepPolicy, ResidualPolicy, Config>;
//@}

}  // namespace convex

template <typename Backend, class LinearOpDT, class LinearOpM, class LinearOpD>
KOKKOS_INLINE_FUNCTION auto make_quadratic_form(const LinearOpDT& DT, const LinearOpM& M, const LinearOpD& D) {
  return convex::QuadraticFormOp<Backend, LinearOpDT, LinearOpM, LinearOpD>(Backend{}, DT, M, D);
}

template <typename Backend, typename LinearOp, typename QVector, typename ConvexSpace>
KOKKOS_INLINE_FUNCTION auto make_cqpp(const LinearOp& A, const QVector& q, const ConvexSpace& space) {
  return convex::CQPPProblem(Backend{}, A, q, space);
}

template <typename Backend, typename LinearOpDT, typename LinearOpM, typename LinearOpD, typename QVector,
          typename ConvexSpace>
KOKKOS_INLINE_FUNCTION auto make_cqpp(const LinearOpDT& DT, const LinearOpM& M, const LinearOpD& D, const QVector& q,
                                      const ConvexSpace& space) {
  auto A = make_quadratic_form<Backend>(DT, M, D);
  return convex::CQPPProblem(Backend{}, A, q, space);
}

template <typename Backend, typename LinearOpDT, typename LinearOpM, typename LinearOpD, typename QVector,
          typename ConvexSpace, typename FVector, typename UVector>
KOKKOS_INLINE_FUNCTION auto make_cqpp(const LinearOpDT& DT, const LinearOpM& M, const LinearOpD& D, const QVector& q,
                                      FVector& f, UVector& u, const ConvexSpace& space) {
  auto A = make_quadratic_form<Backend>(DT, M, D);
  auto workspace = A.make_workspace(f, u);
  return convex::CQPPProblem(Backend{}, A, q, space, workspace);
}

template <typename Backend, typename LinearOp, typename QVector, typename ConvexSpace, typename Workspace>
KOKKOS_INLINE_FUNCTION auto make_cqpp(const LinearOp& A, const QVector& q, const ConvexSpace& space,
                                      const Workspace& workspace) {
  return convex::CQPPProblem(Backend{}, A, q, space, workspace);
}

template <typename Backend, typename LinearOp, typename QVector>
KOKKOS_INLINE_FUNCTION auto make_lcp(const LinearOp& A, const QVector& q) {
  return convex::LCPProblem(Backend{}, A, q);
}

template <typename Backend, typename LinearOpDT, typename LinearOpM, typename LinearOpD, typename QVector>
KOKKOS_INLINE_FUNCTION auto make_lcp(const LinearOpDT& DT, const LinearOpM& M, const LinearOpD& D, const QVector& q) {
  auto A = make_quadratic_form<Backend>(DT, M, D);
  return convex::LCPProblem(Backend{}, A, q);
}

template <typename Backend, typename LinearOpDT, typename LinearOpM, typename LinearOpD, typename QVector,
          typename FVector, typename UVector>
KOKKOS_INLINE_FUNCTION auto make_lcp(const LinearOpDT& DT, const LinearOpM& M, const LinearOpD& D, const QVector& q,
                                     FVector& f, UVector& u) {
  auto A = make_quadratic_form<Backend>(DT, M, D);
  auto workspace = A.make_workspace(f, u);
  return convex::LCPProblem(Backend{}, A, q, workspace);
}

template <typename Backend, typename LinearOp, typename QVector, typename Workspace>
KOKKOS_INLINE_FUNCTION auto make_lcp(const LinearOp& A, const QVector& q, const Workspace& workspace) {
  return convex::LCPProblem(Backend{}, A, q, workspace);
}

template <typename Backend, typename LinearOp, typename QVector, typename ConvexSpace>
KOKKOS_INLINE_FUNCTION auto make_mixed_cqpp(const LinearOp& A, const QVector& q, const ConvexSpace& space) {
  return convex::MCQPPProblem(Backend{}, A, q, space);
}

template <typename Backend, typename LinearOp, typename QVector, typename ConvexSpace, typename Workspace>
KOKKOS_INLINE_FUNCTION auto make_mixed_cqpp(const LinearOp& A, const QVector& q, const ConvexSpace& space,
                                            const Workspace& workspace) {
  return convex::MCQPPProblem(Backend{}, A, q, space, workspace);
}

template <class StepPolicy, class ResidualPolicy, class Scalar>
KOKKOS_INLINE_FUNCTION auto make_pgd_solution_strategy(const StepPolicy& step_policy,          //
                                                       const ResidualPolicy& residual_policy,  //
                                                       const convex::PGDConfig<Scalar>& cfg = {}) {
  return convex::PGDStrategy(step_policy, residual_policy, cfg);
}
//
template <class Scalar>
KOKKOS_INLINE_FUNCTION auto make_pgd_solution_strategy(const convex::PGDConfig<Scalar>& cfg = {}) {
  using DefaultStepPolicy = convex::BBStepStrategy;
  using DefaultResidualPolicy = convex::LinfNormProjectedDiffResidual;
  return convex::PGDStrategy(DefaultStepPolicy{}, DefaultResidualPolicy{}, cfg);
}
//
template <class XVector, class GradVector, class XTmpVector, class GradTmpVector>
KOKKOS_INLINE_FUNCTION auto make_pgd_state(XVector& x,         //
                                           GradVector& grad,   //
                                           XTmpVector& x_tmp,  //
                                           GradTmpVector& grad_tmp) {
  using scalar_t = convex::impl::vector_scalar_t<XVector>;
  return convex::PGDState<scalar_t, XVector, GradVector, XTmpVector, GradTmpVector>(x, grad, x_tmp, grad_tmp);
}

/// \brief Solve a constrained quadratic programming problem (CQPP)
///
/// \param prob The constrained quadratic programming problem to solve.
/// \param strat The solution strategy to use.
/// \param state The state to use for the solution strategy, which will be modified during the solve.
/// \return The result of the solve (contents are defined by the strategy).
template <class Problem, class Strategy, class State>
  requires convex::CQPPSolverStrategy<Strategy, Problem, State>
KOKKOS_INLINE_FUNCTION auto solve_cqpp(const Problem& prob, const Strategy& strat, State& state) {
  strat.initialize(prob, state);
  while (!strat.done(state)) {
    if (strat.iterate(prob, state)) break;
  }
  auto result = strat.result(state);
  MUNDY_THROW_ASSERT(!result.converged || convex::impl::workspace_is_committed(prob.workspace()), std::logic_error,
                     "solve_cqpp: converged solution requires committed operator workspace.");
  return result;
}

/// \brief Solve a mixed constrained convex quadratic programming problem (MCQPP)
///
/// This is for a mixed constrained convex quadratic programming problem such as:
///   x^*, y^* = argmin_{x in Omega_x, y in R^m} q^T x + b^T y + 0.5 (Dx + By)^T M (Dx + By) + 0.5 y^T K^{-1} y
/// where M and K^{-1} are symmetric positive definite matrices, D and B are linear operators,
/// q and b are vectors, and Omega_x is a convex space.
///
/// This can be mapped onto a reduced CQPP in x alone via the Schur complement.
/// Define:
///   S^{-1} := B^T M B + K^{-1} (symmetric positive definite)
///   H := D^T M D - D^T M B S B^T M D
///   g := q - D^T M B S b
/// Then the reduced CQPP is:
///   x^* = argmin_{x in Omega_x} 0.5 x^T H x + g^T x
///   y^* = -S^{-1}(b + B^T M D x^*)
///
/// In more general terms, the mixed problem exists because of an affine operator mapping x -> f:
///   f = (I + L M) D x + f_b,
/// where f_b = f(x = 0) and L is an spsd linear operator
///
/// In the less-general problem above: L = B S B^T and f_b = B S b. In this form, the reduced CQPP is:
///   x^* = argmin_{x in Omega_x} 0.5 x^T H x + g^T x
///   H := D^T M (D - L M D)
///   g := q - D^T M f_b
/// y^*, f^*, and u^* are all intermediates that may or may not be explicitly formed during the solution
/// process depending on the structure of L. We refer to these variables as your "workspace" variables. They are updated
/// by the linear operators during the solution process, and their final values are "committed" at the end of the solve
/// to indicate that they are the correct values corresponding to the optimal x^*.
///
/// We may also have the simpler problem where D = I, in which case the reduced CQPP is:
///   x^* = argmin_{x in Omega_x} 0.5 x^T H x + g^T x
///   H := A (I - L A)
///   g := q - A f_b
///
/// Just like the CQPP, we will either accept the simple case of (A, q, L, f_b) or the more general case of (DT, M, D, q, BT, S, B, b).
///
/// \param prob The mixed constrained convex quadratic programming problem to solve.
/// \param strat The solution strategy to use (for the reduced CQPP in x and the linear solve in y).
/// \param state The state to use for the solution strategy, which will be modified during the solve.
/// \return The result of the solve (contents are defined by the strategy).
template <class Problem, class Strategy>
  requires requires(const Problem& p) {
    { to_cqpp(p) };
  }
KOKKOS_INLINE_FUNCTION auto solve_mixed_cqpp(const Problem& prob, const Strategy& strat, typename Strategy::state_t&
state) -> typename Strategy::result_t {
  // Convert MCQPP to CQPP
  auto ccpp_prob = to_cqpp(prob);
  return solve_cqpp(ccpp_prob, strat, state);
}

/// \brief Solve a linear complementarity problem (LCP) using a constrained quadratic programming solver
///
/// This is for a linear complementarity problem of the form:
///   0 <= A x + q _|_ x >= 0
/// where A is a symmetric positive semi-definite matrix, q is a vector, and x is the solution vector.
///
/// This is equivalent to solving the following constrained quadratic programming problem:
///   x^* = argmin 0.5 x^T A x + q^T x
///          s.t  x in R^n, x >= 0
///
/// Example 1:
/// \code{.cpp}
///    // Problem setup
///    Matrix3d A = {/*...*/};
///    Vector3d q = {/*...*/};
///    Vector3d x{/* initial_guess */}, grad{}, x_tmp{}, grad_tmp{};
///
///    // Build the problem (no template args at callsite)
///    const auto lcp = make_mundy_math_lcp(A, q);
///
///    // Reuse the backend token from the problem
///    const auto backend = lcp.backend();
///
///    // Strategy + state
///    PGDConfig cfg{1000, 1e-6};
///
///    auto pgd = make_pgd_solution_strategy(cfg);  // Use default step/residual strategies
///    // auto pgd = make_pgd_solution_strategy(    //
///        MyStepStrat{}, MyResidualStrat{}, cfg);  // Custom step/residual strategies
///    auto pgd_state = make_pgd_state(x, grad, x_tmp, grad_tmp);
///
///    // Solve (can reuse "lcp" and "pgd" across many states)
///    auto result = solve_lcp(lcp, pgd, pgd_state);
/// \endcode
///
/// Example 2: Congruent LCP
/// \code{.cpp}
///    // Problem setup
///    Matrix3d D = {/*...*/};
///    Matrix3d M = {/*...*/};
///    Vector3d q = {/*...*/};
///    Vector3d x{/* initial_guess */}, grad{}, x_tmp{}, grad_tmp{};
///
///    // Build quadratic-form operator and problem
///    using backend_t = convex::MundyMathBackend<double, 3>;
///    auto A = make_quadratic_form<backend_t>(D.view_transpose(), M, D);
///    const auto lcp = make_mundy_math_lcp(A, q);
///
///   // Reuse the backend token from the problem
///    const auto backend = lcp.backend();
///
///    // Strategy + state
///    PGDConfig cfg{1000, 1e-6};
///
///    auto pgd = make_pgd_solution_strategy(backend, cfg);
///    auto pgd_state = make_pgd_state(backend, x, grad, x_tmp, grad_tmp);
///
///    // Solve (can reuse "lcp" and "pgd" across many states)
///    auto result = solve_lcp(lcp, pgd, pgd_state);
///
///    // Workspace (f/u) now lives on the linear operator in the problem
///    const auto& ws = lcp.workspace();
///    MUNDY_THROW_ASSERT(ws.is_committed(), std::logic_error, "workspace must be committed after convergence");
/// \endcode
///
/// \param prob The linear complementarity problem to solve.
/// \param strat The solution strategy to use.
/// \param state The state to use for the solution strategy, which will be modified during the solve.
/// \return The result of the solve (contents are defined by the strategy).
template <class Problem, class Strategy, class State>
  requires requires(const Problem& p) {
    { to_cqpp(p) };
  }
KOKKOS_INLINE_FUNCTION auto solve_lcp(const Problem& prob, const Strategy& strat, State& state) {
  // Convert LCP to CQPP
  auto ccpp_prob = to_cqpp(prob);
  return solve_cqpp(ccpp_prob, strat, state);
}

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_CONVEX_HPP_