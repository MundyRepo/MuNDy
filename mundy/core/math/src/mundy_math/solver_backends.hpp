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

#ifndef MUNDY_MATH_SOLVER_BACKENDS_HPP_
#define MUNDY_MATH_SOLVER_BACKENDS_HPP_

// Kokkos:
#include <Kokkos_Core.hpp>

// C++ core:
#include <concepts>
#include <type_traits>
#include <utility>

// KokkosKernels
#include <MundyMath_config.hpp>  // for HAVE_MUNDYMATH_*
#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
#include <KokkosBlas.hpp>
#include <KokkosSparse_spmv.hpp>
#endif

// Mundy
#include <mundy_math/Matrix.hpp>     // for mundy::is_matrix_v
#include <mundy_math/NumTraits.hpp>  // for mundy::NumTraits
#include <mundy_math/Tolerance.hpp>  // for mundy::get_zero_tolerance<T>
#include <mundy_math/Vector.hpp>     // for mundy::Vector
#include <mundy_math/cmath.hpp>
#include <mundy_math/impl/solver_backends_impl.hpp>  // for the impl:: contract concepts + workspace machinery
#include <mundy_utils/reference_wrapper.hpp>
#include <mundy_utils/requires.hpp>
#include <mundy_utils/storage.hpp>            // for mundy::storage, mundy::store
#include <mundy_utils/suppress_warnings.hpp>  // for MUNDY_SUPPRESS_GPU_CALL_FROM_HOST_WARNINGS_PUSH/POP
#include <mundy_utils/throw_assert.hpp>
#include <mundy_utils/type_traits.hpp>  // for mundy::dependent_false_v

namespace mundy {

//! \name Backend contract machinery
//@{

/// \brief Concept for a backend's vector-level operations: size, axpby, a plain inner product, and deep_copy.
template <class Backend, class Vector>
concept VectorBackend = requires(Vector& y, const Vector& x, impl::vector_value_type<Vector> a) {
  { Backend::size(x) } -> std::convertible_to<size_t>;
  { Backend::axpby(a, x, a, y) } -> std::same_as<void>;
  { Backend::template dot<impl::vector_value_type<Vector>>(x, x) };
  { Backend::deep_copy(y, x) };
};

/// \brief Concept for a linear operator's shape/apply contract under a given backend: domain_size, range_size,
/// and apply(x, y). An operator may additionally provide apply(x, y, workspace) and/or a scaled-apply
/// apply(alpha, x, beta, y) member, but neither is required by this concept -- both degrade gracefully (see
/// impl::make_workspace and HasScaledApplyMember below).
///
/// Both KokkosBackend and MundyMathBackend dispatch Backend::domain_size/range_size/apply to a
/// MUNDY_THROW_REQUIRE-at-runtime fallback for any Op that isn't otherwise recognized, so those three
/// expressions alone are well-formed (and thus satisfied) for literally any Op -- checking only them would make
/// this concept accept non-operators. The additional clause below requires Op to be one of the backend's native
/// matrices (Backend::is_native_matrix) or a type that provides its own apply member, so a type with neither is
/// correctly rejected instead of silently passing and only failing at runtime.
template <class Backend, class Op, class XVector, class YVector>
concept LinearOperator =
    requires(const Op& op, const XVector& x, YVector& y) {
      { Backend::domain_size(op) } -> std::convertible_to<size_t>;
      { Backend::range_size(op) } -> std::convertible_to<size_t>;
      { Backend::apply(op, x, y) } -> std::same_as<void>;
    } && (Backend::template is_native_matrix<Op> || impl::HasApplyMember<Op, XVector, YVector> ||
          impl::HasApplyMemberWithWorkspace<Op, XVector, YVector, impl::workspace_for_t<Op>>);

/// \brief Concept for an operator that provides its own fused scaled-apply: y := alpha * op(x) + beta * y.
/// Backend::apply(alpha, op, x, beta, y) dispatches here when available, and falls back to a plain
/// apply-then-axpby otherwise -- so an operator need not implement this to be used with a scaled apply, but
/// gets a faster fused path when it does (e.g. to skip recomputing an expensive per-element coefficient when
/// alpha == 0).
template <class Op, class Scalar, class XVector, class YVector>
concept HasScaledApplyMember = requires(const Op& op, Scalar a, const XVector& x, Scalar b, YVector& y) {
  { op.apply(a, x, b, y) } -> std::same_as<void>;
};

/// \brief Backend for Kokkos single process execution
///
/// Applies matrix views, dense (a rank-2 Kokkos::View) or sparse (a KokkosSparse::CrsMatrix), and operators that
/// provide their own apply.
template <typename ExecSpace>
struct KokkosBackend {
 public:
  using exec_space = ExecSpace;

  // Vector and operator sizes are runtime values.
  static constexpr bool has_static_sizes = false;

  /// \brief Whether Op is one of this backend's native matrices: a dense rank-2 view or a sparse matrix.
  template <class Op>
  static constexpr bool is_native_matrix =
#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
      impl::MatView<Op>;
#else
      false;
#endif

 public:
  // The vector factories are host only, but may be called from KOKKOS_FUNCTION code being called on the host
  // This will cause warnings, but is otherwise perfectly valid, so we suppress the warnings for these functions
  MUNDY_SUPPRESS_GPU_CALL_FROM_HOST_WARNINGS_PUSH

  template <class Vector>
  KOKKOS_INLINE_FUNCTION static auto make_vector_like(const Vector& q) {
    // Kokkos::View has no label-less allocating ctor; a size alone resolves to the pointer-wrapping overload.
    KOKKOS_IF_ON_HOST((return Vector("make_vector_like", q.extent(0));));
    KOKKOS_IF_ON_DEVICE((MUNDY_THROW_REQUIRE(false, std::logic_error,
                                             "KokkosBackend::make_vector_like: cannot be called from device code.");
                         return Vector("make_vector_like", q.extent(0));));
  }

  template <class LinearOp>
  KOKKOS_INLINE_FUNCTION static auto make_domain_vector(const LinearOp& op) {
    if constexpr (impl::HasMakeDomainVectorMember<LinearOp>) {
      return op.make_domain_vector();
    } else if constexpr (is_native_matrix<LinearOp>) {
      using op_t = std::remove_reference_t<LinearOp>;
      using value_type = typename op_t::non_const_value_type;
      using mem_space = typename op_t::memory_space;
      using layout_t = typename Kokkos::View<value_type*, mem_space>::array_layout;
      using vector_t = Kokkos::View<value_type*, layout_t, mem_space>;
      KOKKOS_IF_ON_HOST(
          (return vector_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "domain_vector"), domain_size(op));));
      KOKKOS_IF_ON_DEVICE((MUNDY_THROW_REQUIRE(false, std::logic_error,
                                               "KokkosBackend::make_domain_vector: cannot be called from device code.");
                           return vector_t("domain_vector", domain_size(op));));
    } else {
      static_assert(dependent_false_v<LinearOp>,
                    "KokkosBackend::make_domain_vector requires a matrix view or op.make_domain_vector().");
    }
  }

  template <class LinearOp>
  KOKKOS_INLINE_FUNCTION static auto make_range_vector(const LinearOp& op) {
    if constexpr (impl::HasMakeRangeVectorMember<LinearOp>) {
      return op.make_range_vector();
    } else if constexpr (is_native_matrix<LinearOp>) {
      using op_t = std::remove_reference_t<LinearOp>;
      using value_type = typename op_t::non_const_value_type;
      using mem_space = typename op_t::memory_space;
      using layout_t = typename Kokkos::View<value_type*, mem_space>::array_layout;
      using vector_t = Kokkos::View<value_type*, layout_t, mem_space>;
      KOKKOS_IF_ON_HOST(
          (return vector_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "range_vector"), range_size(op));));
      KOKKOS_IF_ON_DEVICE((MUNDY_THROW_REQUIRE(false, std::logic_error,
                                               "KokkosBackend::make_range_vector: cannot be called from device code.");
                           return vector_t("range_vector", range_size(op));));
    } else {
      static_assert(dependent_false_v<LinearOp>,
                    "KokkosBackend::make_range_vector requires a matrix view or op.make_range_vector().");
    }
  }

  /// \brief An uninitialized Vector of the given size.
  template <class Vector>
  KOKKOS_INLINE_FUNCTION static Vector make_vector(size_t size) {
    KOKKOS_IF_ON_HOST((return Vector(Kokkos::view_alloc(Kokkos::WithoutInitializing, "make_vector"), size);));
    KOKKOS_IF_ON_DEVICE(
        (MUNDY_THROW_REQUIRE(false, std::logic_error, "KokkosBackend::make_vector: cannot be called from device code.");
         return Vector("make_vector", size);));
  }

  MUNDY_SUPPRESS_GPU_CALL_FROM_HOST_WARNINGS_POP

  template <class LinearOp>
  static auto make_workspace(const LinearOp& op) {
    return impl::make_workspace(op);
  }

  template <class Vector>
  KOKKOS_INLINE_FUNCTION static size_t size(Vector& x) {
    return x.extent(0);
  }

  template <class LinearOp>
  MUNDY_REQUIRES(is_native_matrix<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t domain_size(LinearOp& op) {
    return impl::mat_view_num_cols(op);
  }
  //
  template <class LinearOp>
  MUNDY_REQUIRES(!is_native_matrix<LinearOp> && impl::HasDomainSizeMember<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t domain_size(LinearOp& op) {
    return op.domain_size();
  }
  //
  template <class LinearOp>
  MUNDY_REQUIRES(!is_native_matrix<LinearOp> && !impl::HasDomainSizeMember<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t domain_size(LinearOp&) {
    MUNDY_THROW_REQUIRE(false, std::logic_error,
                        "KokkosBackend::domain_size: op must be a matrix view or provide size_t domain_size().");
    return 0;
  }

  template <class LinearOp>
  MUNDY_REQUIRES(is_native_matrix<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t range_size(LinearOp& op) {
    return impl::mat_view_num_rows(op);
  }
  //
  template <class LinearOp>
  MUNDY_REQUIRES(!is_native_matrix<LinearOp> && impl::HasRangeSizeMember<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t range_size(LinearOp& op) {
    return op.range_size();
  }
  //
  template <class LinearOp>
  MUNDY_REQUIRES(!is_native_matrix<LinearOp> && !impl::HasRangeSizeMember<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t range_size(LinearOp&) {
    MUNDY_THROW_REQUIRE(false, std::logic_error,
                        "KokkosBackend::range_size: op must be a matrix view or provide size_t range_size().");
    return 0;
  }

  template <class Vector>
  KOKKOS_INLINE_FUNCTION static decltype(auto) vector_data(Vector& x, size_t i) {
    return x(i);
  }

  /// \brief The entries [begin, end) of x, aliasing its storage.
  template <class Vector>
  KOKKOS_INLINE_FUNCTION static auto subvector(const Vector& x, size_t begin, size_t end) {
    return Kokkos::subview(x, Kokkos::pair<size_t, size_t>(begin, end));
  }

  template <class DestVector, class SrcVector>
  static void deep_copy(DestVector& dest, const SrcVector& src) {
    Kokkos::deep_copy(dest, src);
  }

#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
  // Path 1: If op is a matrix view, dense or sparse, multiply by it.
  // y = A*x
  template <class LinearOp, class XVector, class YVector>
  MUNDY_REQUIRES(is_native_matrix<LinearOp>)
  static void apply(const LinearOp& A, const XVector& x, YVector& y) {
    using value_type = impl::vector_value_type<YVector>;
    apply_mat_view(value_type(1), A, x, value_type(0), y);
  }

  // Path 1: If op is a matrix view, dense or sparse, multiply by it.
  // y = alpha * A * x + beta * y
  template <class Scalar, class LinearOp, class XVector, class YVector>
  MUNDY_REQUIRES(is_native_matrix<LinearOp>)
  static void apply(Scalar alpha, const LinearOp& A, const XVector& x, Scalar beta, YVector& y) {
    apply_mat_view(alpha, A, x, beta, y);
  }
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS

  // Path 2: If op has member `apply(x,y)`
  template <class LinearOp, class XVector, class YVector>
  MUNDY_REQUIRES(!is_native_matrix<LinearOp> && impl::HasApplyMember<LinearOp, XVector, YVector>)
  static void apply(const LinearOp& op, const XVector& x, YVector& y) {
    op.apply(x, y);
  }

  // Path 3: Otherwise, runtime error.
  template <typename LinearOp, class XVector, class YVector>
  MUNDY_REQUIRES(!is_native_matrix<LinearOp> && !impl::HasApplyMember<LinearOp, XVector, YVector>)
  static void apply(const LinearOp& op, const XVector& x, YVector& y) {
    MUNDY_THROW_REQUIRE(false, std::logic_error,
                        "KokkosBackend::apply: op must be a matrix view or provide void apply(x,y).");
  }

  /// \brief Dispatch to the matching apply_impl overload below, invalidating the workspace exactly once first.
  template <class LinearOp, class XVector, class YVector, class Workspace>
  static void apply(const LinearOp& op, const XVector& x, YVector& y, Workspace& workspace) {
    impl::workspace_invalidate(workspace);
    apply_impl(op, x, y, workspace);
  }

  // Scaled apply for ops that are not matrix views: y := alpha * op(x) + beta * y.
  // Path 1: op provides its own fused apply(alpha, x, beta, y).
  template <class Scalar, class LinearOp, class XVector, class YVector>
  MUNDY_REQUIRES(!is_native_matrix<LinearOp> && HasScaledApplyMember<LinearOp, Scalar, XVector, YVector>)
  static void apply(Scalar alpha, const LinearOp& op, const XVector& x, Scalar beta, YVector& y) {
    op.apply(alpha, x, beta, y);
  }

  // Path 2: no fused member -- realize it generically from the plain apply plus axpby.
  template <class Scalar, class LinearOp, class XVector, class YVector>
  MUNDY_REQUIRES(!is_native_matrix<LinearOp> && !HasScaledApplyMember<LinearOp, Scalar, XVector, YVector>)
  static void apply(Scalar alpha, const LinearOp& op, const XVector& x, Scalar beta, YVector& y) {
    auto tmp = make_range_vector(op);
    apply(op, x, tmp);
    axpby(alpha, tmp, beta, y);
  }

  // Scaled apply with workspace: y := alpha * op(x) + beta * y, threading the operator's workspace so a scaled
  // apply reaches the same cached scratch as apply(op, x, y, workspace). Mirrors the unworkspaced scaled apply.
#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
  // Path 0: a matrix view carries no workspace.
  template <class Scalar, class LinearOp, class XVector, class YVector, class Workspace>
  MUNDY_REQUIRES(is_native_matrix<LinearOp>)
  static void apply(Scalar alpha, const LinearOp& A, const XVector& x, Scalar beta, YVector& y, Workspace&) {
    apply(alpha, A, x, beta, y);
  }
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS

  // Path 1: op provides its own fused apply(alpha, x, beta, y); its contract carries no workspace.
  template <class Scalar, class LinearOp, class XVector, class YVector, class Workspace>
  MUNDY_REQUIRES(!is_native_matrix<LinearOp> && HasScaledApplyMember<LinearOp, Scalar, XVector, YVector>)
  static void apply(Scalar alpha, const LinearOp& op, const XVector& x, Scalar beta, YVector& y, Workspace& workspace) {
    impl::workspace_invalidate(workspace);
    op.apply(alpha, x, beta, y);
  }

  // Path 2: no fused member -- realize it from the workspace-threaded plain apply plus axpby.
  template <class Scalar, class LinearOp, class XVector, class YVector, class Workspace>
  MUNDY_REQUIRES(!is_native_matrix<LinearOp> && !HasScaledApplyMember<LinearOp, Scalar, XVector, YVector>)
  static void apply(Scalar alpha, const LinearOp& op, const XVector& x, Scalar beta, YVector& y, Workspace& workspace) {
    auto tmp = make_range_vector(op);
    apply(op, x, tmp, workspace);
    axpby(alpha, tmp, beta, y);
  }

 private:
#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
  // y := alpha * A * x + beta * y for a matrix view A, by gemv if dense and spmv if sparse.
  template <class Scalar, class Matrix, class XVector, class YVector>
  static void apply_mat_view(Scalar alpha, const Matrix& A, const XVector& x, Scalar beta, YVector& y) {
    MUNDY_THROW_ASSERT(domain_size(A) == x.extent(0), std::invalid_argument,
                       "KokkosBackend::apply: A's column count must equal x's size.");
    MUNDY_THROW_ASSERT(range_size(A) == y.extent(0), std::invalid_argument,
                       "KokkosBackend::apply: A's row count must equal y's size.");
    if constexpr (impl::DenseMatView<Matrix>) {
      KokkosBlas::gemv(exec_space{}, "N", alpha, A, x, beta, y);
    } else {
      KokkosSparse::spmv(exec_space{}, "N", alpha, A, x, beta, y);
    }
  }

  template <class LinearOp, class XVector, class YVector, class Workspace>
  MUNDY_REQUIRES(is_native_matrix<LinearOp>)
  static void apply_impl(const LinearOp& A, const XVector& x, YVector& y, Workspace&) {
    apply(A, x, y);
  }
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS

  template <class LinearOp, class XVector, class YVector, class Workspace>
  MUNDY_REQUIRES(!is_native_matrix<LinearOp> &&
                 impl::HasApplyMemberWithWorkspace<LinearOp, XVector, YVector, Workspace>)
  static void apply_impl(const LinearOp& op, const XVector& x, YVector& y, Workspace& workspace) {
    op.apply(x, y, workspace);
  }

  template <class LinearOp, class XVector, class YVector, class Workspace>
  MUNDY_REQUIRES(!is_native_matrix<LinearOp> &&
                 !impl::HasApplyMemberWithWorkspace<LinearOp, XVector, YVector, Workspace> &&
                 impl::HasApplyMember<LinearOp, XVector, YVector>)
  static void apply_impl(const LinearOp& op, const XVector& x, YVector& y, Workspace&) {
    op.apply(x, y);
  }

  template <typename LinearOp, class XVector, class YVector, class Workspace>
  MUNDY_REQUIRES(!is_native_matrix<LinearOp> &&
                 !impl::HasApplyMemberWithWorkspace<LinearOp, XVector, YVector, Workspace> &&
                 !impl::HasApplyMember<LinearOp, XVector, YVector>)
  static void apply_impl(const LinearOp&, const XVector&, YVector&, Workspace&) {
    MUNDY_THROW_REQUIRE(false, std::logic_error,
                        "KokkosBackend::apply: op must be a matrix view or provide void apply(x,y[,workspace]).");
  }

 public:
  template <class Scalar, class XVector, class YVector>
  static void axpby(const Scalar alpha, const XVector& x, const Scalar beta, YVector& y) {
    MUNDY_THROW_ASSERT(x.extent(0) == y.extent(0), std::invalid_argument, "x and y must have the same size.");
    const bool alpha_is_zero = abs(alpha) < get_zero_tolerance<Scalar>();
    const bool beta_is_zero = abs(beta) < get_zero_tolerance<Scalar>();

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

  /// \brief z := x .* y, elementwise.
  template <class XVector, class YVector, class ZVector>
  static void elementwise_mul(const XVector& x, const YVector& y, ZVector& z) {
    MUNDY_THROW_ASSERT(x.extent(0) == y.extent(0) && y.extent(0) == z.extent(0), std::invalid_argument,
                       "x, y, and z must have the same size.");
    Kokkos::parallel_for(
        "elementwise_mul", Kokkos::RangePolicy<exec_space>(0, x.extent(0)),
        KOKKOS_LAMBDA(const int i) { z(i) = x(i) * y(i); });
  }

  /// \brief z := x ./ y, elementwise.
  template <class XVector, class YVector, class ZVector>
  static void elementwise_div(const XVector& x, const YVector& y, ZVector& z) {
    MUNDY_THROW_ASSERT(x.extent(0) == y.extent(0) && y.extent(0) == z.extent(0), std::invalid_argument,
                       "x, y, and z must have the same size.");
    Kokkos::parallel_for(
        "elementwise_div", Kokkos::RangePolicy<exec_space>(0, x.extent(0)),
        KOKKOS_LAMBDA(const int i) { z(i) = x(i) / y(i); });
  }

  template <typename Wrapper, class Scalar, class XVector, class YVector, class ZVector>
  static void wrapped_axpbyz(const Scalar alpha, const XVector& x, const Scalar beta, const YVector& y, ZVector& z,
                             const Wrapper& wrapper) {
    MUNDY_THROW_ASSERT(x.extent(0) == y.extent(0) && x.extent(0) == z.extent(0), std::invalid_argument,
                       "x, y, and z must have the same size.");
    const bool alpha_is_zero = abs(alpha) < get_zero_tolerance<Scalar>();
    const bool beta_is_zero = abs(beta) < get_zero_tolerance<Scalar>();

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
  static ReductionScalar dot(const XVector& x, const YVector& y) {
    MUNDY_THROW_ASSERT(x.extent(0) == y.extent(0), std::invalid_argument, "x and y must have the same size.");
    ReductionScalar result = 0;
    Kokkos::parallel_reduce(
        "dot", Kokkos::RangePolicy<exec_space>(0, x.extent(0)),
        KOKKOS_LAMBDA(const int i, ReductionScalar& sum) { sum += x(i) * y(i); }, result);
    return result;
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
  // Vector and operator sizes are compile-time constants.
  static constexpr bool has_static_sizes = true;

  /// \brief Whether Op is one of this backend's native matrices: a mundy::Matrix.
  template <class Op>
  static constexpr bool is_native_matrix = is_matrix_v<Op>;

  template <class Vector>
  KOKKOS_INLINE_FUNCTION static auto make_vector_like(const Vector& /*x*/) {
    return Vector();
  }

  /// \brief A Vector of Size Scalars.
  template <class Scalar, size_t Size>
  KOKKOS_INLINE_FUNCTION static auto make_vector() {
    return ::mundy::Vector<Scalar, Size>();
  }

  /// \brief A Vector of the given size, which must equal Vector's compile-time size.
  template <class Vector>
  KOKKOS_INLINE_FUNCTION static Vector make_vector(size_t size) {
    MUNDY_THROW_ASSERT(size == Vector::size, std::invalid_argument,
                       "MundyMathBackend::make_vector: size must equal the vector's compile-time size.");
    return Vector();
  }

  template <class LinearOp>
  KOKKOS_INLINE_FUNCTION static auto make_domain_vector(const LinearOp& op) {
    if constexpr (impl::HasMakeDomainVectorMember<LinearOp>) {
      return op.make_domain_vector();
    } else if constexpr (is_native_matrix<LinearOp>) {
      using op_t = std::remove_reference_t<LinearOp>;
      return Vector<typename op_t::value_type, op_t::num_cols>{};
    } else {
      static_assert(dependent_false_v<LinearOp>,
                    "MundyMathBackend::make_domain_vector requires a native matrix or op.make_domain_vector().");
    }
  }

  template <class LinearOp>
  KOKKOS_INLINE_FUNCTION static auto make_range_vector(const LinearOp& op) {
    if constexpr (impl::HasMakeRangeVectorMember<LinearOp>) {
      return op.make_range_vector();
    } else if constexpr (is_native_matrix<LinearOp>) {
      using op_t = std::remove_reference_t<LinearOp>;
      return Vector<typename op_t::value_type, op_t::num_rows>{};
    } else {
      static_assert(dependent_false_v<LinearOp>,
                    "MundyMathBackend::make_range_vector requires a native matrix or op.make_range_vector().");
    }
  }

  template <class LinearOp>
  KOKKOS_INLINE_FUNCTION static auto make_workspace(const LinearOp& op) {
    return impl::make_workspace(op);
  }

  template <class Vector>
  KOKKOS_INLINE_FUNCTION static size_t size(const Vector& /*x*/) {
    return std::remove_reference_t<Vector>::size;
  }

  template <class LinearOp>
  MUNDY_REQUIRES(is_native_matrix<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t domain_size(LinearOp& /*op*/) {
    return std::remove_reference_t<LinearOp>::num_cols;
  }
  //
  template <class LinearOp>
  MUNDY_REQUIRES(!is_native_matrix<LinearOp> && impl::HasDomainSizeMember<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t domain_size(LinearOp& op) {
    return op.domain_size();
  }
  //
  template <class LinearOp>
  MUNDY_REQUIRES(!is_native_matrix<LinearOp> && !impl::HasDomainSizeMember<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t domain_size(LinearOp& /*op*/) {
    MUNDY_THROW_REQUIRE(false, std::logic_error,
                        "MundyMathBackend::domain_size: op must be a mundy::Matrix or provide size_t domain_size().");
    return 0;
  }

  template <class LinearOp>
  MUNDY_REQUIRES(is_native_matrix<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t range_size(LinearOp& /*op*/) {
    return std::remove_reference_t<LinearOp>::num_rows;
  }
  //
  template <class LinearOp>
  MUNDY_REQUIRES(!is_native_matrix<LinearOp> && impl::HasRangeSizeMember<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t range_size(LinearOp& op) {
    return op.range_size();
  }
  //
  template <class LinearOp>
  MUNDY_REQUIRES(!is_native_matrix<LinearOp> && !impl::HasRangeSizeMember<LinearOp>)
  KOKKOS_INLINE_FUNCTION static size_t range_size(LinearOp& /*op*/) {
    MUNDY_THROW_REQUIRE(false, std::logic_error,
                        "MundyMathBackend::range_size: op must be a mundy::Matrix or provide size_t range_size().");
    return 0;
  }

  /// \brief The compile-time size of a Vector.
  template <class Vector>
  KOKKOS_INLINE_FUNCTION static constexpr size_t static_size() {
    return std::remove_cvref_t<Vector>::size;
  }

  /// \brief The compile-time domain size of a mundy::Matrix or of an op providing static_domain_size().
  template <class LinearOp>
  KOKKOS_INLINE_FUNCTION static constexpr size_t static_domain_size() {
    using op_t = std::remove_cvref_t<LinearOp>;
    if constexpr (is_native_matrix<op_t>) {
      return op_t::num_cols;
    } else if constexpr (requires { op_t::static_domain_size(); }) {
      return op_t::static_domain_size();
    } else {
      static_assert(dependent_false_v<op_t>,
                    "MundyMathBackend::static_domain_size: op must be a mundy::Matrix or "
                    "provide static_domain_size().");
    }
  }

  /// \brief The compile-time range size of a mundy::Matrix or of an op providing static_range_size().
  template <class LinearOp>
  KOKKOS_INLINE_FUNCTION static constexpr size_t static_range_size() {
    using op_t = std::remove_cvref_t<LinearOp>;
    if constexpr (is_native_matrix<op_t>) {
      return op_t::num_rows;
    } else if constexpr (requires { op_t::static_range_size(); }) {
      return op_t::static_range_size();
    } else {
      static_assert(dependent_false_v<op_t>,
                    "MundyMathBackend::static_range_size: op must be a mundy::Matrix or provide static_range_size().");
    }
  }

  template <class Vector>
  KOKKOS_INLINE_FUNCTION static decltype(auto) vector_data(const Vector& x, size_t i) {
    return x[i];
  }

  template <class Vector>
  KOKKOS_INLINE_FUNCTION static decltype(auto) vector_data(Vector& x, size_t i) {
    return x[i];
  }

  /// \brief The Size entries of x starting at Offset, aliasing its storage.
  template <size_t Offset, size_t Size, class Vector>
  KOKKOS_INLINE_FUNCTION static auto subvector(Vector& x) {
    return x.template view_subset<Offset, Size>();
  }

  template <class DestVector, class SrcVector>
  KOKKOS_INLINE_FUNCTION static void deep_copy(DestVector& dest, const SrcVector& src) {
    dest = src;
  }

  template <typename LinearOp, class XVector, class YVector>
  KOKKOS_INLINE_FUNCTION static void apply(const LinearOp& op, const XVector& x, YVector& y) {
    if constexpr (impl::HasApplyMember<LinearOp, XVector, YVector>) {
      op.apply(x, y);
    } else if constexpr (is_native_matrix<LinearOp>) {
      y = op * x;
    } else {
      static_assert(dependent_false_v<LinearOp>,
                    "MundyMathBackend::apply: op must be a native matrix or provide void apply(x,y).");
    }
  }

  /// \brief Dispatch to apply_impl below, invalidating the workspace exactly once first.
  template <typename LinearOp, class XVector, class YVector, typename Workspace>
  KOKKOS_INLINE_FUNCTION static void apply(const LinearOp& op, const XVector& x, YVector& y, Workspace& workspace) {
    impl::workspace_invalidate(workspace);
    apply_impl(op, x, y, workspace);
  }

  // Scaled apply: y := alpha * op(x) + beta * y.
  // Path 1: op provides its own fused apply(alpha, x, beta, y).
  template <class Scalar, typename LinearOp, class XVector, class YVector>
  MUNDY_REQUIRES(HasScaledApplyMember<LinearOp, Scalar, XVector, YVector>)
  KOKKOS_INLINE_FUNCTION
      static void apply(Scalar alpha, const LinearOp& op, const XVector& x, Scalar beta, YVector& y) {
    op.apply(alpha, x, beta, y);
  }

  // Path 2: no fused member -- realize it generically from the plain apply plus axpby.
  template <class Scalar, typename LinearOp, class XVector, class YVector>
  MUNDY_REQUIRES(!HasScaledApplyMember<LinearOp, Scalar, XVector, YVector>)
  KOKKOS_INLINE_FUNCTION
      static void apply(Scalar alpha, const LinearOp& op, const XVector& x, Scalar beta, YVector& y) {
    auto tmp = make_range_vector(op);
    apply(op, x, tmp);
    axpby(alpha, tmp, beta, y);
  }

  // Scaled apply with workspace: y := alpha * op(x) + beta * y, threading the operator's workspace.
  // Path 1: op provides its own fused apply(alpha, x, beta, y); its contract carries no workspace.
  template <class Scalar, typename LinearOp, class XVector, class YVector, typename Workspace>
  MUNDY_REQUIRES(HasScaledApplyMember<LinearOp, Scalar, XVector, YVector>)
  KOKKOS_INLINE_FUNCTION static void apply(Scalar alpha, const LinearOp& op, const XVector& x, Scalar beta, YVector& y,
                                           Workspace& workspace) {
    impl::workspace_invalidate(workspace);
    op.apply(alpha, x, beta, y);
  }

  // Path 2: no fused member -- realize it from the workspace-threaded plain apply plus axpby.
  template <class Scalar, typename LinearOp, class XVector, class YVector, typename Workspace>
  MUNDY_REQUIRES(!HasScaledApplyMember<LinearOp, Scalar, XVector, YVector>)
  KOKKOS_INLINE_FUNCTION static void apply(Scalar alpha, const LinearOp& op, const XVector& x, Scalar beta, YVector& y,
                                           Workspace& workspace) {
    auto tmp = make_range_vector(op);
    apply(op, x, tmp, workspace);
    axpby(alpha, tmp, beta, y);
  }

 private:
  template <typename LinearOp, class XVector, class YVector, typename Workspace>
  KOKKOS_INLINE_FUNCTION static void apply_impl(const LinearOp& op, const XVector& x, YVector& y,
                                                Workspace& workspace) {
    if constexpr (impl::HasApplyMemberWithWorkspace<LinearOp, XVector, YVector, Workspace>) {
      op.apply(x, y, workspace);
    } else if constexpr (impl::HasApplyMember<LinearOp, XVector, YVector>) {
      op.apply(x, y);
    } else if constexpr (is_native_matrix<LinearOp>) {
      y = op * x;
    } else {
      static_assert(dependent_false_v<LinearOp>,
                    "MundyMathBackend::apply: op must be a native matrix or provide void apply(x,y[,workspace]).");
    }
  }

 public:
  template <class Scalar, class XVector, class YVector>
  KOKKOS_INLINE_FUNCTION static void axpby(const Scalar alpha, const XVector& x, const Scalar beta, YVector& y) {
    y = alpha * x + beta * y;
  }

  /// \brief z := x .* y, elementwise.
  template <class XVector, class YVector, class ZVector>
  KOKKOS_INLINE_FUNCTION static void elementwise_mul(const XVector& x, const YVector& y, ZVector& z) {
    z = ::mundy::elementwise_mul(x, y);
  }

  /// \brief z := x ./ y, elementwise.
  template <class XVector, class YVector, class ZVector>
  KOKKOS_INLINE_FUNCTION static void elementwise_div(const XVector& x, const YVector& y, ZVector& z) {
    z = ::mundy::elementwise_div(x, y);
  }

  template <typename Wrapper, class Scalar, class XVector, class YVector, class ZVector>
  KOKKOS_INLINE_FUNCTION static void wrapped_axpbyz(const Scalar alpha, const XVector& x, const Scalar beta,
                                                    const YVector& y, ZVector& z, const Wrapper& wrapper) {
    z = ::mundy::apply(wrapper, alpha * x + beta * y);
  }

  template <class ReductionScalar, class XVector, class YVector>
  KOKKOS_INLINE_FUNCTION static ReductionScalar dot(const XVector& x, const YVector& y) {
    return static_cast<ReductionScalar>(::mundy::dot(x, y));
  }

  template <class ReductionScalar, class XVector, class YVector>
  KOKKOS_INLINE_FUNCTION static ReductionScalar diff_dot(const XVector& x, const YVector& y) {
    auto diff = x - y;
    return static_cast<ReductionScalar>(::mundy::dot(diff, diff));
  }

  template <class ReductionScalar, class X1Vector, class X2Vector, class Y1Vector, class Y2Vector>
  KOKKOS_INLINE_FUNCTION static ReductionScalar diff_dot(const X1Vector& x1, const X2Vector& x2,  //
                                                         const Y1Vector& y1, const Y2Vector& y2) {
    auto x_diff = x1 - x2;
    auto y_diff = y1 - y2;
    return static_cast<ReductionScalar>(::mundy::dot(x_diff, y_diff));
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
    ReductionScalar max_val = -NumTraits<ReductionScalar>::infinity();
    ((func(Is, max_val)), ...);
    result = max_val;
  }
};  // MundyMathBackend
//@}

}  // namespace mundy

#endif  // MUNDY_MATH_SOLVER_BACKENDS_HPP_
