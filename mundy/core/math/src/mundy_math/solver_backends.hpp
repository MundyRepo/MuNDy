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
#include <tuple>
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

//! \name Vector expressions
//@{
// A vector expression is a lazy vector whose entry i depends only on entry i of its operands. e(i) evaluates it, and
// nothing is stored until a backend assigns or reduces it. Expressions start from a backend's entries(x); the
// operators below combine them, entry by entry, with each other and with scalars.

/// \brief Concept for a vector expression: a value_type, entries e(i), and a size.
template <class E>
concept BackendVectorExpr = impl::HasVectorExprMarker<E> && requires(const std::remove_cvref_t<E>& e, size_t i) {
  typename std::remove_cvref_t<E>::value_type;
  { e(i) } -> std::convertible_to<typename std::remove_cvref_t<E>::value_type>;
  { e.size() } -> std::convertible_to<size_t>;
};

/// \brief The entries of a backend vector x: e(i) = x_i.
///
/// An expression referring to an lvalue vector must not outlive it.
template <class Backend, class Vector>
class EntriesExpr {
 public:
  static constexpr bool is_vector_expression = true;
  using value_type = impl::vector_value_type<std::remove_cvref_t<Vector>>;

  KOKKOS_INLINE_FUNCTION explicit EntriesExpr(Vector&& x) : x_(std::forward<Vector>(x)) {
  }

  KOKKOS_INLINE_FUNCTION value_type operator()(size_t i) const {
    return Backend::vector_data(x_.get(), i);
  }

  KOKKOS_INLINE_FUNCTION size_t size() const {
    return Backend::size(x_.get());
  }

  KOKKOS_INLINE_FUNCTION static constexpr size_t static_size() MUNDY_REQUIRES(Backend::has_static_sizes) {
    return Backend::template static_size<Vector>();
  }

 private:
  ::mundy::storage<Vector> x_;
};

/// \brief (-a)_i = -a_i.
template <class A>
class NegatedExpr : public impl::UnaryOperand<A> {
 public:
  static constexpr bool is_vector_expression = true;
  using value_type = std::remove_cvref_t<decltype(-std::declval<typename A::value_type>())>;

  KOKKOS_INLINE_FUNCTION explicit NegatedExpr(const A& a) : impl::UnaryOperand<A>(a) {
  }

  KOKKOS_INLINE_FUNCTION value_type operator()(size_t i) const {
    return -this->operand(i);
  }
};

/// \brief |a|_i = |a_i|.
template <class A>
class AbsExpr : public impl::UnaryOperand<A> {
 public:
  static constexpr bool is_vector_expression = true;
  using value_type = std::remove_cvref_t<decltype(::mundy::abs(std::declval<typename A::value_type>()))>;

  KOKKOS_INLINE_FUNCTION explicit AbsExpr(const A& a) : impl::UnaryOperand<A>(a) {
  }

  KOKKOS_INLINE_FUNCTION value_type operator()(size_t i) const {
    return ::mundy::abs(this->operand(i));
  }
};

/// \brief (a + b)_i = a_i + b_i, where a scalar operand stands for itself at every entry.
template <class A, class B>
class SumExpr : public impl::BinaryOperands<A, B> {
 public:
  static constexpr bool is_vector_expression = true;
  using value_type = std::remove_cvref_t<decltype(std::declval<impl::operand_value_t<A>>() +
                                                  std::declval<impl::operand_value_t<B>>())>;

  KOKKOS_INLINE_FUNCTION SumExpr(const A& a, const B& b) : impl::BinaryOperands<A, B>(a, b) {
  }

  KOKKOS_INLINE_FUNCTION value_type operator()(size_t i) const {
    return this->left(i) + this->right(i);
  }
};

/// \brief (a - b)_i = a_i - b_i, where a scalar operand stands for itself at every entry.
template <class A, class B>
class DifferenceExpr : public impl::BinaryOperands<A, B> {
 public:
  static constexpr bool is_vector_expression = true;
  using value_type = std::remove_cvref_t<decltype(std::declval<impl::operand_value_t<A>>() -
                                                  std::declval<impl::operand_value_t<B>>())>;

  KOKKOS_INLINE_FUNCTION DifferenceExpr(const A& a, const B& b) : impl::BinaryOperands<A, B>(a, b) {
  }

  KOKKOS_INLINE_FUNCTION value_type operator()(size_t i) const {
    return this->left(i) - this->right(i);
  }
};

/// \brief (a b)_i = a_i b_i, where a scalar operand stands for itself at every entry.
template <class A, class B>
class ProductExpr : public impl::BinaryOperands<A, B> {
 public:
  static constexpr bool is_vector_expression = true;
  using value_type = std::remove_cvref_t<decltype(std::declval<impl::operand_value_t<A>>() *
                                                  std::declval<impl::operand_value_t<B>>())>;

  KOKKOS_INLINE_FUNCTION ProductExpr(const A& a, const B& b) : impl::BinaryOperands<A, B>(a, b) {
  }

  KOKKOS_INLINE_FUNCTION value_type operator()(size_t i) const {
    return this->left(i) * this->right(i);
  }
};

/// \brief (a / b)_i = a_i / b_i, where a scalar operand stands for itself at every entry.
template <class A, class B>
class QuotientExpr : public impl::BinaryOperands<A, B> {
 public:
  static constexpr bool is_vector_expression = true;
  using value_type = std::remove_cvref_t<decltype(std::declval<impl::operand_value_t<A>>() /
                                                  std::declval<impl::operand_value_t<B>>())>;

  KOKKOS_INLINE_FUNCTION QuotientExpr(const A& a, const B& b) : impl::BinaryOperands<A, B>(a, b) {
  }

  KOKKOS_INLINE_FUNCTION value_type operator()(size_t i) const {
    return this->left(i) / this->right(i);
  }
};

template <class A>
MUNDY_REQUIRES(impl::HasVectorExprMarker<A>)
KOKKOS_INLINE_FUNCTION auto operator-(const A& a) {
  return NegatedExpr<A>(a);
}

template <class A>
MUNDY_REQUIRES(impl::HasVectorExprMarker<A>)
KOKKOS_INLINE_FUNCTION auto abs(const A& a) {
  return AbsExpr<A>(a);
}

template <class A, class B>
MUNDY_REQUIRES(impl::EntrywiseOperands<A, B>)
KOKKOS_INLINE_FUNCTION auto operator+(const A& a, const B& b) {
  return SumExpr<A, B>(a, b);
}

template <class A, class B>
MUNDY_REQUIRES(impl::EntrywiseOperands<A, B>)
KOKKOS_INLINE_FUNCTION auto operator-(const A& a, const B& b) {
  return DifferenceExpr<A, B>(a, b);
}

template <class A, class B>
MUNDY_REQUIRES(impl::EntrywiseOperands<A, B>)
KOKKOS_INLINE_FUNCTION auto operator*(const A& a, const B& b) {
  return ProductExpr<A, B>(a, b);
}

template <class A, class B>
MUNDY_REQUIRES(impl::EntrywiseOperands<A, B>)
KOKKOS_INLINE_FUNCTION auto operator/(const A& a, const B& b) {
  return QuotientExpr<A, B>(a, b);
}
//@}

//! \name Backend contract machinery
//@{

/// \brief Concept for a backend's vectors: their size, their entries as an expression, and a deep copy.
template <class Backend, class Vector>
concept VectorBackend = requires(Vector& y, const Vector& x) {
  { Backend::size(x) } -> std::convertible_to<size_t>;
  { Backend::entries(x) } -> BackendVectorExpr;
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

//@}

//! \name Vector operations
//@{
// The common operations on vectors, for every backend. Each takes the backend first, then vectors or expressions:
//   const double r = dot(backend, x, y);
//   axpby(backend, a, x, b, y);
//   const double largest = reduce_max(backend, abs(backend_t::entries(x)));

/// \brief The sum of e's entries, or 0 if it has none. e is a vector or an expression.
template <class Backend, class E>
MUNDY_REQUIRES(BackendVectorExpr<E> || VectorBackend<Backend, E>)
KOKKOS_FUNCTION auto reduce_sum(const Backend&, const E& e) {
  const auto expression = Backend::entries(e);
  using value_type = typename std::remove_cvref_t<decltype(expression)>::value_type;
  value_type sum;
  Backend::reduce("mundy::reduce_sum", Kokkos::Sum<value_type>(sum), expression);
  return sum;
}

/// \brief The least of e's entries, capped at the largest finite value, which it also is if e has none. e is a vector
/// or an expression.
template <class Backend, class E>
MUNDY_REQUIRES(BackendVectorExpr<E> || VectorBackend<Backend, E>)
KOKKOS_FUNCTION auto reduce_min(const Backend&, const E& e) {
  const auto expression = Backend::entries(e);
  using value_type = typename std::remove_cvref_t<decltype(expression)>::value_type;
  value_type least;
  Backend::reduce("mundy::reduce_min", Kokkos::Min<value_type>(least), expression);
  return least;
}

/// \brief The greatest of e's entries, floored at the lowest finite value, which it also is if e has none. e is a
/// vector or an expression.
template <class Backend, class E>
MUNDY_REQUIRES(BackendVectorExpr<E> || VectorBackend<Backend, E>)
KOKKOS_FUNCTION auto reduce_max(const Backend&, const E& e) {
  const auto expression = Backend::entries(e);
  using value_type = typename std::remove_cvref_t<decltype(expression)>::value_type;
  value_type greatest;
  Backend::reduce("mundy::reduce_max", Kokkos::Max<value_type>(greatest), expression);
  return greatest;
}

/// \brief x^T y.
template <class Backend, class X, class Y>
MUNDY_REQUIRES(VectorBackend<Backend, X>)
KOKKOS_FUNCTION auto dot(const Backend& backend, const X& x, const Y& y) {
  return reduce_sum(backend, Backend::entries(x) * Backend::entries(y));
}

/// \brief y := a x + b y. Neither x nor y is read where its coefficient is exactly 0, so either may then hold anything.
template <class Backend, class Scalar, class X, class Y>
MUNDY_REQUIRES(VectorBackend<Backend, Y>)
KOKKOS_FUNCTION void axpby(const Backend&, const Scalar a, const X& x, const Scalar b, Y& y) {
  const Scalar zero = static_cast<Scalar>(0);
  if (b == zero) {
    if (a == zero) {
      Backend::assign("mundy::axpby", y, zero);
    } else {
      Backend::assign("mundy::axpby", y, a * Backend::entries(x));
    }
  } else if (a == zero) {
    Backend::assign("mundy::axpby", y, b * Backend::entries(y));
  } else {
    Backend::assign("mundy::axpby", y, a * Backend::entries(x) + b * Backend::entries(y));
  }
}

/// \brief z := x .* y, entry by entry.
template <class Backend, class X, class Y, class Z>
MUNDY_REQUIRES(VectorBackend<Backend, Z>)
KOKKOS_FUNCTION void elementwise_mul(const Backend&, const X& x, const Y& y, Z& z) {
  Backend::assign("mundy::elementwise_mul", z, Backend::entries(x) * Backend::entries(y));
}

/// \brief z := x ./ y, entry by entry.
template <class Backend, class X, class Y, class Z>
MUNDY_REQUIRES(VectorBackend<Backend, Z>)
KOKKOS_FUNCTION void elementwise_div(const Backend&, const X& x, const Y& y, Z& z) {
  Backend::assign("mundy::elementwise_div", z, Backend::entries(x) / Backend::entries(y));
}
//@}

//! \name Backends
//@{

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

  /// \brief Whether A holds contiguous scalars a solver's driver reads and writes on the host: a Kokkos::Array, or a
  /// rank-1 LayoutLeft or LayoutRight view the host can access.
  template <class A>
  static constexpr bool is_scalar_array = impl::IsKokkosArray<std::remove_cvref_t<A>>::value || [] {
    if constexpr (Kokkos::is_view_v<std::remove_cvref_t<A>>) {
      using view_t = std::remove_cvref_t<A>;
      using layout_t = typename view_t::array_layout;
      return view_t::rank == 1 &&
             (std::same_as<layout_t, Kokkos::LayoutLeft> || std::same_as<layout_t, Kokkos::LayoutRight>) &&
             Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename view_t::memory_space>::accessible;
    } else {
      return false;
    }
  }();

  /// \brief The number of vectors in a multivector: a rank-2 LayoutLeft view with K > 0 columns fixed at compile time.
  template <class MultiVector>
  KOKKOS_INLINE_FUNCTION static constexpr size_t num_columns() {
    using mv_t = std::remove_cvref_t<MultiVector>;
    static_assert(Kokkos::is_view_v<mv_t>, "KokkosBackend: a multivector is a rank-2 Kokkos::View.");
    static_assert(mv_t::rank == 2 && std::same_as<typename mv_t::array_layout, Kokkos::LayoutLeft>,
                  "KokkosBackend: a multivector is a rank-2 LayoutLeft view, so each column is contiguous.");
    static_assert(mv_t::static_extent(1) > 0,
                  "KokkosBackend: a multivector's column count is a compile-time extent, as View<Scalar*[K]>.");
    return mv_t::static_extent(1);
  }

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

  /// \brief Column j of the multivector mv, aliasing its storage.
  template <class MultiVector>
  KOKKOS_INLINE_FUNCTION static auto column(const MultiVector& mv, size_t j) {
    return Kokkos::subview(mv, Kokkos::ALL, j);
  }

  /// \brief sum_{j < count} c_j column_j of the multivector mv; columns past count are not read.
  template <class MultiVector, class Scalar, size_t K>
  KOKKOS_INLINE_FUNCTION static auto combination(const MultiVector& mv, size_t count,
                                                 const Kokkos::Array<Scalar, K>& c) {
    static_assert(K == num_columns<MultiVector>(), "KokkosBackend::combination: one coefficient per column.");
    return impl::ColumnCombinationExpr<std::remove_cvref_t<MultiVector>, Scalar, K>(mv, count, c);
  }

  template <class DestVector, class SrcVector>
  static void deep_copy(DestVector& dest, const SrcVector& src) {
    Kokkos::deep_copy(dest, src);
  }

  /// \brief The entries of x as an expression; an expression is its own entries.
  template <class Vector>
  MUNDY_REQUIRES(BackendVectorExpr<Vector> || impl::HasIndexedEntries<Vector>)
  KOKKOS_INLINE_FUNCTION static auto entries(const Vector& x) {
    if constexpr (BackendVectorExpr<Vector>) {
      return x;
    } else {
      return EntriesExpr<KokkosBackend, Vector>(Vector(x));
    }
  }

  /// \brief Writes expressions into vectors, entry by entry: target = source for each (target, source) pair.
  ///
  /// \code
  ///   const auto X = KokkosBackend::entries(x), Y = KokkosBackend::entries(y);
  ///   KokkosBackend::assign("axpby", y, a * X + b * Y);         // y = a x + b y
  ///   KokkosBackend::assign("swap", x, Y, y, X);               // x and y exchange entries
  ///   KokkosBackend::assign("zero", y, 0.0);                   // y = 0
  /// \endcode
  ///
  /// After the label, which names the pass for profiling, the arguments come in pairs: a target vector, then its
  /// source, which is an expression, a vector, or a scalar written to every entry. All pairs run in one pass. At each
  /// entry every source is evaluated before any target is written, so a target may also appear in any source. Every
  /// target and source has the same size.
  template <class... TargetsAndSources>
  static void assign(const char* label, TargetsAndSources&&... targets_and_sources) {
    static_assert(sizeof...(TargetsAndSources) > 0 && sizeof...(TargetsAndSources) % 2 == 0,
                  "KokkosBackend::assign takes (target, source) pairs.");
    assign_impl(label, std::forward_as_tuple(targets_and_sources...),
                std::index_sequence_for<TargetsAndSources...>{});
  }

  /// \brief Reduces expressions over their entries: one result for each (reducer, source) pair.
  ///
  /// \code
  ///   const auto X = KokkosBackend::entries(x), Y = KokkosBackend::entries(y);
  ///   double x_dot_y, largest;
  ///   KokkosBackend::reduce("norms", Kokkos::Sum<double>(x_dot_y), X * Y,   // x_dot_y = x^T y
  ///                         Kokkos::Max<double>(largest), abs(X));          // largest = max_i |x_i|
  /// \endcode
  ///
  /// After the label, which names the pass for profiling, the arguments come in pairs: a Kokkos reducer bound to its
  /// result, such as Kokkos::Sum, Min, or Max, then its source, which is an expression or a vector. All pairs run in
  /// one pass. A source with no entries leaves its reducer's identity: 0 for Sum, the largest finite value for Min,
  /// and the lowest for Max.
  template <class... ReducersAndSources>
  static void reduce(const char* label, const ReducersAndSources&... reducers_and_sources) {
    static_assert(sizeof...(ReducersAndSources) > 0 && sizeof...(ReducersAndSources) % 2 == 0,
                  "KokkosBackend::reduce takes (reducer, source) pairs.");
    reduce_impl(label, std::forward_as_tuple(reducers_and_sources...),
                std::make_index_sequence<sizeof...(ReducersAndSources) / 2>{});
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

  // Apply with workspace: y = op(x), threading the operator's workspace to an op that takes one.
#ifdef HAVE_MUNDYMATH_KOKKOSKERNELS
  // Path 1: a matrix view carries no workspace.
  template <class LinearOp, class XVector, class YVector, class Workspace>
  MUNDY_REQUIRES(is_native_matrix<LinearOp>)
  static void apply(const LinearOp& A, const XVector& x, YVector& y, Workspace&) {
    apply(A, x, y);
  }
#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS

  // Path 2: op has member `apply(x, y, workspace)`.
  template <class LinearOp, class XVector, class YVector, class Workspace>
  MUNDY_REQUIRES(!is_native_matrix<LinearOp> &&
                 impl::HasApplyMemberWithWorkspace<LinearOp, XVector, YVector, Workspace>)
  static void apply(const LinearOp& op, const XVector& x, YVector& y, Workspace& workspace) {
    op.apply(x, y, workspace);
  }

  // Path 3: op has member `apply(x, y)` only.
  template <class LinearOp, class XVector, class YVector, class Workspace>
  MUNDY_REQUIRES(!is_native_matrix<LinearOp> &&
                 !impl::HasApplyMemberWithWorkspace<LinearOp, XVector, YVector, Workspace> &&
                 impl::HasApplyMember<LinearOp, XVector, YVector>)
  static void apply(const LinearOp& op, const XVector& x, YVector& y, Workspace&) {
    op.apply(x, y);
  }

  // Path 4: Otherwise, runtime error.
  template <typename LinearOp, class XVector, class YVector, class Workspace>
  MUNDY_REQUIRES(!is_native_matrix<LinearOp> &&
                 !impl::HasApplyMemberWithWorkspace<LinearOp, XVector, YVector, Workspace> &&
                 !impl::HasApplyMember<LinearOp, XVector, YVector>)
  static void apply(const LinearOp&, const XVector&, YVector&, Workspace&) {
    MUNDY_THROW_REQUIRE(false, std::logic_error,
                        "KokkosBackend::apply: op must be a matrix view or provide void apply(x,y[,workspace]).");
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
    ::mundy::axpby(KokkosBackend{}, alpha, tmp, beta, y);
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
  static void apply(Scalar alpha, const LinearOp& op, const XVector& x, Scalar beta, YVector& y, Workspace&) {
    op.apply(alpha, x, beta, y);
  }

  // Path 2: no fused member -- realize it from the workspace-threaded plain apply plus axpby.
  template <class Scalar, class LinearOp, class XVector, class YVector, class Workspace>
  MUNDY_REQUIRES(!is_native_matrix<LinearOp> && !HasScaledApplyMember<LinearOp, Scalar, XVector, YVector>)
  static void apply(Scalar alpha, const LinearOp& op, const XVector& x, Scalar beta, YVector& y, Workspace& workspace) {
    auto tmp = make_range_vector(op);
    apply(op, x, tmp, workspace);
    ::mundy::axpby(KokkosBackend{}, alpha, tmp, beta, y);
  }

 private:
  template <size_t K, class Arg>
  static auto assign_member(const Arg& arg) {
    if constexpr (K % 2 == 0) {
      return Arg(arg);
    } else {
      return impl::as_source<KokkosBackend>(arg);
    }
  }

  template <class TargetsAndSources, size_t... K>
  static void assign_impl(const char* label, const TargetsAndSources& targets_and_sources, std::index_sequence<K...>) {
    const size_t n = size(std::get<0>(targets_and_sources));
    MUNDY_THROW_ASSERT((impl::has_size<KokkosBackend>(std::get<K>(targets_and_sources), n) && ...),
                       std::invalid_argument,
                       "KokkosBackend::assign: every target and source must have the same size.");
    using members_t = ::mundy::tuple<decltype(assign_member<K>(std::get<K>(targets_and_sources)))...>;
    const members_t members(assign_member<K>(std::get<K>(targets_and_sources))...);
    Kokkos::parallel_for(label, Kokkos::RangePolicy<exec_space>(0, n),
                         impl::AssignFunctor<KokkosBackend, members_t>{members});
  }

  template <class ReducersAndSources, size_t... K>
  static void reduce_impl(const char* label, const ReducersAndSources& reducers_and_sources,
                          std::index_sequence<K...>) {
    using reducers_t = ::mundy::tuple<std::remove_cvref_t<std::tuple_element_t<2 * K, ReducersAndSources>>...>;
    using sources_t =
        ::mundy::tuple<decltype(impl::as_source<KokkosBackend>(std::get<2 * K + 1>(reducers_and_sources)))...>;
    reducers_t reducers(std::get<2 * K>(reducers_and_sources)...);
    const sources_t sources(impl::as_source<KokkosBackend>(std::get<2 * K + 1>(reducers_and_sources))...);
    const size_t n = ::mundy::get<0>(sources).size();
    MUNDY_THROW_ASSERT((impl::has_size<KokkosBackend>(::mundy::get<K>(sources), n) && ...), std::invalid_argument,
                       "KokkosBackend::reduce: every source must have the same size.");
    Kokkos::parallel_reduce(label, Kokkos::RangePolicy<exec_space>(0, n),
                            impl::ReduceFunctor<reducers_t, sources_t>{reducers, sources},
                            ::mundy::get<K>(reducers)...);
  }

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

#endif  // HAVE_MUNDYMATH_KOKKOSKERNELS
};  // KokkosBackend

/// \brief Backend for Mundy math within a kernel
struct MundyMathBackend {
  // Vector and operator sizes are compile-time constants.
  static constexpr bool has_static_sizes = true;

  /// \brief Whether Op is one of this backend's native matrices: a mundy::Matrix.
  template <class Op>
  static constexpr bool is_native_matrix = is_matrix_v<Op>;

  /// \brief Whether A holds contiguous scalars a solver's driver reads and writes in its thread: a Kokkos::Array.
  template <class A>
  static constexpr bool is_scalar_array = impl::IsKokkosArray<std::remove_cvref_t<A>>::value;

  /// \brief The number of vectors in a multivector: a Kokkos::Array of K > 0 mundy::Vectors.
  template <class MultiVector>
  KOKKOS_INLINE_FUNCTION static constexpr size_t num_columns() {
    using mv_t = std::remove_cvref_t<MultiVector>;
    static_assert(impl::IsKokkosArray<mv_t>::value && is_vector_v<typename mv_t::value_type>,
                  "MundyMathBackend: a multivector is a Kokkos::Array of mundy::Vectors.");
    static_assert(mv_t::size() > 0, "MundyMathBackend: a multivector has at least one vector.");
    return mv_t::size();
  }

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

  /// \brief Vector j of the multivector mv.
  template <class MultiVector>
  KOKKOS_INLINE_FUNCTION static decltype(auto) column(MultiVector&& mv, size_t j) {
    return mv[j];
  }

  /// \brief sum_{j < count} c_j column_j of the multivector mv, which must outlive it; columns past count are not read.
  template <class MultiVector, class Scalar, size_t K>
  KOKKOS_INLINE_FUNCTION static auto combination(const MultiVector& mv, size_t count,
                                                 const Kokkos::Array<Scalar, K>& c) {
    static_assert(K == num_columns<MultiVector>(), "MundyMathBackend::combination: one coefficient per column.");
    return impl::ColumnCombinationExpr<const MultiVector*, Scalar, K>(&mv, count, c);
  }

  template <class DestVector, class SrcVector>
  KOKKOS_INLINE_FUNCTION static void deep_copy(DestVector& dest, const SrcVector& src) {
    dest = src;
  }

  /// \brief The entries of x as an expression; an expression is its own entries.
  template <class Vector>
  MUNDY_REQUIRES(BackendVectorExpr<Vector> || impl::HasIndexedEntries<Vector>)
  KOKKOS_INLINE_FUNCTION static auto entries(Vector&& x) {
    if constexpr (BackendVectorExpr<Vector>) {
      return std::remove_cvref_t<Vector>(x);
    } else {
      return EntriesExpr<MundyMathBackend, Vector>(std::forward<Vector>(x));
    }
  }

  /// \brief Writes expressions into vectors, entry by entry: target = source for each (target, source) pair.
  ///
  /// \code
  ///   const auto X = MundyMathBackend::entries(x), Y = MundyMathBackend::entries(y);
  ///   MundyMathBackend::assign("axpby", y, a * X + b * Y);      // y = a x + b y
  ///   MundyMathBackend::assign("swap", x, Y, y, X);            // x and y exchange entries
  ///   MundyMathBackend::assign("zero", y, 0.0);                // y = 0
  /// \endcode
  ///
  /// After the label, which this backend does not use, the arguments come in pairs: a target vector, then its source,
  /// which is an expression, a vector, or a scalar written to every entry. At each entry every source is evaluated
  /// before any target is written, so a target may also appear in any source. Every target and source has the same
  /// size.
  template <class... TargetsAndSources>
  KOKKOS_INLINE_FUNCTION static void assign(const char* /*label*/, TargetsAndSources&&... targets_and_sources) {
    static_assert(sizeof...(TargetsAndSources) > 0 && sizeof...(TargetsAndSources) % 2 == 0,
                  "MundyMathBackend::assign takes (target, source) pairs.");
    constexpr size_t n = static_size<type_at_index_t<0, TargetsAndSources...>>();
    MUNDY_THROW_ASSERT((impl::has_size<MundyMathBackend>(targets_and_sources, n) && ...), std::invalid_argument,
                       "MundyMathBackend::assign: every target and source must have the same size.");
    static_assign(std::make_index_sequence<n>{}, targets_and_sources...);
  }

  /// \brief Reduces expressions over their entries: one result for each (reducer, source) pair.
  ///
  /// \code
  ///   const auto X = MundyMathBackend::entries(x), Y = MundyMathBackend::entries(y);
  ///   double x_dot_y, largest;
  ///   MundyMathBackend::reduce("norms", Kokkos::Sum<double>(x_dot_y), X * Y,   // x_dot_y = x^T y
  ///                            Kokkos::Max<double>(largest), abs(X));          // largest = max_i |x_i|
  /// \endcode
  ///
  /// After the label, which this backend does not use, the arguments come in pairs: a Kokkos reducer bound to its
  /// result, such as Kokkos::Sum, Min, or Max, then its source, which is an expression or a vector.
  template <class... ReducersAndSources>
  KOKKOS_INLINE_FUNCTION static void reduce(const char* /*label*/, const ReducersAndSources&... reducers_and_sources) {
    static_assert(sizeof...(ReducersAndSources) > 0 && sizeof...(ReducersAndSources) % 2 == 0,
                  "MundyMathBackend::reduce takes (reducer, source) pairs.");
    impl::static_reduce<MundyMathBackend>(reducers_and_sources...);
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

  // Apply with workspace: y = op(x), threading the operator's workspace to an op that takes one.
  template <typename LinearOp, class XVector, class YVector, typename Workspace>
  KOKKOS_INLINE_FUNCTION static void apply(const LinearOp& op, const XVector& x, YVector& y, Workspace& workspace) {
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
    ::mundy::axpby(MundyMathBackend{}, alpha, tmp, beta, y);
  }

  // Scaled apply with workspace: y := alpha * op(x) + beta * y, threading the operator's workspace.
  // Path 1: op provides its own fused apply(alpha, x, beta, y); its contract carries no workspace.
  template <class Scalar, typename LinearOp, class XVector, class YVector, typename Workspace>
  MUNDY_REQUIRES(HasScaledApplyMember<LinearOp, Scalar, XVector, YVector>)
  KOKKOS_INLINE_FUNCTION static void apply(Scalar alpha, const LinearOp& op, const XVector& x, Scalar beta, YVector& y,
                                           Workspace&) {
    op.apply(alpha, x, beta, y);
  }

  // Path 2: no fused member -- realize it from the workspace-threaded plain apply plus axpby.
  template <class Scalar, typename LinearOp, class XVector, class YVector, typename Workspace>
  MUNDY_REQUIRES(!HasScaledApplyMember<LinearOp, Scalar, XVector, YVector>)
  KOKKOS_INLINE_FUNCTION static void apply(Scalar alpha, const LinearOp& op, const XVector& x, Scalar beta, YVector& y,
                                           Workspace& workspace) {
    auto tmp = make_range_vector(op);
    apply(op, x, tmp, workspace);
    ::mundy::axpby(MundyMathBackend{}, alpha, tmp, beta, y);
  }

 private:
  template <size_t... Is, class... TargetsAndSources>
  KOKKOS_INLINE_FUNCTION static void static_assign(std::index_sequence<Is...>,
                                                   TargetsAndSources&... targets_and_sources) {
    (impl::assign_entry<MundyMathBackend>(Is, targets_and_sources...), ...);
  }
};  // MundyMathBackend
//@}

}  // namespace mundy

#endif  // MUNDY_MATH_SOLVER_BACKENDS_HPP_
