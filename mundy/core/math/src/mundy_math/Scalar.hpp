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

#ifndef MUNDY_MATH_SCALAR_HPP_
#define MUNDY_MATH_SCALAR_HPP_

// External
#include <Kokkos_Core.hpp>

// C++ core
#include <cmath>
#include <concepts>
#include <iostream>
#include <type_traits>
#include <utility>

// Mundy
#include <mundy_math/Accessor.hpp>   // for mundy::ValidAccessor, impl::access_at
#include <mundy_math/Array.hpp>      // for mundy::Array
#include <mundy_math/Matrix.hpp>     // for mundy::AMatrix (interaction operators)
#include <mundy_math/Tolerance.hpp>  // for mundy::get_comparison_tolerance
#include <mundy_math/Vector.hpp>     // for mundy::AVector (interaction operators)
#include <mundy_utils/requires.hpp>
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_ASSERT

namespace mundy {

/// \brief Class for an owning or viewing arithmetic scalar
///
/// AScalar is the scalar analogue of AVector and AMatrix. It holds a single arithmetic value through a
/// templated Accessor, which may be owning (e.g., Array<T,1>) or non-owning (e.g., a pointer or
/// Kokkos::View slice). This separation of concerns — storage policy vs. value semantics — mirrors the
/// rest of the Mundy math library and makes AScalar Kokkos-compatible.
///
/// The primary interface is value(), operator T() (implicit conversion to the underlying type), and the
/// full set of arithmetic operators. AScalar participates naturally in expressions with AVector and
/// AMatrix through non-member operators defined at the bottom of this file.
///
/// \code{.cpp}
///   Scalar<double> s{3.14};                         // owning, default accessor
///   double raw[1] = {2.0};
///   AScalar<double, double*> view(raw);             // non-owning view of raw memory
///
///   Vector3<double> v{1.0, 2.0, 3.0};
///   auto scaled = v * s;                            // AVector<double, 3>
///   auto doubled = s * s;                           // Scalar<double>{9.87...}
///   double x = s;                                   // implicit conversion
/// \endcode
///
/// \note Accessors may be owning or non-owning; they should be lightweight so they can be copied cheaply.
///       The lifetime of the underlying data must exceed the lifetime of any AScalar that views it.
template <typename T, ValidAccessor<T> Accessor = Array<T, 1>>
MUNDY_REQUIRES(std::is_arithmetic_v<T>)
class AScalar;

//! \name AScalar type traits
//@{

/// \brief (Implementation) Type trait to determine if a type is an AScalar
template <typename TypeToCheck>
struct is_scalar_impl : std::false_type {};
//
template <typename T, typename Accessor>
struct is_scalar_impl<AScalar<T, Accessor>> : std::true_type {};

/// \brief Type trait to determine if a type is an AScalar
template <typename TypeToCheck>
struct is_scalar : public is_scalar_impl<std::decay_t<TypeToCheck>> {};
//
template <typename TypeToCheck>
constexpr bool is_scalar_v = is_scalar<TypeToCheck>::value;

/// \brief Concept satisfied by any type that behaves as a mathematical scalar.
///
/// Accepts fundamental arithmetic types (float, double, int, ...), std::complex,
/// Sacado FAD types, autodiff duals, Ceres Jets — any type providing scalar arithmetic
/// operators and constructibility from a numeric literal.
///
/// Use this instead of \c std::is_arithmetic_v<T> wherever the intent is
/// "I need a type I can do math with", not "I specifically need a primitive type".
/// The broader acceptance enables automatic differentiation without code changes.
template <typename T>
concept ValidScalarType = requires(T a, T b) {
    { a + b } -> std::convertible_to<T>;
    { a - b } -> std::convertible_to<T>;
    { a * b } -> std::convertible_to<T>;
    { a / b } -> std::convertible_to<T>;
    { -a }    -> std::convertible_to<T>;
    T(1.0);  // constructible from a double literal
};

static_assert(ValidScalarType<float>);
static_assert(ValidScalarType<double>);
static_assert(ValidScalarType<int>);
//@}

// =============================================================================
// AScalar class definition
// =============================================================================

template <typename T, ValidAccessor<T> Accessor>
MUNDY_REQUIRES(std::is_arithmetic_v<T>)
class AScalar {
 public:
  //! \name Internal data
  //@{

  /// \brief Our data accessor (public, matching AVector / AMatrix convention)
  storage<Accessor> accessor_;
  //@}

  //! \name Type aliases
  //@{

  /// \brief The type of the stored value
  using value_type = T;

  /// \brief Non-const version of value_type
  using non_const_value_type = std::remove_const_t<T>;

  /// \brief Owning deep-copy type
  using deep_copy_t = AScalar<T>;

  /// \brief The type of the accessor
  using accessor_t = Accessor;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor. Element is uninitialized.
  KOKKOS_DEFAULTED_FUNCTION constexpr AScalar() MUNDY_REQUIRES(HasDefaultConstructor<Accessor>) = default;

  /// \brief Construct from a copy of the given accessor
  KOKKOS_INLINE_FUNCTION
  constexpr explicit AScalar(const Accessor& data) MUNDY_REQUIRES(std::is_copy_constructible_v<Accessor>)
      : accessor_(data) {
  }

  /// \brief Construct from a moved accessor
  KOKKOS_INLINE_FUNCTION
  constexpr explicit AScalar(Accessor&& data)
      MUNDY_REQUIRES(std::is_copy_constructible_v<Accessor> || std::is_move_constructible_v<Accessor>)
      : accessor_(std::forward<Accessor>(data)) {
  }

  /// \brief Construct from a single value (requires the Accessor to have a 1-argument constructor)
  KOKKOS_INLINE_FUNCTION
  constexpr explicit AScalar(const T& val) MUNDY_REQUIRES(HasNArgConstructor<Accessor, T, 1>) : accessor_(val) {
  }

  /// \brief Destructor
  KOKKOS_DEFAULTED_FUNCTION constexpr ~AScalar() = default;

  // Same-type copy and move (shallow — copies/moves the accessor, not necessarily the underlying data)

  /// \brief Copy constructor (shallow)
  KOKKOS_DEFAULTED_FUNCTION constexpr AScalar(const AScalar<T, Accessor>&) = default;

  /// \brief Move constructor (shallow)
  KOKKOS_DEFAULTED_FUNCTION constexpr AScalar(AScalar<T, Accessor>&&) = default;

  /// \brief Copy assignment (deep — copies the stored value)
  KOKKOS_INLINE_FUNCTION constexpr AScalar<T, Accessor>& operator=(const AScalar<T, Accessor>& other) {
    impl::access_at(accessor_, 0) = impl::access_at(other.accessor_, 0);
    return *this;
  }

  /// \brief Move assignment (deep — copies the stored value)
  KOKKOS_INLINE_FUNCTION constexpr AScalar<T, Accessor>& operator=(AScalar<T, Accessor>&& other) {
    impl::access_at(accessor_, 0) = impl::access_at(other.accessor_, 0);
    return *this;
  }

  // Cross-accessor copy / move constructors and assignments

  /// \brief Deep copy constructor from a different AScalar accessor or ownership
  template <typename OtherScalarType>
  KOKKOS_INLINE_FUNCTION constexpr AScalar(const OtherScalarType& other)
      MUNDY_REQUIRES((!std::is_same_v<OtherScalarType, AScalar<T, Accessor>>) &&
                     (std::is_convertible_v<typename OtherScalarType::value_type, T>) &&
                     HasDefaultConstructor<Accessor>)
      : accessor_() {
    impl::access_at(accessor_, 0) = static_cast<T>(other.value());
  }

  /// \brief Deep move constructor from a different AScalar accessor or ownership
  template <typename OtherScalarType>
  KOKKOS_INLINE_FUNCTION constexpr AScalar(OtherScalarType&& other)
      MUNDY_REQUIRES((!std::is_same_v<std::decay_t<OtherScalarType>, AScalar<T, Accessor>>) &&
                     (std::is_convertible_v<typename std::decay_t<OtherScalarType>::value_type, T>) &&
                     HasDefaultConstructor<Accessor>)
      : accessor_() {
    impl::access_at(accessor_, 0) = static_cast<T>(other.value());
  }

  /// \brief Deep copy assignment from a different AScalar accessor or ownership
  template <typename OtherScalarType>
  KOKKOS_INLINE_FUNCTION constexpr AScalar<T, Accessor>& operator=(const OtherScalarType& other)
      MUNDY_REQUIRES((!std::is_same_v<OtherScalarType, AScalar<T, Accessor>>) &&
                     (std::is_convertible_v<typename OtherScalarType::value_type, T>) &&
                     HasNonConstAccessOperator<Accessor, T>) {
    impl::access_at(accessor_, 0) = static_cast<T>(other.value());
    return *this;
  }

  /// \brief Deep move assignment from a different AScalar accessor or ownership
  template <typename OtherScalarType>
  KOKKOS_INLINE_FUNCTION constexpr AScalar<T, Accessor>& operator=(OtherScalarType&& other)
      MUNDY_REQUIRES((!std::is_same_v<std::decay_t<OtherScalarType>, AScalar<T, Accessor>>) &&
                     (std::is_convertible_v<typename std::decay_t<OtherScalarType>::value_type, T>) &&
                     HasNonConstAccessOperator<Accessor, T>) {
    impl::access_at(accessor_, 0) = static_cast<T>(other.value());
    return *this;
  }

  /// \brief Assignment from a raw arithmetic value
  KOKKOS_INLINE_FUNCTION constexpr AScalar<T, Accessor>& operator=(const T val)
      MUNDY_REQUIRES(HasNonConstAccessOperator<Accessor, T>) {
    impl::access_at(accessor_, 0) = val;
    return *this;
  }

  /// \brief Implicit conversion to the underlying arithmetic type
  ///
  /// This allows an AScalar to be passed anywhere a T is expected, and lets existing
  /// AVector / AMatrix member operators (which require std::is_arithmetic_v<U>) accept
  /// an AScalar via implicit conversion rather than requiring separate overloads.
  KOKKOS_INLINE_FUNCTION constexpr operator T() const {
    return impl::access_at(accessor_, 0);
  }
  //@}

  //! \name Value accessors
  //@{

  /// \brief Primary named accessor — returns a reference to the stored value
  KOKKOS_INLINE_FUNCTION constexpr T& value() {
    return impl::access_at(accessor_, 0);
  }
  KOKKOS_INLINE_FUNCTION constexpr const T& value() const {
    return impl::access_at(accessor_, 0);
  }

  /// \brief Subscript accessor for compatibility with generic index-based code (index must be 0)
  KOKKOS_INLINE_FUNCTION constexpr T& operator[](size_t index) {
    MUNDY_THROW_ASSERT(index == 0, std::out_of_range, "AScalar only has one element; index must be 0.");
    return impl::access_at(accessor_, 0);
  }
  KOKKOS_INLINE_FUNCTION constexpr const T& operator[](size_t index) const {
    MUNDY_THROW_ASSERT(index == 0, std::out_of_range, "AScalar only has one element; index must be 0.");
    return impl::access_at(accessor_, 0);
  }

  /// \brief Call-operator accessor (no argument — returns the value directly)
  KOKKOS_INLINE_FUNCTION constexpr T& operator()() {
    return impl::access_at(accessor_, 0);
  }
  KOKKOS_INLINE_FUNCTION constexpr const T& operator()() const {
    return impl::access_at(accessor_, 0);
  }

  /// \brief Call-operator accessor with an index argument (index must be 0, for generic compatibility)
  KOKKOS_INLINE_FUNCTION constexpr T& operator()(size_t index) {
    MUNDY_THROW_ASSERT(index == 0, std::out_of_range, "AScalar only has one element; index must be 0.");
    return impl::access_at(accessor_, 0);
  }
  KOKKOS_INLINE_FUNCTION constexpr const T& operator()(size_t index) const {
    MUNDY_THROW_ASSERT(index == 0, std::out_of_range, "AScalar only has one element; index must be 0.");
    return impl::access_at(accessor_, 0);
  }

  /// \brief Access the underlying accessor data
  KOKKOS_INLINE_FUNCTION constexpr decltype(auto) data() {
    return accessor_.get();
  }
  KOKKOS_INLINE_FUNCTION constexpr decltype(auto) data() const {
    return accessor_.get();
  }
  //@}

  //! \name Setters and modifiers
  //@{

  /// \brief Set the stored value
  KOKKOS_INLINE_FUNCTION constexpr void set(const T& val) MUNDY_REQUIRES(HasNonConstAccessOperator<Accessor, T>) {
    impl::access_at(accessor_, 0) = val;
  }

  /// \brief Fill the scalar with a value (synonym for set; mirrors AVector::fill)
  KOKKOS_INLINE_FUNCTION constexpr void fill(const T& val) MUNDY_REQUIRES(HasNonConstAccessOperator<Accessor, T>) {
    impl::access_at(accessor_, 0) = val;
  }
  //@}

  //! \name Deep copy and cast
  //@{

  /// \brief Return an owning deep copy of this scalar
  KOKKOS_INLINE_FUNCTION constexpr deep_copy_t copy() const {
    return AScalar<T>{static_cast<T>(impl::access_at(accessor_, 0))};
  }

  /// \brief Cast the value to a different arithmetic type and return an owning AScalar
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr auto cast() const {
    return AScalar<U>{static_cast<U>(impl::access_at(accessor_, 0))};
  }
  //@}

  //! \name Unary operators
  //@{

  /// \brief Unary plus
  KOKKOS_INLINE_FUNCTION constexpr AScalar<T> operator+() const {
    return AScalar<T>{impl::access_at(accessor_, 0)};
  }

  /// \brief Unary minus
  KOKKOS_INLINE_FUNCTION constexpr AScalar<T> operator-() const {
    return AScalar<T>{-impl::access_at(accessor_, 0)};
  }
  //@}

  //! \name Addition and subtraction (AScalar op AScalar)
  //@{

  template <typename U, ValidAccessor<U> OtherAccessor>
  KOKKOS_INLINE_FUNCTION constexpr auto operator+(const AScalar<U, OtherAccessor>& other) const {
    using Common = std::common_type_t<T, U>;
    return AScalar<Common>{static_cast<Common>(impl::access_at(accessor_, 0)) +
                           static_cast<Common>(other.value())};
  }

  template <typename U, ValidAccessor<U> OtherAccessor>
  KOKKOS_INLINE_FUNCTION constexpr AScalar<T, Accessor>& operator+=(const AScalar<U, OtherAccessor>& other)
      MUNDY_REQUIRES(HasNonConstAccessOperator<Accessor, T>) {
    impl::access_at(accessor_, 0) += static_cast<T>(other.value());
    return *this;
  }

  template <typename U, ValidAccessor<U> OtherAccessor>
  KOKKOS_INLINE_FUNCTION constexpr auto operator-(const AScalar<U, OtherAccessor>& other) const {
    using Common = std::common_type_t<T, U>;
    return AScalar<Common>{static_cast<Common>(impl::access_at(accessor_, 0)) -
                           static_cast<Common>(other.value())};
  }

  template <typename U, ValidAccessor<U> OtherAccessor>
  KOKKOS_INLINE_FUNCTION constexpr AScalar<T, Accessor>& operator-=(const AScalar<U, OtherAccessor>& other)
      MUNDY_REQUIRES(HasNonConstAccessOperator<Accessor, T>) {
    impl::access_at(accessor_, 0) -= static_cast<T>(other.value());
    return *this;
  }
  //@}

  //! \name Multiplication and division (AScalar op AScalar)
  //@{

  template <typename U, ValidAccessor<U> OtherAccessor>
  KOKKOS_INLINE_FUNCTION constexpr auto operator*(const AScalar<U, OtherAccessor>& other) const {
    using Common = std::common_type_t<T, U>;
    return AScalar<Common>{static_cast<Common>(impl::access_at(accessor_, 0)) *
                           static_cast<Common>(other.value())};
  }

  template <typename U, ValidAccessor<U> OtherAccessor>
  KOKKOS_INLINE_FUNCTION constexpr AScalar<T, Accessor>& operator*=(const AScalar<U, OtherAccessor>& other)
      MUNDY_REQUIRES(HasNonConstAccessOperator<Accessor, T>) {
    impl::access_at(accessor_, 0) *= static_cast<T>(other.value());
    return *this;
  }

  /// \brief Division — promotes integral/integral pairs to double, matching AVector::operator/ semantics.
  template <typename U, ValidAccessor<U> OtherAccessor>
  KOKKOS_INLINE_FUNCTION constexpr auto operator/(const AScalar<U, OtherAccessor>& other) const {
    using Common = std::common_type_t<T, U>;
    using Promoted = std::conditional_t<std::is_integral_v<T> && std::is_integral_v<U>, double, Common>;
    return AScalar<Promoted>{static_cast<Promoted>(impl::access_at(accessor_, 0)) /
                             static_cast<Promoted>(other.value())};
  }

  /// \brief Self-division — does NOT type-promote (integer division is possible).
  template <typename U, ValidAccessor<U> OtherAccessor>
  KOKKOS_INLINE_FUNCTION constexpr AScalar<T, Accessor>& operator/=(const AScalar<U, OtherAccessor>& other)
      MUNDY_REQUIRES(HasNonConstAccessOperator<Accessor, T>) {
    impl::access_at(accessor_, 0) /= static_cast<T>(other.value());
    return *this;
  }
  //@}

  //! \name Addition and subtraction (AScalar op arithmetic)
  //@{

  template <typename U>
  MUNDY_REQUIRES(std::is_arithmetic_v<U>)
  KOKKOS_INLINE_FUNCTION constexpr auto operator+(const U& scalar) const {
    using Common = std::common_type_t<T, U>;
    return AScalar<Common>{static_cast<Common>(impl::access_at(accessor_, 0)) + static_cast<Common>(scalar)};
  }

  template <typename U>
  MUNDY_REQUIRES(std::is_arithmetic_v<U>)
  KOKKOS_INLINE_FUNCTION constexpr AScalar<T, Accessor>& operator+=(const U& scalar)
      MUNDY_REQUIRES(HasNonConstAccessOperator<Accessor, T>) {
    impl::access_at(accessor_, 0) += static_cast<T>(scalar);
    return *this;
  }

  template <typename U>
  MUNDY_REQUIRES(std::is_arithmetic_v<U>)
  KOKKOS_INLINE_FUNCTION constexpr auto operator-(const U& scalar) const {
    using Common = std::common_type_t<T, U>;
    return AScalar<Common>{static_cast<Common>(impl::access_at(accessor_, 0)) - static_cast<Common>(scalar)};
  }

  template <typename U>
  MUNDY_REQUIRES(std::is_arithmetic_v<U>)
  KOKKOS_INLINE_FUNCTION constexpr AScalar<T, Accessor>& operator-=(const U& scalar)
      MUNDY_REQUIRES(HasNonConstAccessOperator<Accessor, T>) {
    impl::access_at(accessor_, 0) -= static_cast<T>(scalar);
    return *this;
  }
  //@}

  //! \name Multiplication and division (AScalar op arithmetic)
  //@{

  template <typename U>
  MUNDY_REQUIRES(std::is_arithmetic_v<U>)
  KOKKOS_INLINE_FUNCTION constexpr auto operator*(const U& scalar) const {
    using Common = std::common_type_t<T, U>;
    return AScalar<Common>{static_cast<Common>(impl::access_at(accessor_, 0)) * static_cast<Common>(scalar)};
  }

  template <typename U>
  MUNDY_REQUIRES(std::is_arithmetic_v<U>)
  KOKKOS_INLINE_FUNCTION constexpr AScalar<T, Accessor>& operator*=(const U& scalar)
      MUNDY_REQUIRES(HasNonConstAccessOperator<Accessor, T>) {
    impl::access_at(accessor_, 0) *= static_cast<T>(scalar);
    return *this;
  }

  /// \brief Division by arithmetic scalar — promotes integral/integral pairs to double.
  template <typename U>
  MUNDY_REQUIRES(std::is_arithmetic_v<U>)
  KOKKOS_INLINE_FUNCTION constexpr auto operator/(const U& scalar) const {
    using Common = std::common_type_t<T, U>;
    using Promoted = std::conditional_t<std::is_integral_v<T> && std::is_integral_v<U>, double, Common>;
    return AScalar<Promoted>{static_cast<Promoted>(impl::access_at(accessor_, 0)) / static_cast<Promoted>(scalar)};
  }

  /// \brief Self-division by arithmetic scalar — does NOT type-promote.
  template <typename U>
  MUNDY_REQUIRES(std::is_arithmetic_v<U>)
  KOKKOS_INLINE_FUNCTION constexpr AScalar<T, Accessor>& operator/=(const U& scalar)
      MUNDY_REQUIRES(HasNonConstAccessOperator<Accessor, T>) {
    impl::access_at(accessor_, 0) /= static_cast<T>(scalar);
    return *this;
  }
  //@}

  //! \name Static factory methods
  //@{

  KOKKOS_INLINE_FUNCTION static constexpr AScalar<T> zero() {
    return AScalar<T>{T(0)};
  }

  KOKKOS_INLINE_FUNCTION static constexpr AScalar<T> one() {
    return AScalar<T>{T(1)};
  }
  //@}

  //! \name Friends
  //@{

  template <typename U, ValidAccessor<U> OtherAccessor>
  MUNDY_REQUIRES(std::is_arithmetic_v<U>)
  friend class AScalar;

  template <typename U, ValidAccessor<U> OtherAccessor>
  friend std::ostream& operator<<(std::ostream& os, const AScalar<U, OtherAccessor>& s);
  //@}
};  // class AScalar

// =============================================================================
// Compile-time sanity checks
// =============================================================================

static_assert(is_scalar_v<AScalar<int, Array<int, 1>>>, "AScalar<int, Array<int,1>> should satisfy is_scalar.");
static_assert(!is_scalar_v<int>, "int should not satisfy is_scalar.");
static_assert(ValidScalarType<AScalar<double>>, "AScalar<double> must satisfy ValidScalarType.");
static_assert(ValidScalarType<AScalar<float>>,  "AScalar<float> must satisfy ValidScalarType.");

// =============================================================================
// Type alias: Scalar<T> — the owning, default-accessor specialisation
// =============================================================================

/// \brief Owning scalar with the default Array<T,1> accessor
template <typename T>
using Scalar = AScalar<T, Array<T, 1>>;

static_assert(is_scalar_v<Scalar<int>>, "Scalar<int> should satisfy is_scalar.");
static_assert(is_scalar_v<Scalar<double>>, "Scalar<double> should satisfy is_scalar.");

// =============================================================================
// Non-member functions
// =============================================================================

//! \name Output stream
//@{

template <typename T, ValidAccessor<T> Accessor>
std::ostream& operator<<(std::ostream& os, const AScalar<T, Accessor>& s) {
  os << s.value();
  return os;
}
//@}

//! \name Comparison
//@{

template <typename U, typename T, ValidAccessor<U> Accessor1, ValidAccessor<T> Accessor2>
KOKKOS_INLINE_FUNCTION constexpr bool is_close(
    const AScalar<U, Accessor1>& a, const AScalar<T, Accessor2>& b,
    const decltype(get_comparison_tolerance<T, U>())& tol = get_comparison_tolerance<T, U>()) {
  using ComparisonType = std::remove_reference_t<decltype(tol)>;
  return Kokkos::abs(static_cast<ComparisonType>(a.value()) - static_cast<ComparisonType>(b.value())) <= tol;
}

template <typename U, typename T, ValidAccessor<U> Accessor1, ValidAccessor<T> Accessor2>
KOKKOS_INLINE_FUNCTION constexpr bool is_approx_close(
    const AScalar<U, Accessor1>& a, const AScalar<T, Accessor2>& b,
    const decltype(get_relaxed_comparison_tolerance<T, U>())& tol = get_relaxed_comparison_tolerance<T, U>()) {
  return is_close(a, b, tol);
}
//@}

//! \name Non-member arithmetic: arithmetic op AScalar
//@{

template <typename U, typename T, ValidAccessor<T> Accessor>
MUNDY_REQUIRES(std::is_arithmetic_v<U>)
KOKKOS_INLINE_FUNCTION constexpr auto operator+(const U& scalar, const AScalar<T, Accessor>& s) {
  return s + scalar;
}

template <typename U, typename T, ValidAccessor<T> Accessor>
MUNDY_REQUIRES(std::is_arithmetic_v<U>)
KOKKOS_INLINE_FUNCTION constexpr auto operator-(const U& scalar, const AScalar<T, Accessor>& s) {
  using Common = std::common_type_t<T, U>;
  return AScalar<Common>{static_cast<Common>(scalar) - static_cast<Common>(s.value())};
}

template <typename U, typename T, ValidAccessor<T> Accessor>
MUNDY_REQUIRES(std::is_arithmetic_v<U>)
KOKKOS_INLINE_FUNCTION constexpr auto operator*(const U& scalar, const AScalar<T, Accessor>& s) {
  return s * scalar;
}

/// \brief arithmetic / AScalar — promotes integral/integral to double
template <typename U, typename T, ValidAccessor<T> Accessor>
MUNDY_REQUIRES(std::is_arithmetic_v<U>)
KOKKOS_INLINE_FUNCTION constexpr auto operator/(const U& scalar, const AScalar<T, Accessor>& s) {
  using Common = std::common_type_t<T, U>;
  using Promoted = std::conditional_t<std::is_integral_v<T> && std::is_integral_v<U>, double, Common>;
  return AScalar<Promoted>{static_cast<Promoted>(scalar) / static_cast<Promoted>(s.value())};
}
//@}

//! \name Non-member arithmetic: AScalar op AVector / AVector op AScalar
//
// AVector's member operators require std::is_arithmetic_v<U>, so they do not fire when U is AScalar.
// These non-member operators bridge the gap by extracting the stored value and delegating.
//@{

/// \brief AVector * AScalar
template <size_t N, typename T, typename U, ValidAccessor<T> Accessor1, ValidAccessor<U> Accessor2>
KOKKOS_INLINE_FUNCTION constexpr auto operator*(const AVector<T, N, Accessor1>& vec,
                                                const AScalar<U, Accessor2>& s)
    -> AVector<std::common_type_t<T, U>, N> {
  return vec * s.value();
}

/// \brief AScalar * AVector  (commutative)
template <size_t N, typename U, typename T, ValidAccessor<U> Accessor1, ValidAccessor<T> Accessor2>
KOKKOS_INLINE_FUNCTION constexpr auto operator*(const AScalar<U, Accessor1>& s,
                                                const AVector<T, N, Accessor2>& vec)
    -> AVector<std::common_type_t<T, U>, N> {
  return vec * s.value();
}

/// \brief AVector / AScalar — delegates to AVector::operator/, preserving its integer-promotion behaviour
template <size_t N, typename T, typename U, ValidAccessor<T> Accessor1, ValidAccessor<U> Accessor2>
KOKKOS_INLINE_FUNCTION constexpr auto operator/(const AVector<T, N, Accessor1>& vec,
                                                const AScalar<U, Accessor2>& s) {
  return vec / s.value();
}
//@}

//! \name Non-member arithmetic: AScalar op AMatrix / AMatrix op AScalar
//@{

/// \brief AMatrix * AScalar
template <size_t N, size_t M, typename T, typename U, ValidAccessor<T> Accessor1, ValidAccessor<U> Accessor2>
KOKKOS_INLINE_FUNCTION constexpr auto operator*(const AMatrix<T, N, M, Accessor1>& mat,
                                                const AScalar<U, Accessor2>& s)
    -> AMatrix<std::common_type_t<T, U>, N, M> {
  return mat * s.value();
}

/// \brief AScalar * AMatrix  (commutative)
template <size_t N, size_t M, typename U, typename T, ValidAccessor<U> Accessor1, ValidAccessor<T> Accessor2>
KOKKOS_INLINE_FUNCTION constexpr auto operator*(const AScalar<U, Accessor1>& s,
                                                const AMatrix<T, N, M, Accessor2>& mat)
    -> AMatrix<std::common_type_t<T, U>, N, M> {
  return mat * s.value();
}

/// \brief AMatrix / AScalar — delegates to AMatrix::operator/, preserving its integer-promotion behaviour
template <size_t N, size_t M, typename T, typename U, ValidAccessor<T> Accessor1, ValidAccessor<U> Accessor2>
KOKKOS_INLINE_FUNCTION constexpr auto operator/(const AMatrix<T, N, M, Accessor1>& mat,
                                                const AScalar<U, Accessor2>& s) {
  return mat / s.value();
}
//@}

//! \name Utility free functions
//@{

/// \brief Absolute value of a scalar
template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto abs(const AScalar<T, Accessor>& s) {
  return AScalar<T>{Kokkos::abs(s.value())};
}

/// \brief Deep copy (mirrors the AVector free function)
template <ValidScalarType ScalarType>
KOKKOS_INLINE_FUNCTION constexpr auto copy(const ScalarType& s) {
  return s.copy();
}

/// \brief Cast a scalar to a different arithmetic type
template <typename U, ValidScalarType ScalarType>
KOKKOS_INLINE_FUNCTION constexpr auto cast(const ScalarType& s) {
  return s.template cast<U>();
}
//@}

//! \name Atomic memory operations on raw scalars
//
// These operate on raw T* pointers rather than on AScalar objects so that the same mundy::atomic_add etc.
// call works regardless of whether you're dealing with a raw scalar or a vector/matrix element.
//@{

/// \brief Atomic load: s_copy = *s
template <typename T>
KOKKOS_INLINE_FUNCTION T atomic_load(T* const s) {
  return Kokkos::atomic_load(s);
}

/// \brief Atomic store: *s = value
template <typename T, typename U>
KOKKOS_INLINE_FUNCTION void atomic_store(T* const s, const U& value) {
  Kokkos::atomic_store(s, static_cast<T>(value));
}

/// \brief Atomic *s += value
template <typename T, typename U>
KOKKOS_INLINE_FUNCTION void atomic_add(T* const s, const U& value) {
  Kokkos::atomic_add(s, static_cast<T>(value));
}

/// \brief Atomic *s += value — AScalar overload: operates on the underlying scalar, bypassing
/// AScalar's operator+ (which returns a different storage type incompatible with CAS loops).
template <typename T, typename Acc, typename U>
KOKKOS_INLINE_FUNCTION void atomic_add(AScalar<T, Acc>* const s, const U& value) {
  Kokkos::atomic_add(&s->value(), static_cast<T>(value));
}

/// \brief Atomic *s -= value
template <typename T, typename U>
KOKKOS_INLINE_FUNCTION void atomic_sub(T* const s, const U& value) {
  Kokkos::atomic_sub(s, static_cast<T>(value));
}

/// \brief Atomic *s -= value — AScalar overload.
template <typename T, typename Acc, typename U>
KOKKOS_INLINE_FUNCTION void atomic_sub(AScalar<T, Acc>* const s, const U& value) {
  Kokkos::atomic_sub(&s->value(), static_cast<T>(value));
}

/// \brief Atomic *s *= value
template <typename T, typename U>
KOKKOS_INLINE_FUNCTION void atomic_mul(T* const s, const U& value) {
  Kokkos::atomic_mul(s, static_cast<T>(value));
}

/// \brief Atomic *s *= value — AScalar overload.
template <typename T, typename Acc, typename U>
KOKKOS_INLINE_FUNCTION void atomic_mul(AScalar<T, Acc>* const s, const U& value) {
  Kokkos::atomic_mul(&s->value(), static_cast<T>(value));
}

/// \brief Atomic *s /= value
template <typename T, typename U>
KOKKOS_INLINE_FUNCTION void atomic_div(T* const s, const U& value) {
  Kokkos::atomic_div(s, static_cast<T>(value));
}

/// \brief Atomic *s /= value — AScalar overload.
template <typename T, typename Acc, typename U>
KOKKOS_INLINE_FUNCTION void atomic_div(AScalar<T, Acc>* const s, const U& value) {
  Kokkos::atomic_div(&s->value(), static_cast<T>(value));
}

/// \brief Atomic *s += value; returns old *s
template <typename T, typename U>
KOKKOS_INLINE_FUNCTION T atomic_fetch_add(T* const s, const U& value) {
  return Kokkos::atomic_fetch_add(s, static_cast<T>(value));
}

/// \brief Atomic *s -= value; returns old *s
template <typename T, typename U>
KOKKOS_INLINE_FUNCTION T atomic_fetch_sub(T* const s, const U& value) {
  return Kokkos::atomic_fetch_sub(s, static_cast<T>(value));
}

/// \brief Atomic *s *= value; returns old *s
template <typename T, typename U>
KOKKOS_INLINE_FUNCTION T atomic_fetch_mul(T* const s, const U& value) {
  return Kokkos::atomic_fetch_mul(s, static_cast<T>(value));
}

/// \brief Atomic *s /= value; returns old *s
template <typename T, typename U>
KOKKOS_INLINE_FUNCTION T atomic_fetch_div(T* const s, const U& value) {
  return Kokkos::atomic_fetch_div(s, static_cast<T>(value));
}

/// \brief Atomic *s += value; returns new *s
template <typename T, typename U>
KOKKOS_INLINE_FUNCTION T atomic_add_fetch(T* const s, const U& value) {
  return Kokkos::atomic_add_fetch(s, static_cast<T>(value));
}

/// \brief Atomic *s -= value; returns new *s
template <typename T, typename U>
KOKKOS_INLINE_FUNCTION T atomic_sub_fetch(T* const s, const U& value) {
  return Kokkos::atomic_sub_fetch(s, static_cast<T>(value));
}

/// \brief Atomic *s *= value; returns new *s
template <typename T, typename U>
KOKKOS_INLINE_FUNCTION T atomic_mul_fetch(T* const s, const U& value) {
  return Kokkos::atomic_mul_fetch(s, static_cast<T>(value));
}

/// \brief Atomic *s /= value; returns new *s
template <typename T, typename U>
KOKKOS_INLINE_FUNCTION T atomic_div_fetch(T* const s, const U& value) {
  return Kokkos::atomic_div_fetch(s, static_cast<T>(value));
}
//@}

//! \name View and owning helpers
//@{

/// \brief Create a non-owning AScalar view over an existing accessor.
///
/// \code
///   double x = 3.14;
///   auto s = get_scalar_view<double>(&x);  // views x, does not copy
/// \endcode
template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_scalar_view(Accessor&& data) {
  auto data_storage = store(impl::unwrap_accessor(std::forward<Accessor>(data)));
  return AScalar<T, decltype(data_storage)>(data_storage);
}

/// \brief Create an owning AScalar by moving the accessor.
template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_owning_scalar(Accessor&& data) {
  auto data_storage = store(impl::unwrap_accessor(std::move(data)));
  return AScalar<T, decltype(data_storage)>(data_storage);
}
//@}

}  // namespace mundy

#endif  // MUNDY_MATH_SCALAR_HPP_
