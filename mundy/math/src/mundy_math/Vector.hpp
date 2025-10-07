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

#ifndef MUNDY_MATH_VECTOR_HPP_
#define MUNDY_MATH_VECTOR_HPP_

// External
#include <Kokkos_Core.hpp>

// C++ core
#include <cmath>
#include <concepts>
#include <initializer_list>
#include <iostream>
#include <type_traits>  // for std::decay_t
#include <utility>

// Mundy
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_math/Accessor.hpp>      // for mundy::math::ValidAccessor
#include <mundy_math/Array.hpp>         // for mundy::math::Array
#include <mundy_math/Tolerance.hpp>     // for mundy::math::get_zero_tolerance
#include <mundy_math/impl/VectorImpl.hpp>

namespace mundy {

namespace math {

/// \brief (Implementation) Type trait to determine if a type is a vector
template <typename TypeToCheck>
struct is_vector_impl : std::false_type {};
//
template <typename T, size_t N, typename Accessor, typename OwnershipType>
struct is_vector_impl<AVector<T, N, Accessor, OwnershipType>> : std::true_type {};

/// \brief Type trait to determine if a type is a AVector
template <typename T>
struct is_vector : public is_vector_impl<std::decay_t<T>> {};
//
template <typename TypeToCheck>
constexpr bool is_vector_v = is_vector<TypeToCheck>::value;

/// \brief A temporary concept to check if a type is a valid AVector type
/// TODO(palmerb4): Extend this concept to contain all shared setters and getters for our vectors.
template <typename VectorType>
concept ValidVectorType =
    is_vector_v<std::decay_t<VectorType>> &&
    requires(std::decay_t<VectorType> vector, const std::decay_t<VectorType> const_vector, size_t i) {
      typename std::decay_t<VectorType>::scalar_t;
      { vector[i] } -> std::convertible_to<typename std::decay_t<VectorType>::scalar_t>;
      { vector(i) } -> std::convertible_to<typename std::decay_t<VectorType>::scalar_t>;
      { const_vector[i] } -> std::convertible_to<const typename std::decay_t<VectorType>::scalar_t>;
      { const_vector(i) } -> std::convertible_to<const typename std::decay_t<VectorType>::scalar_t>;
    };  // ValidVectorType

/// \brief Class for an Nx1 vector with arithmetic entries
/// \tparam T The type of the entries.
/// \tparam Accessor The type of the accessor.
///
/// This class is designed to be used with Kokkos. It is a simple Nx1 vector with arithmetic entries. It is templated
/// on the type of the entries and Accessor type. See Accessor.hpp for more details on the Accessor type requirements.
///
/// The goal of AVector is to be a lightweight class that can be used with Kokkos to perform mathematical operations on
/// vectors in R3. It does not own the data, but rather it is templated on an Accessor type that provides access to the
/// underlying data. This allows us to use AVector with Kokkos Ownership::Views, raw pointers, or any other type that
/// meets the ValidAccessor requirements without copying the data. This is especially important for GPU-compatable code.
///
/// AVectors can be constructed by passing an accessor to the constructor. However, if the accessor has a N-argument
/// constructor, then the AVector can also be constructed by passing the elements directly to the constructor.
/// Similarly, if the accessor has an initializer list constructor, then the AVector can be constructed by passing an
/// initializer list to the constructor. This is a convenience feature which makes working with the default accessor
/// (Array<T, N>) easier. For example, the following are all valid ways to construct a AVector:
///
/// \code{.cpp}
///   // Constructs a AVector with the default accessor (Array<int, N>)
///   AVector<int, 3> vec1({1, 2, 3});
///   AVector<int, 3> vec2(1, 2, 3);
///   AVector<int, 3> vec3(Array<int, 3>({1, 2, 3}));
///   AVector<int, 3> vec4;
///   vec4.set(1, 2, 3);
///
///   // Construct a VectorView from a double array
///   double data[3] = {1.0, 2.0, 3.0};
///   VectorView<double, 3, double*> vec5(data);
///
///   // Do math with Ownership::Views and AVectors interchangeably
///   double mundy::math::dot(vec1, vec5);
/// \endcode
///
/// \note Accessors may be owning or non-owning, that is irrelevant to the AVector class; however, these accessors
/// should be lightweight such that they can be copied around without much overhead. Furthermore, the lifetime of the
/// data underlying the accessor should be as long as the AVector that use it.
template <typename T, size_t N, ValidAccessor<T> Accessor>
  requires std::is_arithmetic_v<T>
class AVector<T, N, Accessor, Ownership::Views> {
 public:
  //! \name Internal data
  //@{

  /// \brief A reference or a pointer to an external data accessor.
  std::conditional_t<std::is_pointer_v<Accessor>, Accessor, Accessor&> accessor_;
  //@}

  //! \name Type aliases
  //@{

  /// \brief The type of the entries
  using scalar_t = T;

  /// \brief The non-const type of the entries
  using non_const_scalar_t = std::remove_const_t<T>;

  /// \brief Our ownership type
  using ownership_t = Ownership::Views;

  /// \brief Deep copy type
  using deep_copy_t = AVector<T, N>;

  /// \brief The size of the vector
  static constexpr size_t size = N;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor since we don't own the data.
  KOKKOS_INLINE_FUNCTION AVector() = delete;

  /// \brief Constructor for reference accessors
  KOKKOS_INLINE_FUNCTION
  explicit constexpr AVector(Accessor& accessor)
    requires(!std::is_pointer_v<Accessor>)
      : accessor_(accessor) {
  }

  /// \brief Constructor for pointer accessors
  KOKKOS_INLINE_FUNCTION
  explicit constexpr AVector(Accessor accessor)
    requires std::is_pointer_v<Accessor>
      : accessor_(accessor) {
  }

  /// \brief Destructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr ~AVector() = default;

  // Default copy/move constructors and assignment operators when interacting with a AVector of the same type

  /// \brief Default copy constructor (shallow copy)
  KOKKOS_DEFAULTED_FUNCTION
  constexpr AVector(const AVector<T, N, Accessor, Ownership::Views>&) = default;

  /// \brief Default move constructor (shallow move)
  KOKKOS_DEFAULTED_FUNCTION
  constexpr AVector(AVector<T, N, Accessor, Ownership::Views>&&) = default;

  /// \brief Default copy assignment operator (shallow copy)
  KOKKOS_DEFAULTED_FUNCTION
  constexpr AVector<T, N, Accessor, Ownership::Views>& operator=(const AVector<T, N, Accessor, Ownership::Views>&) =
      default;

  /// \brief Default move assignment operator (shallow move)
  KOKKOS_DEFAULTED_FUNCTION
  constexpr AVector<T, N, Accessor, Ownership::Views>& operator=(AVector<T, N, Accessor, Ownership::Views>&&) = default;

  // Custom copy/move constructors and assignment operators when interacting with a AVector of a different type
  // We do not allow copy/move construction from a AVector of a different type. This is undefined behavior.

  /// \brief Deep copy assignment operator with different accessor or ownership
  /// \details Copies the data from the other vector to our data. This is only enabled if T is not const.
  template <ValidVectorType OtherVectorType>
  KOKKOS_INLINE_FUNCTION constexpr AVector<T, N, Accessor, Ownership::Views>& operator=(const OtherVectorType& other)
    requires(!std::is_same_v<OtherVectorType, AVector<T, N, Accessor, Ownership::Views>>) &&
            (OtherVectorType::size == N) &&
            (std::is_same_v<typename OtherVectorType::scalar_t, T>) && HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(std::make_index_sequence<N>{}, *this, other);
    return *this;
  }

  /// \brief Deep move assignment operator with different accessor or ownership
  /// \details Moves the data from the other vector to our data. This is only enabled if T is not const.
  template <ValidVectorType OtherVectorType>
  KOKKOS_INLINE_FUNCTION constexpr AVector<T, N, Accessor, Ownership::Views>& operator=(OtherVectorType&& other)
    requires(!std::is_same_v<OtherVectorType, AVector<T, N, Accessor, Ownership::Views>>) &&
            (OtherVectorType::size == N) &&
            (std::is_same_v<typename OtherVectorType::scalar_t, T>) && HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(std::make_index_sequence<N>{}, *this, std::move(other));
    return *this;
  }

  /// \brief Deep copy assignment operator from a single value
  /// \param[in] value The value to set all elements to.
  KOKKOS_INLINE_FUNCTION constexpr AVector<T, N, Accessor, Ownership::Views>& operator=(const T value)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::fill_impl(std::make_index_sequence<N>{}, *this, value);
    return *this;
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Element access operator via a single index
  /// \param[in] index The index of the element.
  KOKKOS_INLINE_FUNCTION
  constexpr T& operator[](size_t index) {
    return accessor_[index];
  }
  KOKKOS_INLINE_FUNCTION
  constexpr const T& operator[](size_t index) const {
    return accessor_[index];
  }

  /// \brief Element access operator via a single index
  /// \param[in] index The index of the element.
  KOKKOS_INLINE_FUNCTION
  constexpr T& operator()(size_t index) {
    return accessor_[index];
  }
  KOKKOS_INLINE_FUNCTION
  constexpr const T& operator()(size_t index) const {
    return accessor_[index];
  }

  /// \brief Get the internal data accessor
  KOKKOS_INLINE_FUNCTION
  constexpr std::conditional_t<std::is_pointer_v<Accessor>, Accessor, Accessor&> data() {
    return accessor_;
  }
  KOKKOS_INLINE_FUNCTION
  constexpr const std::conditional_t<std::is_pointer_v<Accessor>, Accessor, Accessor&> data() const {
    return accessor_;
  }

  /// \brief Get a deep copy of the vector
  KOKKOS_INLINE_FUNCTION
  constexpr deep_copy_t copy() const {
    return *this;
  }

  /// \brief Cast (and copy) the vector to a different type
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr auto cast() const {
    return impl::cast_impl<U>(std::make_index_sequence<N>{}, *this);
  }
  //@}

  //! \name Setters and modifiers
  //@{

  /// \brief Set all elements of the vector
  template <typename... Args>
    requires(sizeof...(Args) == N) && (std::is_convertible_v<Args, T> && ...) && HasNonConstAccessOperator<Accessor, T>
  KOKKOS_INLINE_FUNCTION constexpr void set(Args&&... args) {
    impl::set_from_args_impl(std::make_index_sequence<N>{}, *this, static_cast<T>(std::forward<Args>(args))...);
  }

  /// \brief Set all elements of the vector using an accessor
  /// \param[in] accessor A valid accessor.
  /// \note A AVector is also a valid accessor.
  template <ValidAccessor<T> OtherAccessor>
  KOKKOS_INLINE_FUNCTION constexpr void set(const OtherAccessor& accessor)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::set_from_accessor_impl(std::make_index_sequence<N>{}, *this, accessor);
  }

  /// \brief Set all elements of the vector to a single value
  /// \param[in] value The value to set all elements to.
  KOKKOS_INLINE_FUNCTION
  constexpr void fill(const T& value)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::fill_impl(std::make_index_sequence<N>{}, *this, value);
  }
  //@}

  //! \name Unary operators
  //@{

  /// \brief Unary plus operator
  KOKKOS_INLINE_FUNCTION
  constexpr AVector<T, N> operator+() const {
    return *this;
  }

  /// \brief Unary minus operator
  KOKKOS_INLINE_FUNCTION
  constexpr AVector<T, N> operator-() const {
    return impl::unary_minus_impl(std::make_index_sequence<N>{}, *this);
  }
  //@}

  //! \name Addition and subtraction
  //@{

  /// \brief AVector-vector addition
  /// \param[in] other The other vector.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr auto operator+(const AVector<U, N, OtherAccessor, OtherOwnershipType>& other) const {
    return impl::vector_vector_add_impl(std::make_index_sequence<N>{}, *this, other);
  }

  /// \brief AVector-vector addition
  /// \param[in] other The other vector.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr AVector<T, N, Accessor, Ownership::Views>& operator+=(
      const AVector<U, N, OtherAccessor, OtherOwnershipType>& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_vector_add_impl(std::make_index_sequence<N>{}, *this, other);
    return *this;
  }

  /// \brief AVector-vector subtraction
  /// \param[in] other The other vector.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr auto operator-(const AVector<U, N, OtherAccessor, OtherOwnershipType>& other) const {
    return impl::vector_vector_subtraction_impl(std::make_index_sequence<N>{}, *this, other);
  }

  /// \brief Self-vector subtraction
  /// \param[in] other The other vector.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr AVector<T, N, Accessor, Ownership::Views>& operator-=(
      const AVector<U, N, OtherAccessor, OtherOwnershipType>& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_vector_subtraction_impl(std::make_index_sequence<N>{}, *this, other);
    return *this;
  }

  /// \brief AVector-scalar addition
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr auto operator+(const U& scalar) const {
    return impl::vector_scalar_add_impl(std::make_index_sequence<N>{}, *this, scalar);
  }

  /// \brief Self-scalar addition
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr AVector<T, N, Accessor, Ownership::Views>& operator+=(const U& scalar)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_scalar_add_impl(std::make_index_sequence<N>{}, *this, scalar);
    return *this;
  }

  /// \brief AVector-scalar subtraction
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr auto operator-(const U& scalar) const {
    return impl::vector_scalar_subtraction_impl(std::make_index_sequence<N>{}, *this, scalar);
  }

  /// \brief AVector-scalar subtraction
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr AVector<T, N, Accessor, Ownership::Views>& operator-=(const U& scalar)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_scalar_subtraction_impl(std::make_index_sequence<N>{}, *this, scalar);
    return *this;
  }
  //@}

  //! \name Multiplication and division
  //@{

  /// \brief AVector-scalar multiplication
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr auto operator*(const U& scalar) const {
    return impl::vector_scalar_multiplication_impl(std::make_index_sequence<N>{}, *this, scalar);
  }

  /// \brief AVector-scalar multiplication
  /// \param[in] scalar The scalar.
  template <typename U>
    requires HasNonConstAccessOperator<Accessor, T>
  KOKKOS_INLINE_FUNCTION constexpr AVector<T, N, Accessor, Ownership::Views>& operator*=(const U& scalar) {
    impl::self_scalar_multiplication_impl(std::make_index_sequence<N>{}, *this, scalar);
    return *this;
  }

  /// \brief AVector-scalar division. (Type promotes the result to a double if the scalar is not a floating point.)
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr auto operator/(const U& scalar) const {
    return impl::vector_scalar_division_impl(std::make_index_sequence<N>{}, *this, scalar);
  }

  /// \brief AVector-scalar division (Does not type-promote the result!!).
  /// \note Because there is no type promotion, this will perform integer division if the scalar is an integer.
  /// \param[in] scalar The scalar.
  template <typename U>
    requires HasNonConstAccessOperator<Accessor, T>
  KOKKOS_INLINE_FUNCTION constexpr AVector<T, N, Accessor, Ownership::Views>& operator/=(const U& scalar) {
    impl::self_scalar_division_impl(std::make_index_sequence<N>{}, *this, scalar);
    return *this;
  }
  //@}

  //! \name Friends <3
  //@{

  // Declare the << operator as a friend
  template <typename U, size_t M, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  friend std::ostream& operator<<(std::ostream& os, const AVector<U, M, OtherAccessor, OtherOwnershipType>& vec);

  // We are friends with all AVectors regardless of their Accessor, type, or ownership
  template <typename U, size_t M, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
    requires std::is_arithmetic_v<U>
  friend class AVector;
  //@}
};  // class AVector (non-owning)

template <typename T, size_t N, ValidAccessor<T> Accessor>
  requires std::is_arithmetic_v<T>
class AVector<T, N, Accessor, Ownership::Owns> {
 public:
  //! \name Internal data
  //@{

  /// \brief Our data accessor. Owning
  Accessor accessor_;
  //@}

  //! \name Type aliases
  //@{

  /// \brief The type of the entries
  using scalar_t = T;

  /// \brief The non-const type of the entries
  using non_const_scalar_t = std::remove_const_t<T>;

  /// \brief Our ownership type
  using ownership_t = Ownership::Owns;

  /// \brief Deep copy type
  using deep_copy_t = AVector<T, N>;

  /// \brief The type of the accessor
  using accessor_t = Accessor;

  /// \brief The size of the vector
  static constexpr size_t size = N;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor. Assume elements are uninitialized.
  /// \note This constructor is only enabled if the Accessor has a default constructor.
  KOKKOS_DEFAULTED_FUNCTION constexpr AVector()
    requires HasDefaultConstructor<Accessor>
  = default;

  /// \brief Constructor from a given accessor
  /// \param[in] data The accessor.
  KOKKOS_INLINE_FUNCTION
  constexpr explicit AVector(const Accessor& data)
    requires std::is_copy_constructible_v<Accessor>
      : accessor_(data) {
  }

  /// \brief Constructor from a given accessor
  /// \param[in] data The accessor.
  KOKKOS_INLINE_FUNCTION
  constexpr explicit AVector(Accessor&& data)
    requires(std::is_copy_constructible_v<Accessor> || std::is_move_constructible_v<Accessor>)
      : accessor_(std::forward<Accessor>(data)) {
  }

  /// \brief Constructor to initialize all elements to a single value.
  /// Requires the number of arguments to be N and the type of each to be T.
  /// Only enabled if the Accessor has a N-argument constructor.
  KOKKOS_INLINE_FUNCTION constexpr explicit AVector(const T& value)
    requires HasNArgConstructor<Accessor, T, 1>
      : accessor_(value) {
  }

  /// \brief Constructor to initialize all elements explicitly.
  /// Requires the number of arguments to be N and the type of each to be T.
  /// Only enabled if the Accessor has a N-argument constructor.
  template <typename... Args>
    requires(sizeof...(Args) == N) && (N != 1) &&
            (std::is_convertible_v<Args, T> && ...) && HasNArgConstructor<Accessor, T, N>
  KOKKOS_INLINE_FUNCTION constexpr explicit AVector(Args&&... args)
      : accessor_{static_cast<T>(std::forward<Args>(args))...} {
  }

  /// \brief Constructor to initialize all elements via initializer list
  /// \param[in] list The initializer list.
  KOKKOS_INLINE_FUNCTION constexpr AVector(const std::initializer_list<T>& list)
    requires HasInitializerListConstructor<Accessor, T>
      : accessor_(list) {
  }

  /// \brief Destructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr ~AVector() = default;

  // Default copy/move constructors and assignment operators when interacting with a AVector of the same type

  /// \brief Default copy constructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr AVector(const AVector<T, N, Accessor, Ownership::Owns>&) = default;

  /// \brief Default move constructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr AVector(AVector<T, N, Accessor, Ownership::Owns>&&) = default;

  /// \brief Default copy assignment operator
  KOKKOS_DEFAULTED_FUNCTION
  constexpr AVector<T, N, Accessor, Ownership::Owns>& operator=(const AVector<T, N, Accessor, Ownership::Owns>&) =
      default;

  /// \brief Default move assignment operator
  KOKKOS_DEFAULTED_FUNCTION
  constexpr AVector<T, N, Accessor, Ownership::Owns>& operator=(AVector<T, N, Accessor, Ownership::Owns>&&) = default;

  // Custom copy/move constructors and assignment operators when interacting with a AVector of a different type

  /// \brief Deep copy constructor with different accessor or ownership
  template <ValidVectorType OtherVectorType>
  KOKKOS_INLINE_FUNCTION constexpr AVector(const OtherVectorType& other)
    requires(!std::is_same_v<OtherVectorType, AVector<T, N, Accessor, Ownership::Owns>>) &&
            (OtherVectorType::size == N) &&
            (std::is_same_v<typename OtherVectorType::scalar_t, T>) && HasDefaultConstructor<Accessor>
      : accessor_() {
    impl::deep_copy_impl(std::make_index_sequence<N>{}, *this, other);
  }

  /// \brief Deep move constructor with different accessor or ownership
  template <ValidVectorType OtherVectorType>
  KOKKOS_INLINE_FUNCTION constexpr AVector(OtherVectorType&& other)
    requires(!std::is_same_v<OtherVectorType, AVector<T, N, Accessor, Ownership::Owns>>) &&
            (OtherVectorType::size == N) &&
            (std::is_same_v<typename OtherVectorType::scalar_t, T>) && HasDefaultConstructor<Accessor>
      : accessor_() {
    impl::deep_copy_impl(std::make_index_sequence<N>{}, *this, std::move(other));
  }

  /// \brief Deep copy assignment operator with different accessor or ownership
  /// \details Copies the data from the other vector to our data. This is only enabled if T is not const.
  template <ValidVectorType OtherVectorType>
  KOKKOS_INLINE_FUNCTION constexpr AVector<T, N, Accessor, Ownership::Owns>& operator=(const OtherVectorType& other)
    requires(!std::is_same_v<OtherVectorType, AVector<T, N, Accessor, Ownership::Owns>>) &&
            (OtherVectorType::size == N) &&
            (std::is_same_v<typename OtherVectorType::scalar_t, T>) && HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(std::make_index_sequence<N>{}, *this, other);
    return *this;
  }

  /// \brief Deep move assignment operator with different accessor or ownership
  /// \details Moves the data from the other vector to our data. This is only enabled if T is not const.
  template <ValidVectorType OtherVectorType>
  KOKKOS_INLINE_FUNCTION constexpr AVector<T, N, Accessor, Ownership::Owns>& operator=(OtherVectorType&& other)
    requires(!std::is_same_v<OtherVectorType, AVector<T, N, Accessor, Ownership::Owns>>) &&
            (OtherVectorType::size == N) &&
            (std::is_same_v<typename OtherVectorType::scalar_t, T>) && HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(std::make_index_sequence<N>{}, *this, std::move(other));
    return *this;
  }

  /// \brief Deep copy assignment operator from a single value
  /// \param[in] value The value to set all elements to.
  KOKKOS_INLINE_FUNCTION constexpr AVector<T, N, Accessor, Ownership::Owns>& operator=(const T value)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::fill_impl(std::make_index_sequence<N>{}, *this, value);
    return *this;
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Element access operator via a single index
  /// \param[in] index The index of the element.
  KOKKOS_INLINE_FUNCTION
  constexpr T& operator[](size_t index) {
    return accessor_[index];
  }
  KOKKOS_INLINE_FUNCTION
  constexpr const T& operator[](size_t index) const {
    return accessor_[index];
  }

  /// \brief Element access operator via a single index
  /// \param[in] index The index of the element.
  KOKKOS_INLINE_FUNCTION
  constexpr T& operator()(size_t index) {
    return accessor_[index];
  }
  KOKKOS_INLINE_FUNCTION
  constexpr const T& operator()(size_t index) const {
    return accessor_[index];
  }

  /// \brief Get the internal data accessor
  KOKKOS_INLINE_FUNCTION
  constexpr Accessor& data() {
    return accessor_;
  }
  KOKKOS_INLINE_FUNCTION
  constexpr const Accessor& data() const {
    return accessor_;
  }

  /// \brief Get a deep copy of the vector
  KOKKOS_INLINE_FUNCTION 
  constexpr deep_copy_t copy() const {
    return *this;
  }

  /// \brief Cast (and copy) the vector to a different type
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr auto cast() const {
    return impl::cast_impl<U>(std::make_index_sequence<N>{}, *this);
  }
  //@}

  //! \name Setters and modifiers
  //@{

  /// \brief Set all elements of the vector
  template <typename... Args>
    requires(sizeof...(Args) == N) && (std::is_convertible_v<Args, T> && ...) && HasNonConstAccessOperator<Accessor, T>
  KOKKOS_INLINE_FUNCTION constexpr void set(Args&&... args) {
    impl::set_from_args_impl(std::make_index_sequence<N>{}, *this, static_cast<T>(std::forward<Args>(args))...);
  }

  /// \brief Set all elements of the vector using an accessor
  /// \param[in] accessor A valid accessor.
  /// \note A AVector is also a valid accessor.
  template <ValidAccessor<T> OtherAccessor>
  KOKKOS_INLINE_FUNCTION constexpr void set(const OtherAccessor& accessor)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::set_from_accessor_impl(std::make_index_sequence<N>{}, *this, accessor);
  }

  /// \brief Set all elements of the vector to a single value
  /// \param[in] value The value to set all elements to.
  KOKKOS_INLINE_FUNCTION
  constexpr void fill(const T& value)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::fill_impl(std::make_index_sequence<N>{}, *this, value);
  }
  //@}

  //! \name Unary operators
  //@{

  /// \brief Unary plus operator
  KOKKOS_INLINE_FUNCTION
  constexpr AVector<T, N> operator+() const {
    return *this;
  }

  /// \brief Unary minus operator
  KOKKOS_INLINE_FUNCTION
  constexpr AVector<T, N> operator-() const {
    return impl::unary_minus_impl(std::make_index_sequence<N>{}, *this);
  }
  //@}

  //! \name Addition and subtraction
  //@{

  /// \brief AVector-vector addition
  /// \param[in] other The other vector.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr auto operator+(const AVector<U, N, OtherAccessor, OtherOwnershipType>& other) const {
    return impl::vector_vector_add_impl(std::make_index_sequence<N>{}, *this, other);
  }

  /// \brief AVector-vector addition
  /// \param[in] other The other vector.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr AVector<T, N, Accessor, Ownership::Owns>& operator+=(
      const AVector<U, N, OtherAccessor, OtherOwnershipType>& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_vector_add_impl(std::make_index_sequence<N>{}, *this, other);
    return *this;
  }

  /// \brief AVector-vector subtraction
  /// \param[in] other The other vector.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr auto operator-(const AVector<U, N, OtherAccessor, OtherOwnershipType>& other) const {
    return impl::vector_vector_subtraction_impl(std::make_index_sequence<N>{}, *this, other);
  }

  /// \brief Self-vector subtraction
  /// \param[in] other The other vector.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr AVector<T, N, Accessor, Ownership::Owns>& operator-=(
      const AVector<U, N, OtherAccessor, OtherOwnershipType>& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_vector_subtraction_impl(std::make_index_sequence<N>{}, *this, other);
    return *this;
  }

  /// \brief AVector-scalar addition
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr auto operator+(const U& scalar) const {
    return impl::vector_scalar_add_impl(std::make_index_sequence<N>{}, *this, scalar);
  }

  /// \brief Self-scalar addition
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr AVector<T, N, Accessor, Ownership::Owns>& operator+=(const U& scalar)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_scalar_add_impl(std::make_index_sequence<N>{}, *this, scalar);
    return *this;
  }

  /// \brief AVector-scalar subtraction
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr auto operator-(const U& scalar) const {
    return impl::vector_scalar_subtraction_impl(std::make_index_sequence<N>{}, *this, scalar);
  }

  /// \brief AVector-scalar subtraction
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr AVector<T, N, Accessor, Ownership::Owns>& operator-=(const U& scalar)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_scalar_subtraction_impl(std::make_index_sequence<N>{}, *this, scalar);
    return *this;
  }
  //@}

  //! \name Multiplication and division
  //@{

  /// \brief AVector-scalar multiplication
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr auto operator*(const U& scalar) const {
    return impl::vector_scalar_multiplication_impl(std::make_index_sequence<N>{}, *this, scalar);
  }

  /// \brief Self-scalar multiplication
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr AVector<T, N, Accessor, Ownership::Owns>& operator*=(const U& scalar)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_scalar_multiplication_impl(std::make_index_sequence<N>{}, *this, scalar);
    return *this;
  }

  /// \brief AVector-scalar division. (Type promotes the result to a double if the scalar is not a floating point.)
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr auto operator/(const U& scalar) const {
    return impl::vector_scalar_division_impl(std::make_index_sequence<N>{}, *this, scalar);
  }

  /// \brief Self-scalar division (Does not type-promote the result!!).
  /// \note Because there is no type promotion, this will perform integer division if the scalar is an integer.
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr AVector<T, N, Accessor, Ownership::Owns>& operator/=(const U& scalar)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_scalar_division_impl(std::make_index_sequence<N>{}, *this, scalar);
    return *this;
  }
  //@}

  //! \name Static methods
  //@{

  /// \brief Get a vector of ones
  KOKKOS_INLINE_FUNCTION static constexpr AVector<T, N> ones() {
    return ones_impl(std::make_index_sequence<N>{});
  }

  /// \brief Get the zero vector
  KOKKOS_INLINE_FUNCTION static constexpr AVector<T, N> zeros() {
    return zeros_impl(std::make_index_sequence<N>{});
  }
  //@}

  //! \name Friends <3
  //@{

  // Declare the << operator as a friend
  template <typename U, size_t M, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  friend std::ostream& operator<<(std::ostream& os, const AVector<U, M, OtherAccessor, OtherOwnershipType>& vec);

  // We are friends with all AVectors regardless of their Accessor, type, or ownership
  template <typename U, size_t M, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
    requires std::is_arithmetic_v<U>
  friend class AVector;
  //@}

 private:
  //! \name Private helper functions
  //@{

  /// \brief Get a vector of ones
  template <size_t... Is>
  KOKKOS_INLINE_FUNCTION static constexpr AVector<T, N> ones_impl(std::index_sequence<Is...>) {
    AVector<std::remove_const_t<T>, N> result;
    ((result[Is] = static_cast<T>(1)), ...);
    return result;
  }

  /// \brief Get a vector of zeros
  template <size_t... Is>
  KOKKOS_INLINE_FUNCTION static constexpr AVector<T, N> zeros_impl(std::index_sequence<Is...>) {
    AVector<std::remove_const_t<T>, N> result;
    ((result[Is] = static_cast<T>(0)), ...);
    return result;
  }
  //@}
};  // class AVector

template <typename T, size_t N, ValidAccessor<T> Accessor = Array<T, N>>
  requires std::is_arithmetic_v<T>
using VectorView = AVector<T, N, Accessor, Ownership::Views>;

template <typename T, size_t N, ValidAccessor<T> Accessor = Array<T, N>>
  requires std::is_arithmetic_v<T>
using OwningVector = AVector<T, N, Accessor, Ownership::Owns>;

template <typename T, size_t N>
  requires std::is_arithmetic_v<T>
using Vector = OwningVector<T, N, Array<T, N>>;

static_assert(is_vector_v<AVector<int, 3>>, "Odd, default AVector is not a vector.");
static_assert(is_vector_v<AVector<int, 3, Array<int, 3>>>, "Odd, default vector with Array accessor is not a vector.");
static_assert(is_vector_v<VectorView<int, 3>>, "Odd, VectorView is not a vector.");
static_assert(is_vector_v<Vector<int, 3>>, "Odd, Vector is not a vector.");

//! \name Non-member functions
//@{

//! \name Write to output stream
//@{

/// \brief Write the vector to an output stream
/// \param[in] os The output stream.
/// \param[in] vec The vector.
template <typename T, size_t N, ValidAccessor<T> Accessor, typename OwnershipType>
std::ostream& operator<<(std::ostream& os, const AVector<T, N, Accessor, OwnershipType>& vec) {
  os << "(";
  if constexpr (N == 0) {
    // Do nothing
  } else if constexpr (N == 1) {
    os << vec[0];
  } else {
    for (size_t i = 0; i < N; ++i) {
      os << vec[i];
      if (i < N - 1) {
        os << ", ";
      }
    }
  }
  os << ")";
  return os;
}
//@}

//! \name Non-member comparison functions
//@{

/// TODO(palmerb4): These really shouldn't be in the vector class. They should be in a separate file.
/// \brief Scalar-scalar equality (within a tolerance)
/// \param[in] scalar1 The first scalar.
/// \param[in] scalar2 The second scalar.
/// \param[in] tol The tolerance (default is determined by the given type).
template <typename U, typename T>
  requires std::is_arithmetic_v<U> && std::is_arithmetic_v<T>
KOKKOS_INLINE_FUNCTION constexpr bool is_close(
    const U& scalar1, const T& scalar2,
    const decltype(get_comparison_tolerance<T, U>())& tol = get_comparison_tolerance<T, U>()) {
  // Use the tolerance type as the comparison type
  using ComparisonType = std::remove_reference_t<decltype(tol)>;
  return std::abs(static_cast<ComparisonType>(scalar1) - static_cast<ComparisonType>(scalar2)) <= tol;
}

/// \brief Scalar-scalar equality (within a relaxed tolerance)
/// \param[in] scalar1 The first scalar.
/// \param[in] scalar2 The second scalar.
/// \param[in] tol The tolerance (default is determined by the given type).
template <typename U, typename T>
  requires std::is_arithmetic_v<U> && std::is_arithmetic_v<T>
KOKKOS_INLINE_FUNCTION constexpr bool is_approx_close(
    const U& scalar1, const T& scalar2,
    const decltype(get_relaxed_comparison_tolerance<T, U>())& tol = get_relaxed_comparison_tolerance<T, U>()) {
  return is_close(scalar1, scalar2, tol);
}

/// \brief AVector-vector equality (element-wise within a tolerance)
/// \param[in] vec1 The first vector.
/// \param[in] vec2 The second vector.
/// \param[in] tol The tolerance (default is determined by the given type).
template <size_t N, typename U, typename T, ValidAccessor<U> Accessor1, typename Ownership1, ValidAccessor<T> Accessor2,
          typename Ownership2>
KOKKOS_INLINE_FUNCTION constexpr bool is_close(
    const AVector<U, N, Accessor1, Ownership1>& vec1, const AVector<T, N, Accessor2, Ownership2>& vec2,
    const decltype(get_comparison_tolerance<T, U>())& tol = get_comparison_tolerance<T, U>()) {
  return impl::is_close_impl(std::make_index_sequence<N>{}, vec1, vec2, tol);
}

/// \brief AVector-vector equality (element-wise within a relaxed tolerance)
/// \param[in] vec1 The first vector.
/// \param[in] vec2 The second vector.
/// \param[in] tol The tolerance (default is determined by the given type).
template <size_t N, typename U, typename T, ValidAccessor<U> Accessor1, typename Ownership1, ValidAccessor<T> Accessor2,
          typename Ownership2>
KOKKOS_INLINE_FUNCTION constexpr bool is_approx_close(
    const AVector<U, N, Accessor1, Ownership1>& vec1, const AVector<T, N, Accessor2, Ownership2>& vec2,
    const decltype(get_relaxed_comparison_tolerance<T, U>())& tol = get_relaxed_comparison_tolerance<T, U>()) {
  return is_close(vec1, vec2, tol);
}
//@}

//! \name Non-member addition and subtraction operators
//@{

/// \brief Scalar-vector addition
/// \param[in] scalar The scalar.
/// \param[in] vec The vector.
template <size_t N, typename U, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto operator+(const U& scalar, const AVector<T, N, Accessor, OwnershipType>& vec)
    -> AVector<std::common_type_t<T, U>, N> {
  return vec + scalar;
}

/// \brief Scalar-vector subtraction
/// \param[in] scalar The scalar.
/// \param[in] vec The vector.
template <size_t N, typename U, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto operator-(const U& scalar, const AVector<T, N, Accessor, OwnershipType>& vec)
    -> AVector<std::common_type_t<T, U>, N> {
  return -vec + scalar;
}
//@}

//! \name Non-member multiplication and division operators
//@{

/// \brief Scalar-vector multiplication
/// \param[in] scalar The scalar.
/// \param[in] vec The vector.
template <size_t N, typename U, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto operator*(const U& scalar, const AVector<T, N, Accessor, OwnershipType>& vec)
    -> AVector<std::common_type_t<T, U>, N> {
  return vec * scalar;
}
//@}

//! \name Basic arithmetic reduction operations
//@{

/// \brief Sum of all elements
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto sum(const AVector<T, N, Accessor, OwnershipType>& vec) {
  return impl::sum_impl(std::make_index_sequence<N>{}, vec);
}

/// \brief Product of all elements
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto product(const AVector<T, N, Accessor, OwnershipType>& vec) {
  return impl::product_impl(std::make_index_sequence<N>{}, vec);
}

/// \brief Minimum element
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto min(const AVector<T, N, Accessor, OwnershipType>& vec) {
  return impl::min_impl(std::make_index_sequence<N>{}, vec);
}

/// \brief Maximum element
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto max(const AVector<T, N, Accessor, OwnershipType>& vec) {
  return impl::max_impl(std::make_index_sequence<N>{}, vec);
}

/// \brief Mean of all elements (returns a double if T is an integral type, otherwise returns T)
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, double, T>>
KOKKOS_INLINE_FUNCTION constexpr OutputType mean(const AVector<T, N, Accessor, OwnershipType>& vec) {
  auto vec_sum = sum(vec);
  return static_cast<OutputType>(vec_sum) / OutputType(N);
}

/// \brief Mean of all elements (returns a float if T is an integral type, otherwise returns T)
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, float, T>>
KOKKOS_INLINE_FUNCTION constexpr OutputType mean_f(const AVector<T, N, Accessor, OwnershipType>& vec) {
  return mean(vec);
}

/// \brief Variance of all elements (returns a double if T is an integral type, otherwise returns T)
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, double, T>>
KOKKOS_INLINE_FUNCTION constexpr OutputType variance(const AVector<T, N, Accessor, OwnershipType>& vec) {
  return impl::variance_impl(std::make_index_sequence<N>{}, vec);
}

/// \brief Variance of all elements (returns a float if T is an integral type, otherwise returns T)
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, float, T>>
KOKKOS_INLINE_FUNCTION constexpr OutputType variance_f(const AVector<T, N, Accessor, OwnershipType>& vec) {
  return variance(vec);
}

/// \brief Standard deviation of all elements (returns a double if T is an integral type, otherwise returns T)
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, double, T>>
KOKKOS_INLINE_FUNCTION constexpr OutputType stddev(const AVector<T, N, Accessor, OwnershipType>& vec) {
  return impl::standard_deviation_impl(std::make_index_sequence<N>{}, vec);
}

/// \brief Standard deviation of all elements (returns a float if T is an integral type, otherwise returns T)
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, float, T>>
KOKKOS_INLINE_FUNCTION constexpr OutputType stddev_f(const AVector<T, N, Accessor, OwnershipType>& vec) {
  return stddev(vec);
}
//@}

//! \name Special vector operations
//@{

/// \brief Dot product of two vectors
/// \param[in] a The first vector.
/// \param[in] b The second vector.
template <size_t N, typename U, typename T, ValidAccessor<U> Accessor1, typename Ownership1, ValidAccessor<T> Accessor2,
          typename Ownership2>
KOKKOS_INLINE_FUNCTION constexpr auto dot(const AVector<U, N, Accessor1, Ownership1>& a,
                                          const AVector<T, N, Accessor2, Ownership2>& b) -> std::common_type_t<T, U> {
  return impl::dot_product_impl(std::make_index_sequence<N>{}, a, b);
}

/// \brief Element-wise product
/// \param[in] a The first vector.
/// \param[in] b The second vector.
template <size_t N, typename U, typename T, ValidAccessor<U> Accessor1, typename Ownership1, ValidAccessor<T> Accessor2,
          typename Ownership2>
KOKKOS_INLINE_FUNCTION constexpr auto elementwise_mul(const AVector<U, N, Accessor1, Ownership1>& a,
                                                      const AVector<T, N, Accessor2, Ownership2>& b) {
  return impl::vector_vector_elementwise_mul_impl(std::make_index_sequence<N>{}, a, b);
}

/// \brief Element-wise division
/// \param[in] a The first vector.
/// \param[in] b The second vector.
template <size_t N, typename U, typename T, ValidAccessor<U> Accessor1, typename Ownership1, ValidAccessor<T> Accessor2,
          typename Ownership2>
KOKKOS_INLINE_FUNCTION constexpr auto elementwise_div(const AVector<U, N, Accessor1, Ownership1>& a,
                                                      const AVector<T, N, Accessor2, Ownership2>& b) {
  return impl::vector_vector_elementwise_div_impl(std::make_index_sequence<N>{}, a, b);
}

/// \brief Apply a function to each element of the vector
/// \param[in] func The function to apply.
/// \param[in] vec The vector.
template <typename Func, size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto apply(Func&& func, const AVector<T, N, Accessor, OwnershipType>& vec)
    -> AVector<std::invoke_result_t<Func, T>, N> {
  return impl::apply_impl(std::make_index_sequence<N>{}, std::forward<Func>(func), vec);
}
//@}

//! \name AVector norms
//@{

/// \brief AVector infinity norm
/// \param[in] vec The vector.
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto infinity_norm(const AVector<T, N, Accessor, OwnershipType>& vec) {
  return max(vec);
}

/// \brief AVector 1-norm
/// \param[in] vec The vector.
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto one_norm(const AVector<T, N, Accessor, OwnershipType>& vec) {
  return sum(vec);
}

/// \brief AVector 2-norm (Returns a double if T is an integral type, otherwise returns T)
/// \param[in] vec The vector.
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, double, T>>
KOKKOS_INLINE_FUNCTION constexpr OutputType two_norm(const AVector<T, N, Accessor, OwnershipType>& vec) {
  return std::sqrt(static_cast<OutputType>(dot(vec, vec)));
}

/// \brief AVector 2-norm (Returns a float if T is an integral type, otherwise returns T)
/// \param[in] vec The vector.
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, float, T>>
KOKKOS_INLINE_FUNCTION constexpr OutputType two_norm_f(const AVector<T, N, Accessor, OwnershipType>& vec) {
  return two_norm(vec);
}

/// \brief AVector squared 2-norm
/// \param[in] vec The vector.
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto two_norm_squared(const AVector<T, N, Accessor, OwnershipType>& vec) {
  return dot(vec, vec);
}

/// \brief Default vector norm (2-norm, returns a double if T is an integral type, otherwise returns T)
/// \param[in] vec The vector.
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, double, T>>
KOKKOS_INLINE_FUNCTION constexpr OutputType norm(const AVector<T, N, Accessor, OwnershipType>& vec) {
  return two_norm(vec);
}

/// \brief Default vector norm (2-norm, returns a float if T is an integral type, otherwise returns T)
/// \param[in] vec The vector.
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, float, T>>
KOKKOS_INLINE_FUNCTION constexpr OutputType norm_f(const AVector<T, N, Accessor, OwnershipType>& vec) {
  return norm(vec);
}

/// \brief Default vector norm squared (2-norm)
/// \param[in] vec The vector.
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto norm_squared(const AVector<T, N, Accessor, OwnershipType>& vec) {
  return two_norm_squared(vec);
}

/// \brief Minor angle between two vectors (returns a double if T is an integral type, otherwise returns T)
/// \param[in] a The first vector.
/// \param[in] b The second vector.
template <size_t N, typename U, typename T, ValidAccessor<U> Accessor1, typename Ownership1, ValidAccessor<T> Accessor2,
          typename Ownership2, typename OutputType = std::conditional_t<std::is_integral_v<T>, double, T>>
KOKKOS_INLINE_FUNCTION constexpr OutputType minor_angle(const AVector<U, N, Accessor1, Ownership1>& a,
                                                        const AVector<T, N, Accessor2, Ownership2>& b) {
  return std::acos(static_cast<OutputType>(dot(a, b)) /
                   (static_cast<OutputType>(two_norm(a)) * static_cast<OutputType>(two_norm(b))));
}

/// \brief Minor angle between two vectors (returns a float if T is an integral type, otherwise returns T)
/// \param[in] a The first vector.
/// \param[in] b The second vector.
template <size_t N, typename U, typename T, ValidAccessor<U> Accessor1, typename Ownership1, ValidAccessor<T> Accessor2,
          typename Ownership2, typename OutputType = std::conditional_t<std::is_integral_v<T>, float, T>>
KOKKOS_INLINE_FUNCTION constexpr OutputType minor_angle_f(const AVector<U, N, Accessor1, Ownership1>& a,
                                                          const AVector<T, N, Accessor2, Ownership2>& b) {
  return minor_angle(a, b);
}

/// \brief Major angle between two vectors (returns a double if T is an integral type, otherwise returns T)
/// \param[in] a The first vector.
/// \param[in] b The second vector.
template <size_t N, typename U, typename T, ValidAccessor<U> Accessor1, typename Ownership1, ValidAccessor<T> Accessor2,
          typename Ownership2, typename OutputType = std::conditional_t<std::is_integral_v<T>, double, T>>
KOKKOS_INLINE_FUNCTION constexpr OutputType major_angle(const AVector<U, N, Accessor1, Ownership1>& a,
                                                        const AVector<T, N, Accessor2, Ownership2>& b) {
  return OutputType(M_PI) - minor_angle(a, b);
}

/// \brief Major angle between two vectors (returns a float if T is an integral type, otherwise returns T)
/// \param[in] a The first vector.
/// \param[in] b The second vector.
template <size_t N, typename U, typename T, ValidAccessor<U> Accessor1, typename Ownership1, ValidAccessor<T> Accessor2,
          typename Ownership2, typename OutputType = std::conditional_t<std::is_integral_v<T>, float, T>>
KOKKOS_INLINE_FUNCTION constexpr OutputType major_angle_f(const AVector<U, N, Accessor1, Ownership1>& a,
                                                          const AVector<T, N, Accessor2, Ownership2>& b) {
  return major_angle(a, b);
}
//@}

//! \name atomic_load/store. Atomic memory management operations.
//@{

/// \brief Atomic v_copy = v.
///
/// Note: Even if the input is a view, the return is a plain owning vector.
template <size_t N, typename T, ValidAccessor<T> A, typename OT>
KOKKOS_INLINE_FUNCTION AVector<T, N> atomic_load(AVector<T, N, A, OT>* const v) {
  return impl::atomic_vector_load_impl(std::make_index_sequence<N>{}, v);
}

/// \brief Atomic v[i] = s.
template <size_t N, typename T1, ValidAccessor<T1> A, typename OT, typename T2>
KOKKOS_INLINE_FUNCTION void atomic_store(AVector<T1, N, A, OT>* const v, const T2& s) {
  impl::atomic_vector_scalar_store_impl(std::make_index_sequence<N>{}, v, s);
}

/// \brief Atomic v1[i] = v2[i].
template <size_t N, typename T1, ValidAccessor<T1> A1, typename OT1, typename T2, ValidAccessor<T2> A2, typename OT2>
KOKKOS_INLINE_FUNCTION void atomic_store(AVector<T1, N, A1, OT1>* const v1, const AVector<T2, N, A2, OT2>& v2) {
  impl::atomic_vector_vector_store_impl(std::make_index_sequence<N>{}, v1, v2);
}
//@}

//! \name atomic_[op] Atomic operation which don’t return anything. [op] might be add, sub, mul, div.
//@{

#define MUNDY_MATH_VECTOR_SCALAR_ATOMIC_OP(op_name)                                                     \
  template <size_t N, typename T1, ValidAccessor<T1> A1, typename OT1, typename T2>                     \
  KOKKOS_INLINE_FUNCTION void atomic_##op_name(AVector<T1, N, A1, OT1>* const v, const T2& s) {         \
    impl::atomic_vector_scalar_##op_name##_impl(std::make_index_sequence<N>{}, v, s);                   \
  }                                                                                                     

#define MUNDY_MATH_VECTOR_VECTOR_ATOMIC_OP(op_name)                                                                    \
  template <size_t N, typename T1, ValidAccessor<T1> A1, typename OT1, typename T2, ValidAccessor<T2> A2,              \
            typename OT2>                                                                                              \
  KOKKOS_INLINE_FUNCTION void atomic_##op_name(AVector<T1, N, A1, OT1>* const v1, const AVector<T2, N, A2, OT2>& v2) { \
    impl::atomic_vector_vector_##op_name##_impl(std::make_index_sequence<N>{}, v1, v2);                                \
  }                                                                                                                    

/// \brief Atomic v[i] += s
MUNDY_MATH_VECTOR_SCALAR_ATOMIC_OP(add)

/// \brief Atomic v[i] -= s
MUNDY_MATH_VECTOR_SCALAR_ATOMIC_OP(sub)

/// \brief Atomic v[i] *= s
MUNDY_MATH_VECTOR_SCALAR_ATOMIC_OP(mul)

/// \brief Atomic v[i] /= s
MUNDY_MATH_VECTOR_SCALAR_ATOMIC_OP(div)

/// \brief Atomic v1[i] += v2[i]
MUNDY_MATH_VECTOR_VECTOR_ATOMIC_OP(add)

/// \brief Atomic v1[i] -= v2[i]
MUNDY_MATH_VECTOR_VECTOR_ATOMIC_OP(sub)

/// \brief Atomic v1[i] *= v2[i]
MUNDY_MATH_VECTOR_VECTOR_ATOMIC_OP(elementwise_mul)

/// \brief Atomic v1[i] /= v2[i]
MUNDY_MATH_VECTOR_VECTOR_ATOMIC_OP(elementwise_div)
//@}

//! \name atomic_fetch_[op] Various atomic operations which return the old value. [op] might be add, sub, mul, div.
//
// Note: Even if the input is a view, the return is a plain owning vector.
//@{

#define MUNDY_MATH_VECTOR_SCALAR_ATOMIC_FETCH_OP(op_name)                                             \
  template <size_t N, typename T1, ValidAccessor<T1> A1, typename OT1, typename T2>                   \
  KOKKOS_INLINE_FUNCTION auto atomic_fetch_##op_name(AVector<T1, N, A1, OT1>* const v, const T2& s) { \
    return impl::vector_scalar_atomic_fetch_##op_name##_impl(std::make_index_sequence<N>{}, v, s);    \
  }

#define MUNDY_MATH_VECTOR_VECTOR_ATOMIC_FETCH_OP(op_name)                                                 \
  template <size_t N, typename T1, ValidAccessor<T1> A1, typename OT1, typename T2, ValidAccessor<T2> A2, \
            typename OT2>                                                                                 \
  KOKKOS_INLINE_FUNCTION auto atomic_fetch_##op_name(AVector<T1, N, A1, OT1>* const v1,                   \
                                                     const AVector<T2, N, A2, OT2>& v2) {                 \
    return impl::vector_vector_atomic_fetch_##op_name##_impl(std::make_index_sequence<N>{}, v1, v2);      \
  }

/// \brief Atomic v[i] += s (returns old v)
MUNDY_MATH_VECTOR_SCALAR_ATOMIC_FETCH_OP(add)

/// \brief Atomic v[i] -= s (returns old v)
MUNDY_MATH_VECTOR_SCALAR_ATOMIC_FETCH_OP(sub)

/// \brief Atomic v[i] *= s (returns old v)
MUNDY_MATH_VECTOR_SCALAR_ATOMIC_FETCH_OP(mul)

/// \brief Atomic v[i] /= s (returns old v)
MUNDY_MATH_VECTOR_SCALAR_ATOMIC_FETCH_OP(div)

/// \brief Atomic v1[i] += v2[i] (returns old v1)
MUNDY_MATH_VECTOR_VECTOR_ATOMIC_FETCH_OP(add)

/// \brief Atomic v1[i] -= v2[i] (returns old v1)
MUNDY_MATH_VECTOR_VECTOR_ATOMIC_FETCH_OP(sub)

/// \brief Atomic v1[i] *= v2[i] (returns old v1)
MUNDY_MATH_VECTOR_VECTOR_ATOMIC_FETCH_OP(elementwise_mul)

/// \brief Atomic v1[i] /= v2[i] (returns old v1)
MUNDY_MATH_VECTOR_VECTOR_ATOMIC_FETCH_OP(elementwise_div)
//@}

//! \name atomic_[op]_fetch Various atomic operations which return the new value. [op] might be add, sub, mul, div.
//
// Note: Even if the input is a view, the return is a plain owning vector.
//@{

#define MUNDY_MATH_VECTOR_SCALAR_ATOMIC_OP_FETCH(op_name)                                               \
  template <size_t N, typename T1, ValidAccessor<T1> A1, typename OT1, typename T2>                     \
  KOKKOS_INLINE_FUNCTION auto atomic_##op_name##_fetch(AVector<T1, N, A1, OT1>* const v, const T2& s) { \
    return impl::vector_scalar_atomic_##op_name##_fetch_impl(std::make_index_sequence<N>{}, v, s);      \
  }

#define MUNDY_MATH_VECTOR_VECTOR_ATOMIC_OP_FETCH(op_name)                                                 \
  template <size_t N, typename T1, ValidAccessor<T1> A1, typename OT1, typename T2, ValidAccessor<T2> A2, \
            typename OT2>                                                                                 \
  KOKKOS_INLINE_FUNCTION auto atomic_##op_name##_fetch(AVector<T1, N, A1, OT1>* const v1,                 \
                                                       const AVector<T2, N, A2, OT2>& v2) {               \
    return impl::vector_vector_atomic_##op_name##_fetch_impl(std::make_index_sequence<N>{}, v1, v2);      \
  }

/// \brief Atomic v[i] += s (returns new v)
MUNDY_MATH_VECTOR_SCALAR_ATOMIC_OP_FETCH(add)

/// \brief Atomic v[i] -= s (returns new v)
MUNDY_MATH_VECTOR_SCALAR_ATOMIC_OP_FETCH(sub)

/// \brief Atomic v[i] *= s (returns new v)
MUNDY_MATH_VECTOR_SCALAR_ATOMIC_OP_FETCH(mul)

/// \brief Atomic v[i] /= s (returns new v)
MUNDY_MATH_VECTOR_SCALAR_ATOMIC_OP_FETCH(div)

/// \brief Atomic v1[i] += v2[i] (returns new v1)
MUNDY_MATH_VECTOR_VECTOR_ATOMIC_OP_FETCH(add)

/// \brief Atomic v1[i] -= v2[i] (returns new v1)
MUNDY_MATH_VECTOR_VECTOR_ATOMIC_OP_FETCH(sub)

/// \brief Atomic v1[i] *= v2[i] (returns new v1)
MUNDY_MATH_VECTOR_VECTOR_ATOMIC_OP_FETCH(elementwise_mul)

/// \brief Atomic v1[i] /= v2[i] (returns new v1)
MUNDY_MATH_VECTOR_VECTOR_ATOMIC_OP_FETCH(elementwise_div)
//@}

// Just to double check
static_assert(std::is_trivially_copyable_v<AVector<double, 3>>);
static_assert(std::is_trivially_destructible_v<AVector<double, 3>>);
static_assert(std::is_copy_constructible_v<AVector<double, 3>>);
static_assert(std::is_move_constructible_v<AVector<double, 3>>);

//! \name Type specializations
//@{

#define MUNDY_MATH_VECTOR_SIZE_SPECIALIZATION(alias, alias_lower, N)                                       \
  template <typename T, ValidAccessor<T> Accessor = Array<T, N>, typename OwnershipType = Ownership::Owns> \
    requires std::is_arithmetic_v<T>                                                                       \
  using A##alias = AVector<T, N, Accessor, OwnershipType>;                                                 \
  template <typename T, ValidAccessor<T> Accessor = Array<T, N>>                                           \
    requires std::is_arithmetic_v<T>                                                                       \
  using alias##View = AVector<T, N, Accessor, Ownership::Views>;                                           \
  template <typename T, ValidAccessor<T> Accessor = Array<T, N>>                                           \
    requires std::is_arithmetic_v<T>                                                                       \
  using Owning##alias = AVector<T, N, Accessor, Ownership::Owns>;                                          \
  template <typename T>                                                                                    \
    requires std::is_arithmetic_v<T>                                                                       \
  using alias = Owning##alias<T>;                                                                          \
  template <typename TypeToCheck>                                                                          \
  struct is_##alias_lower##_impl : std::false_type {};                                                     \
  template <typename T, typename Accessor, typename OwnershipType>                                         \
  struct is_##alias_lower##_impl<A##alias<T, Accessor, OwnershipType>> : std::true_type {};                \
  template <typename TypeToCheck>                                                                          \
  struct is_##alias_lower : public is_##alias_lower##_impl<std::decay_t<TypeToCheck>> {};                  \
  template <typename TypeToCheck>                                                                          \
  constexpr bool is_##alias_lower##_v = is_##alias_lower<TypeToCheck>::value;

#define MUNDY_MATH_VECTOR_TYPE_AND_SIZE_SPECIALIZATION(alias, alias_lower, T, N)               \
  template <ValidAccessor<T> Accessor = Array<T, N>, typename OwnershipType = Ownership::Owns> \
  using A##alias = AVector<T, N, Accessor, OwnershipType>;                                     \
  template <ValidAccessor<T> Accessor = Array<T, N>>                                           \
  using alias##View = AVector<T, N, Accessor, Ownership::Views>;                               \
  template <ValidAccessor<T> Accessor = Array<T, N>>                                           \
  using Owning##alias = AVector<T, N, Accessor, Ownership::Owns>;                              \
  using alias = Owning##alias<>;                                                               \
  template <typename TypeToCheck>                                                              \
  struct is_##alias_lower##_impl : std::false_type {};                                         \
  template <typename Accessor, typename OwnershipType>                                         \
  struct is_##alias_lower##_impl<A##alias<Accessor, OwnershipType>> : std::true_type {};       \
  template <typename TypeToCheck>                                                              \
  struct is_##alias_lower : public is_##alias_lower##_impl<std::decay_t<TypeToCheck>> {};      \
  template <typename TypeToCheck>                                                              \
  constexpr bool is_##alias_lower##_v = is_##alias_lower<TypeToCheck>::value;

MUNDY_MATH_VECTOR_SIZE_SPECIALIZATION(Vector1, vector1, 1)
MUNDY_MATH_VECTOR_SIZE_SPECIALIZATION(Vector2, vector2, 2)
MUNDY_MATH_VECTOR_SIZE_SPECIALIZATION(Vector3, vector3, 3)
MUNDY_MATH_VECTOR_SIZE_SPECIALIZATION(Vector4, vector4, 4)
MUNDY_MATH_VECTOR_SIZE_SPECIALIZATION(Vector5, vector5, 5)
MUNDY_MATH_VECTOR_SIZE_SPECIALIZATION(Vector6, vector6, 6)

MUNDY_MATH_VECTOR_TYPE_AND_SIZE_SPECIALIZATION(Vector1d, vector1d, double, 1)
MUNDY_MATH_VECTOR_TYPE_AND_SIZE_SPECIALIZATION(Vector2d, vector2d, double, 2)
MUNDY_MATH_VECTOR_TYPE_AND_SIZE_SPECIALIZATION(Vector3d, vector3d, double, 3)
MUNDY_MATH_VECTOR_TYPE_AND_SIZE_SPECIALIZATION(Vector4d, vector4d, double, 4)
MUNDY_MATH_VECTOR_TYPE_AND_SIZE_SPECIALIZATION(Vector5d, vector5d, double, 5)
MUNDY_MATH_VECTOR_TYPE_AND_SIZE_SPECIALIZATION(Vector6d, vector6d, double, 6)

MUNDY_MATH_VECTOR_TYPE_AND_SIZE_SPECIALIZATION(Vector1f, vector1f, float, 1)
MUNDY_MATH_VECTOR_TYPE_AND_SIZE_SPECIALIZATION(Vector2f, vector2f, float, 2)
MUNDY_MATH_VECTOR_TYPE_AND_SIZE_SPECIALIZATION(Vector3f, vector3f, float, 3)
MUNDY_MATH_VECTOR_TYPE_AND_SIZE_SPECIALIZATION(Vector4f, vector4f, float, 4)
MUNDY_MATH_VECTOR_TYPE_AND_SIZE_SPECIALIZATION(Vector5f, vector5f, float, 5)
MUNDY_MATH_VECTOR_TYPE_AND_SIZE_SPECIALIZATION(Vector6f, vector6f, float, 6)
//@}

//! \name AVector views
//@{

/// \brief A helper function to create a VectorView<T, N, Accessor> based on a given accessor.
/// \param[in] data The data accessor.
///
/// In practice, this function is syntactic sugar to avoid having to specify the template parameters
/// when creating a VectorView<T, N, Accessor> from a data accessor.
/// Instead of writing
/// \code
///   VectorView<T, N, Accessor> vec(data);
/// \endcode
/// you can write
/// \code
///   auto vec = get_vector_view<T>(data);
/// \endcode
template <typename T, size_t N, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_vector_view(Accessor& data) {
  return VectorView<T, N, Accessor>(data);
}

template <typename T, size_t N, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_vector_view(Accessor&& data) {
  return VectorView<T, N, Accessor>(std::forward<Accessor>(data));
}

template <typename T, size_t N, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_owning_vector(Accessor& data) {
  return OwningVector<T, N, Accessor>(data);
}

template <typename T, size_t N, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_owning_vector(Accessor&& data) {
  return OwningVector<T, N, Accessor>(std::forward<Accessor>(data));
}
//@}

//@}

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_VECTOR_HPP_
