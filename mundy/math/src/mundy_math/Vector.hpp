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

/// \brief (Implementation) Type trait to determine if a type is a Vector
template <typename TypeToCheck>
struct is_vector_impl : std::false_type {};
//
template <typename T, size_t N, typename Accessor, typename OwnershipType>
struct is_vector_impl<Vector<T, N, Accessor, OwnershipType>> : std::true_type {};

/// \brief Type trait to determine if a type is a Vector
template <typename T>
struct is_vector : public is_vector_impl<std::decay_t<T>> {};
//
template <typename TypeToCheck>
constexpr bool is_vector_v = is_vector<TypeToCheck>::value;

/// \brief A temporary concept to check if a type is a valid Vector type
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
/// The goal of Vector is to be a lightweight class that can be used with Kokkos to perform mathematical operations on
/// vectors in R3. It does not own the data, but rather it is templated on an Accessor type that provides access to the
/// underlying data. This allows us to use Vector with Kokkos Ownership::Views, raw pointers, or any other type that
/// meets the ValidAccessor requirements without copying the data. This is especially important for GPU-compatable code.
///
/// Vectors can be constructed by passing an accessor to the constructor. However, if the accessor has a N-argument
/// constructor, then the Vector can also be constructed by passing the elements directly to the constructor.
/// Similarly, if the accessor has an initializer list constructor, then the Vector can be constructed by passing an
/// initializer list to the constructor. This is a convenience feature which makes working with the default accessor
/// (Array<T, N>) easier. For example, the following are all valid ways to construct a Vector:
///
/// \code{.cpp}
///   // Constructs a Vector with the default accessor (Array<int, N>)
///   Vector<int, 3> vec1({1, 2, 3});
///   Vector<int, 3> vec2(1, 2, 3);
///   Vector<int, 3> vec3(Array<int, 3>({1, 2, 3}));
///   Vector<int, 3> vec4;
///   vec4.set(1, 2, 3);
///
///   // Construct a VectorView from a double array
///   double data[3] = {1.0, 2.0, 3.0};
///   VectorView<double, 3, double*> vec5(data);
///
///   // Do math with Ownership::Views and Vectors interchangeably
///   double mundy::math::dot(vec1, vec5);
/// \endcode
///
/// \note Accessors may be owning or non-owning, that is irrelevant to the Vector class; however, these accessors
/// should be lightweight such that they can be copied around without much overhead. Furthermore, the lifetime of the
/// data underlying the accessor should be as long as the Vector that use it.
template <typename T, size_t N, ValidAccessor<T> Accessor>
  requires std::is_arithmetic_v<T>
class Vector<T, N, Accessor, Ownership::Views> {
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

  /// \brief The size of the vector
  static constexpr size_t size = N;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor since we don't own the data.
  KOKKOS_FUNCTION Vector() = delete;

  /// \brief Constructor for reference accessors
  KOKKOS_FUNCTION
  explicit constexpr Vector(Accessor& accessor)
    requires(!std::is_pointer_v<Accessor>)
      : accessor_(accessor) {
  }

  /// \brief Constructor for pointer accessors
  KOKKOS_FUNCTION
  explicit constexpr Vector(Accessor accessor)
    requires std::is_pointer_v<Accessor>
      : accessor_(accessor) {
  }

  /// \brief Destructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr ~Vector() = default;

  /// \brief Shallow copy constructor. Stores a reference to the accessor in the other vector.
  KOKKOS_FUNCTION constexpr Vector(const Vector<T, N, Accessor, Ownership::Views>& other) : accessor_(other.data()) {
  }

  /// \brief Shallow move constructor. Stores and moves the reference to the accessor from the other vector.
  KOKKOS_FUNCTION constexpr Vector(Vector<T, N, Accessor, Ownership::Views>&& other)
      : accessor_(std::move(other.data())) {
  }

  /// \brief Deep copy assignment operator with different accessor
  /// \details Copies the data from the other vector to our data. This is only enabled if T is not const.
  template <typename OtherAccessor>
  KOKKOS_FUNCTION constexpr Vector<T, N, Accessor, Ownership::Views>& operator=(
      const Vector<T, N, OtherAccessor, Ownership::Views>& other)
    requires(!std::is_same_v<Accessor, OtherAccessor>) && HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(std::make_index_sequence<N>{}, *this, other);
    return *this;
  }

  /// \brief Deep copy assignment operator with different accessor
  /// \details Copies the data from the other vector to our data. This is only enabled if T is not const.
  template <typename OtherAccessor>
  KOKKOS_FUNCTION constexpr Vector<T, N, Accessor, Ownership::Views>& operator=(
      const Vector<T, N, OtherAccessor, Ownership::Owns>& other)
    requires(!std::is_same_v<Accessor, OtherAccessor>) && HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(std::make_index_sequence<N>{}, *this, other);
    return *this;
  }

  /// \brief Deep copy assignment operator with same accessor
  /// \details Copies the data from the other vector to our data. This is only enabled if T is not const.
  /// Yes, this function is necessary. If we only use the version for differing accessor, the compiler can get confused.
  KOKKOS_FUNCTION constexpr Vector<T, N, Accessor, Ownership::Views>& operator=(
      const Vector<T, N, Accessor, Ownership::Views>& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(std::make_index_sequence<N>{}, *this, other);
    return *this;
  }

  /// \brief Deep copy assignment operator with same accessor
  /// \details Copies the data from the other vector to our data. This is only enabled if T is not const.
  /// Yes, this function is necessary. If we only use the version for differing accessor, the compiler can get confused.
  KOKKOS_FUNCTION constexpr Vector<T, N, Accessor, Ownership::Views>& operator=(
      const Vector<T, N, Accessor, Ownership::Owns>& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(std::make_index_sequence<N>{}, *this, other);
    return *this;
  }

  /// \brief Deep copy assignment operator from a single value
  /// \param[in] value The value to set all elements to.
  KOKKOS_FUNCTION constexpr Vector<T, N, Accessor, Ownership::Views>& operator=(const T value)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::fill_impl(std::make_index_sequence<N>{}, *this, value);
    return *this;
  }

  /// \brief Move assignment operator with different accessor.
  /// \details Moves the data from the other vector to our data. This is only enabled if T is not const.
  template <typename OtherAccessor>
  KOKKOS_FUNCTION constexpr Vector<T, N, Accessor, Ownership::Views>& operator=(
      Vector<T, N, OtherAccessor, Ownership::Owns>&& other)
    requires(!std::is_same_v<Accessor, OtherAccessor>) && HasNonConstAccessOperator<Accessor, T>
  {
    // Other owns its accessor but that doesn't mean that it owns the data the accessor accesses.
    // Since the accessor neither has a copy constructor nor a move constructor, we must deep copy.
    impl::deep_copy_impl(std::make_index_sequence<N>{}, *this, std::move(other));
    return *this;
  }

  /// \brief Move assignment operator with same accessor
  /// \details Moves the data from the other vector to our data. This is only enabled if T is not const.
  /// Yes, this function is necessary. If we only use the version for differing accessor, the compiler can get confused.
  KOKKOS_FUNCTION constexpr Vector<T, N, Accessor, Ownership::Views>& operator=(
      Vector<T, N, Accessor, Ownership::Owns>&& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(std::make_index_sequence<N>{}, *this, std::move(other));
    return *this;
  }

  /// \brief Move assignment operator with different accessor.
  /// Same as deep copy since a other's data is not owned.
  /// \details Moves the data from the other vector to our data. This is only enabled if T is not const.
  template <typename OtherAccessor>
  KOKKOS_FUNCTION constexpr Vector<T, N, Accessor, Ownership::Views>& operator=(
      Vector<T, N, OtherAccessor, Ownership::Views>&& other)
    requires(!std::is_same_v<Accessor, OtherAccessor>) && HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(std::make_index_sequence<N>{}, *this, std::move(other));
    return *this;
  }

  /// \brief Move assignment operator with same accessor
  /// Same as deep copy since a other's data is not owned.
  /// \details Moves the data from the other vector to our data. This is only enabled if T is not const.
  /// Yes, this function is necessary. If we only use the version for differing accessor, the compiler can get confused.
  KOKKOS_FUNCTION constexpr Vector<T, N, Accessor, Ownership::Views>& operator=(
      Vector<T, N, Accessor, Ownership::Views>&& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(std::make_index_sequence<N>{}, *this, std::move(other));
    return *this;
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Element access operator via a single index
  /// \param[in] index The index of the element.
  KOKKOS_FUNCTION
  constexpr T& operator[](size_t index) {
    return accessor_[index];
  }

  /// \brief Const element access operator via a single index
  /// \param[in] index The index of the element.
  KOKKOS_FUNCTION
  constexpr const T& operator[](size_t index) const {
    return accessor_[index];
  }

  /// \brief Element access operator via a single index
  /// \param[in] index The index of the element.
  KOKKOS_FUNCTION
  constexpr T& operator()(size_t index) {
    return accessor_[index];
  }

  /// \brief Const element access operator via a single index
  /// \param[in] index The index of the element.
  KOKKOS_FUNCTION
  constexpr const T& operator()(size_t index) const {
    return accessor_[index];
  }

  /// \brief Get the internal data accessor
  KOKKOS_FUNCTION
  constexpr std::conditional_t<std::is_pointer_v<Accessor>, Accessor, Accessor&> data() {
    return accessor_;
  }

  /// \brief Get the internal data accessor
  KOKKOS_FUNCTION
  constexpr const std::conditional_t<std::is_pointer_v<Accessor>, Accessor, Accessor&> data() const {
    return accessor_;
  }
  //@}

  //! \name Setters and modifiers
  //@{

  /// \brief Set all elements of the vector
  template <typename... Args>
    requires(sizeof...(Args) == N) && (std::is_convertible_v<Args, T> && ...) && HasNonConstAccessOperator<Accessor, T>
  KOKKOS_FUNCTION constexpr void set(Args&&... args) {
    impl::set_from_args_impl(std::make_index_sequence<N>{}, *this, static_cast<T>(std::forward<Args>(args))...);
  }

  /// \brief Set all elements of the vector using an accessor
  /// \param[in] accessor A valid accessor.
  /// \note A Vector is also a valid accessor.
  template <ValidAccessor<T> OtherAccessor>
  KOKKOS_FUNCTION constexpr void set(const OtherAccessor& accessor)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::set_from_accessor_impl(std::make_index_sequence<N>{}, *this, accessor);
  }

  /// \brief Set all elements of the vector to a single value
  /// \param[in] value The value to set all elements to.
  KOKKOS_FUNCTION
  constexpr void fill(const T& value)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::fill_impl(std::make_index_sequence<N>{}, *this, value);
  }
  //@}

  //! \name Unary operators
  //@{

  /// \brief Unary plus operator
  KOKKOS_FUNCTION
  constexpr Vector<T, N> operator+() const {
    return *this;
  }

  /// \brief Unary minus operator
  KOKKOS_FUNCTION
  constexpr Vector<T, N> operator-() const {
    return impl::unary_minus_impl(std::make_index_sequence<N>{}, *this);
  }
  //@}

  //! \name Addition and subtraction
  //@{

  /// \brief Vector-vector addition
  /// \param[in] other The other vector.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_FUNCTION constexpr auto operator+(const Vector<U, N, OtherAccessor, OtherOwnershipType>& other) const {
    return impl::vector_vector_add_impl(std::make_index_sequence<N>{}, *this, other);
  }

  /// \brief Vector-vector addition
  /// \param[in] other The other vector.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_FUNCTION constexpr Vector<T, N, Accessor, Ownership::Views>& operator+=(
      const Vector<U, N, OtherAccessor, OtherOwnershipType>& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_vector_add_impl(std::make_index_sequence<N>{}, *this, other);
    return *this;
  }

  /// \brief Vector-vector subtraction
  /// \param[in] other The other vector.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_FUNCTION constexpr auto operator-(const Vector<U, N, OtherAccessor, OtherOwnershipType>& other) const {
    return impl::vector_vector_subtraction_impl(std::make_index_sequence<N>{}, *this, other);
  }

  /// \brief Self-vector subtraction
  /// \param[in] other The other vector.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_FUNCTION constexpr Vector<T, N, Accessor, Ownership::Views>& operator-=(
      const Vector<U, N, OtherAccessor, OtherOwnershipType>& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_vector_subtraction_impl(std::make_index_sequence<N>{}, *this, other);
    return *this;
  }

  /// \brief Vector-scalar addition
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_FUNCTION constexpr auto operator+(const U& scalar) const {
    return impl::vector_scalar_add_impl(std::make_index_sequence<N>{}, *this, scalar);
  }

  /// \brief Self-scalar addition
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_FUNCTION constexpr Vector<T, N, Accessor, Ownership::Views>& operator+=(const U& scalar)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_scalar_add_impl(std::make_index_sequence<N>{}, *this, scalar);
    return *this;
  }

  /// \brief Vector-scalar subtraction
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_FUNCTION constexpr auto operator-(const U& scalar) const {
    return impl::vector_scalar_subtraction_impl(std::make_index_sequence<N>{}, *this, scalar);
  }

  /// \brief Vector-scalar subtraction
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_FUNCTION constexpr Vector<T, N, Accessor, Ownership::Views>& operator-=(const U& scalar)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_scalar_subtraction_impl(std::make_index_sequence<N>{}, *this, scalar);
    return *this;
  }
  //@}

  //! \name Multiplication and division
  //@{

  /// \brief Vector-scalar multiplication
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_FUNCTION constexpr auto operator*(const U& scalar) const {
    return impl::vector_scalar_multiplication_impl(std::make_index_sequence<N>{}, *this, scalar);
  }

  /// \brief Vector-scalar multiplication
  /// \param[in] scalar The scalar.
  template <typename U>
    requires HasNonConstAccessOperator<Accessor, T>
  KOKKOS_FUNCTION constexpr Vector<T, N, Accessor, Ownership::Views>& operator*=(const U& scalar) {
    impl::self_scalar_multiplication_impl(std::make_index_sequence<N>{}, *this, scalar);
    return *this;
  }

  /// \brief Vector-scalar division. (Type promotes the result to a double if the scalar is not a floating point.)
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_FUNCTION constexpr auto operator/(const U& scalar) const {
    return impl::vector_scalar_division_impl(std::make_index_sequence<N>{}, *this, scalar);
  }

  /// \brief Vector-scalar division (Does not type-promote the result!!).
  /// \note Because there is no type promotion, this will perform integer division if the scalar is an integer.
  /// \param[in] scalar The scalar.
  template <typename U>
    requires HasNonConstAccessOperator<Accessor, T>
  KOKKOS_FUNCTION constexpr Vector<T, N, Accessor, Ownership::Views>& operator/=(const U& scalar) {
    impl::self_scalar_division_impl(std::make_index_sequence<N>{}, *this, scalar);
    return *this;
  }
  //@}

  //! \name Friends <3
  //@{

  // Declare the << operator as a friend
  template <typename U, size_t M, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  friend std::ostream& operator<<(std::ostream& os, const Vector<U, M, OtherAccessor, OtherOwnershipType>& vec);

  // We are friends with all Vectors regardless of their Accessor, type, or ownership
  template <typename U, size_t M, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
    requires std::is_arithmetic_v<U>
  friend class Vector;
  //@}
};  // class Vector (non-owning)

template <typename T, size_t N, ValidAccessor<T> Accessor>
  requires std::is_arithmetic_v<T>
class Vector<T, N, Accessor, Ownership::Owns> {
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

  /// \brief The size of the vector
  static constexpr size_t size = N;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor. Assume elements are uninitialized.
  /// \note This constructor is only enabled if the Accessor has a default constructor.
  KOKKOS_DEFAULTED_FUNCTION constexpr Vector()
    requires HasDefaultConstructor<Accessor>
  = default;

  /// \brief Constructor from a given accessor
  /// \param[in] data The accessor.
  KOKKOS_FUNCTION
  constexpr explicit Vector(const Accessor& data)
    requires std::is_copy_constructible_v<Accessor>
      : accessor_(data) {
  }

  /// \brief Constructor from a given accessor
  /// \param[in] data The accessor.
  KOKKOS_FUNCTION
  constexpr explicit Vector(Accessor&& data)
    requires(std::is_copy_constructible_v<Accessor> || std::is_move_constructible_v<Accessor>)
      : accessor_(std::forward<Accessor>(data)) {
  }

  /// \brief Constructor to initialize all elements to a single value.
  /// Requires the number of arguments to be N and the type of each to be T.
  /// Only enabled if the Accessor has a N-argument constructor.
  KOKKOS_FUNCTION constexpr explicit Vector(const T& value)
    requires HasNArgConstructor<Accessor, T, 1>
      : accessor_(value) {
  }

  /// \brief Constructor to initialize all elements explicitly.
  /// Requires the number of arguments to be N and the type of each to be T.
  /// Only enabled if the Accessor has a N-argument constructor.
  template <typename... Args>
    requires(sizeof...(Args) == N) && (N != 1) &&
            (std::is_convertible_v<Args, T> && ...) && HasNArgConstructor<Accessor, T, N>
  KOKKOS_FUNCTION constexpr explicit Vector(Args&&... args) : accessor_{static_cast<T>(std::forward<Args>(args))...} {
  }

  /// \brief Constructor to initialize all elements via initializer list
  /// \param[in] list The initializer list.
  KOKKOS_FUNCTION constexpr Vector(const std::initializer_list<T>& list)
    requires HasInitializerListConstructor<Accessor, T>
      : accessor_(list) {
  }

  /// \brief Destructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr ~Vector() = default;

  /// \brief Deep copy constructor
  KOKKOS_FUNCTION constexpr Vector(const Vector<T, N, Accessor, Ownership::Owns>& other)
    requires HasCopyConstructor<Accessor> && (!std::is_same_v<Accessor, Vector<T, N, Accessor, Ownership::Owns>>)
      : accessor_(other.accessor_) {
  }

  /// \brief Deep copy constructor
  KOKKOS_FUNCTION constexpr Vector(const Vector<T, N, Accessor, Ownership::Views>& other)
    requires HasCopyConstructor<Accessor> && (!std::is_same_v<Accessor, Vector<T, N, Accessor, Ownership::Views>>)
      : accessor_(other.accessor_) {
  }

  /// TODO(palmerb4): CUDA can't handle the fact that the following requires make these constructors mutually exclusive from the above ones.
  /// \brief Deep copy constructor
  KOKKOS_FUNCTION constexpr Vector(const Vector<T, N, Accessor, Ownership::Owns>& other)
    requires(!HasCopyConstructor<Accessor>) &&
            HasNonConstAccessOperator<Accessor, T> && (!std::is_same_v<Accessor, Vector<T, N, Accessor, Ownership::Owns>>)
      : accessor_() {
    impl::deep_copy_impl(std::make_index_sequence<N>{}, *this, other);
  }

  /// \brief Deep copy constructor
  KOKKOS_FUNCTION constexpr Vector(const Vector<T, N, Accessor, Ownership::Views>& other)
    requires(!HasCopyConstructor<Accessor>) &&
            HasNonConstAccessOperator<Accessor, T> && (!std::is_same_v<Accessor, Vector<T, N, Accessor, Ownership::Views>>)
      : accessor_() {
    impl::deep_copy_impl(std::make_index_sequence<N>{}, *this, other);
  }

  /// \brief Deep copy constructor
  template <typename OtherAccessor>
  KOKKOS_FUNCTION constexpr Vector(const Vector<T, N, OtherAccessor, Ownership::Owns>& other)
    requires(!std::is_same_v<Accessor, OtherAccessor>) && HasNonConstAccessOperator<Accessor, T>
      : accessor_() {
    impl::deep_copy_impl(std::make_index_sequence<N>{}, *this, other);
  }

  /// \brief Deep copy constructor
  template <typename OtherAccessor>
  KOKKOS_FUNCTION constexpr Vector(const Vector<T, N, OtherAccessor, Ownership::Views>& other)
    requires(!std::is_same_v<Accessor, OtherAccessor>) && HasNonConstAccessOperator<Accessor, T>
      : accessor_() {
    impl::deep_copy_impl(std::make_index_sequence<N>{}, *this, other);
  }

  /// \brief Deep move constructor
  KOKKOS_FUNCTION constexpr Vector(Vector<T, N, Accessor, Ownership::Owns>&& other)
    requires(HasCopyConstructor<Accessor> || HasMoveConstructor<Accessor>)
      : accessor_(std::move(other.accessor_)) {
  }

  /// \brief Deep move constructor
  template <typename OtherAccessor>
    requires(!std::is_same_v<Accessor, OtherAccessor>) && HasNonConstAccessOperator<Accessor, T>
  KOKKOS_FUNCTION constexpr Vector(Vector<T, N, OtherAccessor, Ownership::Owns>&& other) : accessor_() {
    impl::deep_copy_impl(std::make_index_sequence<N>{}, *this, std::move(other));
  }
  /// \brief Deep move constructor
  template <typename OtherAccessor>
    requires(!std::is_same_v<Accessor, OtherAccessor>) && HasNonConstAccessOperator<Accessor, T>
  KOKKOS_FUNCTION constexpr Vector(Vector<T, N, OtherAccessor, Ownership::Views>&& other) : accessor_() {
    impl::deep_copy_impl(std::make_index_sequence<N>{}, *this, std::move(other));
  }

  /// \brief Deep copy assignment operator with different accessor
  /// \details Copies the data from the other vector to our data. This is only enabled if T is not const.
  template <typename OtherAccessor>
  KOKKOS_FUNCTION constexpr Vector<T, N, Accessor, Ownership::Owns>& operator=(
      const Vector<T, N, OtherAccessor, Ownership::Owns>& other)
    requires(!std::is_same_v<Accessor, OtherAccessor>) && HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(std::make_index_sequence<N>{}, *this, other);
    return *this;
  }

  /// \brief Deep copy assignment operator with same accessor
  /// \details Copies the data from the other vector to our data. This is only enabled if T is not const.
  /// Yes, this function is necessary. If we only use the version for differing accessor, the compiler can get confused.
  KOKKOS_FUNCTION constexpr Vector<T, N, Accessor, Ownership::Owns>& operator=(
      const Vector<T, N, Accessor, Ownership::Owns>& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(std::make_index_sequence<N>{}, *this, other);
    return *this;
  }

  /// \brief Deep copy assignment operator with different accessor
  /// \details Copies the data from the other vector to our data. This is only enabled if T is not const.
  template <typename OtherAccessor>
  KOKKOS_FUNCTION constexpr Vector<T, N, Accessor, Ownership::Owns>& operator=(
      const Vector<T, N, OtherAccessor, Ownership::Views>& other)
    requires(!std::is_same_v<Accessor, OtherAccessor>) && HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(std::make_index_sequence<N>{}, *this, other);
    return *this;
  }

  /// \brief Deep copy assignment operator with same accessor
  /// \details Copies the data from the other vector to our data. This is only enabled if T is not const.
  /// Yes, this function is necessary. If we only use the version for differing accessor, the compiler can get confused.
  KOKKOS_FUNCTION constexpr Vector<T, N, Accessor, Ownership::Owns>& operator=(
      const Vector<T, N, Accessor, Ownership::Views>& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(std::make_index_sequence<N>{}, *this, other);
    return *this;
  }

  /// \brief Deep copy assignment operator from a single value
  /// \param[in] value The value to set all elements to.
  KOKKOS_FUNCTION constexpr Vector<T, N, Accessor, Ownership::Owns>& operator=(const T value)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::fill_impl(std::make_index_sequence<N>{}, *this, value);
    return *this;
  }

  /// \brief Move assignment operator with different accessor.
  /// \details Moves the data from the other vector to our data. This is only enabled if T is not const.
  template <typename OtherAccessor>
  KOKKOS_FUNCTION constexpr Vector<T, N, Accessor, Ownership::Owns>& operator=(
      Vector<T, N, OtherAccessor, Ownership::Owns>&& other)
    requires(!std::is_same_v<Accessor, OtherAccessor>) && HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(std::make_index_sequence<N>{}, *this, std::move(other));
    return *this;
  }

  /// \brief Move assignment operator with same accessor
  /// \details Moves the data from the other vector to our data. This is only enabled if T is not const.
  /// Yes, this function is necessary. If we only use the version for differing accessor, the compiler can get confused.
  KOKKOS_FUNCTION constexpr Vector<T, N, Accessor, Ownership::Owns>& operator=(
      Vector<T, N, Accessor, Ownership::Owns>&& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(std::make_index_sequence<N>{}, *this, std::move(other));
    return *this;
  }

  /// \brief Move assignment operator with different accessor.
  /// Same as deep copy since a other's data is not owned.
  /// \details Moves the data from the other vector to our data. This is only enabled if T is not const.
  template <typename OtherAccessor>
  KOKKOS_FUNCTION constexpr Vector<T, N, Accessor, Ownership::Owns>& operator=(
      Vector<T, N, OtherAccessor, Ownership::Views>&& other)
    requires(!std::is_same_v<Accessor, OtherAccessor>) && HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(std::make_index_sequence<N>{}, *this, std::move(other));
    return *this;
  }

  /// \brief Move assignment operator with same accessor
  /// Same as deep copy since a other's data is not owned.
  /// \details Moves the data from the other vector to our data. This is only enabled if T is not const.
  /// Yes, this function is necessary. If we only use the version for differing accessor, the compiler can get confused.
  KOKKOS_FUNCTION constexpr Vector<T, N, Accessor, Ownership::Owns>& operator=(
      Vector<T, N, Accessor, Ownership::Views>&& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(std::make_index_sequence<N>{}, *this, std::move(other));
    return *this;
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Element access operator via a single index
  /// \param[in] index The index of the element.
  KOKKOS_FUNCTION
  constexpr T& operator[](size_t index) {
    return accessor_[index];
  }

  /// \brief Const element access operator via a single index
  /// \param[in] index The index of the element.
  KOKKOS_FUNCTION
  constexpr const T& operator[](size_t index) const {
    return accessor_[index];
  }

  /// \brief Element access operator via a single index
  /// \param[in] index The index of the element.
  KOKKOS_FUNCTION
  constexpr T& operator()(size_t index) {
    return accessor_[index];
  }

  /// \brief Const element access operator via a single index
  /// \param[in] index The index of the element.
  KOKKOS_FUNCTION
  constexpr const T& operator()(size_t index) const {
    return accessor_[index];
  }

  /// \brief Get the internal data accessor
  KOKKOS_FUNCTION
  constexpr Accessor& data() {
    return accessor_;
  }

  /// \brief Get the internal data accessor
  KOKKOS_FUNCTION
  constexpr const Accessor& data() const {
    return accessor_;
  }
  //@}

  //! \name Setters and modifiers
  //@{

  /// \brief Set all elements of the vector
  template <typename... Args>
    requires(sizeof...(Args) == N) && (std::is_convertible_v<Args, T> && ...) && HasNonConstAccessOperator<Accessor, T>
  KOKKOS_FUNCTION constexpr void set(Args&&... args) {
    impl::set_from_args_impl(std::make_index_sequence<N>{}, *this, static_cast<T>(std::forward<Args>(args))...);
  }

  /// \brief Set all elements of the vector using an accessor
  /// \param[in] accessor A valid accessor.
  /// \note A Vector is also a valid accessor.
  template <ValidAccessor<T> OtherAccessor>
  KOKKOS_FUNCTION constexpr void set(const OtherAccessor& accessor)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::set_from_accessor_impl(std::make_index_sequence<N>{}, *this, accessor);
  }

  /// \brief Set all elements of the vector to a single value
  /// \param[in] value The value to set all elements to.
  KOKKOS_FUNCTION
  constexpr void fill(const T& value)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::fill_impl(std::make_index_sequence<N>{}, *this, value);
  }
  //@}

  //! \name Unary operators
  //@{

  /// \brief Unary plus operator
  KOKKOS_FUNCTION
  constexpr Vector<T, N> operator+() const {
    return *this;
  }

  /// \brief Unary minus operator
  KOKKOS_FUNCTION
  constexpr Vector<T, N> operator-() const {
    return impl::unary_minus_impl(std::make_index_sequence<N>{}, *this);
  }
  //@}

  //! \name Addition and subtraction
  //@{

  /// \brief Vector-vector addition
  /// \param[in] other The other vector.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_FUNCTION constexpr auto operator+(const Vector<U, N, OtherAccessor, OtherOwnershipType>& other) const {
    return impl::vector_vector_add_impl(std::make_index_sequence<N>{}, *this, other);
  }

  /// \brief Vector-vector addition
  /// \param[in] other The other vector.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_FUNCTION constexpr Vector<T, N, Accessor, Ownership::Owns>& operator+=(
      const Vector<U, N, OtherAccessor, OtherOwnershipType>& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_vector_add_impl(std::make_index_sequence<N>{}, *this, other);
    return *this;
  }

  /// \brief Vector-vector subtraction
  /// \param[in] other The other vector.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_FUNCTION constexpr auto operator-(const Vector<U, N, OtherAccessor, OtherOwnershipType>& other) const {
    return impl::vector_vector_subtraction_impl(std::make_index_sequence<N>{}, *this, other);
  }

  /// \brief Self-vector subtraction
  /// \param[in] other The other vector.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_FUNCTION constexpr Vector<T, N, Accessor, Ownership::Owns>& operator-=(
      const Vector<U, N, OtherAccessor, OtherOwnershipType>& other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_vector_subtraction_impl(std::make_index_sequence<N>{}, *this, other);
    return *this;
  }

  /// \brief Vector-scalar addition
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_FUNCTION constexpr auto operator+(const U& scalar) const {
    return impl::vector_scalar_add_impl(std::make_index_sequence<N>{}, *this, scalar);
  }

  /// \brief Self-scalar addition
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_FUNCTION constexpr Vector<T, N, Accessor, Ownership::Owns>& operator+=(const U& scalar)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_scalar_add_impl(std::make_index_sequence<N>{}, *this, scalar);
    return *this;
  }

  /// \brief Vector-scalar subtraction
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_FUNCTION constexpr auto operator-(const U& scalar) const {
    return impl::vector_scalar_subtraction_impl(std::make_index_sequence<N>{}, *this, scalar);
  }

  /// \brief Vector-scalar subtraction
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_FUNCTION constexpr Vector<T, N, Accessor, Ownership::Owns>& operator-=(const U& scalar)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_scalar_subtraction_impl(std::make_index_sequence<N>{}, *this, scalar);
    return *this;
  }
  //@}

  //! \name Multiplication and division
  //@{

  /// \brief Vector-scalar multiplication
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_FUNCTION constexpr auto operator*(const U& scalar) const {
    return impl::vector_scalar_multiplication_impl(std::make_index_sequence<N>{}, *this, scalar);
  }

  /// \brief Self-scalar multiplication
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_FUNCTION constexpr Vector<T, N, Accessor, Ownership::Owns>& operator*=(const U& scalar)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_scalar_multiplication_impl(std::make_index_sequence<N>{}, *this, scalar);
    return *this;
  }

  /// \brief Vector-scalar division. (Type promotes the result to a double if the scalar is not a floating point.)
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_FUNCTION constexpr auto operator/(const U& scalar) const {
    return impl::vector_scalar_division_impl(std::make_index_sequence<N>{}, *this, scalar);
  }

  /// \brief Self-scalar division (Does not type-promote the result!!).
  /// \note Because there is no type promotion, this will perform integer division if the scalar is an integer.
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_FUNCTION constexpr Vector<T, N, Accessor, Ownership::Owns>& operator/=(const U& scalar)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_scalar_division_impl(std::make_index_sequence<N>{}, *this, scalar);
    return *this;
  }
  //@}

  //! \name Static methods
  //@{

  /// \brief Get a vector of ones
  KOKKOS_FUNCTION static constexpr Vector<T, N> ones() {
    return ones_impl(std::make_index_sequence<N>{});
  }

  /// \brief Get the zero vector
  KOKKOS_FUNCTION static constexpr Vector<T, N> zeros() {
    return zeros_impl(std::make_index_sequence<N>{});
  }
  //@}

  //! \name Friends <3
  //@{

  // Declare the << operator as a friend
  template <typename U, size_t M, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  friend std::ostream& operator<<(std::ostream& os, const Vector<U, M, OtherAccessor, OtherOwnershipType>& vec);

  // We are friends with all Vectors regardless of their Accessor, type, or ownership
  template <typename U, size_t M, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
    requires std::is_arithmetic_v<U>
  friend class Vector;
  //@}

 private:
  //! \name Private helper functions
  //@{

  /// \brief Get a vector of ones
  template <size_t... Is>
  KOKKOS_FUNCTION static constexpr Vector<T, N> ones_impl(std::index_sequence<Is...>) {
    Vector<std::remove_const_t<T>, N> result;
    ((result[Is] = static_cast<T>(1)), ...);
    return result;
  }

  /// \brief Get a vector of zeros
  template <size_t... Is>
  KOKKOS_FUNCTION static constexpr Vector<T, N> zeros_impl(std::index_sequence<Is...>) {
    Vector<std::remove_const_t<T>, N> result;
    ((result[Is] = static_cast<T>(0)), ...);
    return result;
  }
  //@}
};  // class Vector

template <typename T, size_t N, ValidAccessor<T> Accessor = Array<T, N>>
  requires std::is_arithmetic_v<T>
using VectorView = Vector<T, N, Accessor, Ownership::Views>;

template <typename T, size_t N, ValidAccessor<T> Accessor = Array<T, N>>
  requires std::is_arithmetic_v<T>
using OwningVector = Vector<T, N, Accessor, Ownership::Owns>;

static_assert(is_vector_v<Vector<int, 3>>, "Odd, default Vector is not a vector.");
static_assert(is_vector_v<Vector<int, 3, Array<int, 3>>>, "Odd, default vector with Array accessor is not a vector.");
static_assert(is_vector_v<VectorView<int, 3>>, "Odd, VectorView is not a vector.");
static_assert(is_vector_v<OwningVector<int, 3>>, "Odd, OwningVector is not a vector.");

//! \name Non-member functions
//@{

//! \name Write to output stream
//@{

/// \brief Write the vector to an output stream
/// \param[in] os The output stream.
/// \param[in] vec The vector.
template <typename T, size_t N, ValidAccessor<T> Accessor, typename OwnershipType>
std::ostream& operator<<(std::ostream& os, const Vector<T, N, Accessor, OwnershipType>& vec) {
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
KOKKOS_FUNCTION constexpr bool is_close(
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
KOKKOS_FUNCTION constexpr bool is_approx_close(
    const U& scalar1, const T& scalar2,
    const decltype(get_relaxed_comparison_tolerance<T, U>())& tol = get_relaxed_comparison_tolerance<T, U>()) {
  return is_close(scalar1, scalar2, tol);
}

/// \brief Vector-vector equality (element-wise within a tolerance)
/// \param[in] vec1 The first vector.
/// \param[in] vec2 The second vector.
/// \param[in] tol The tolerance (default is determined by the given type).
template <size_t N, typename U, typename T, ValidAccessor<U> Accessor1, typename Ownership1, ValidAccessor<T> Accessor2,
          typename Ownership2>
KOKKOS_FUNCTION constexpr bool is_close(
    const Vector<U, N, Accessor1, Ownership1>& vec1, const Vector<T, N, Accessor2, Ownership2>& vec2,
    const decltype(get_comparison_tolerance<T, U>())& tol = get_comparison_tolerance<T, U>()) {
  return impl::is_close_impl(std::make_index_sequence<N>{}, vec1, vec2, tol);
}

/// \brief Vector-vector equality (element-wise within a relaxed tolerance)
/// \param[in] vec1 The first vector.
/// \param[in] vec2 The second vector.
/// \param[in] tol The tolerance (default is determined by the given type).
template <size_t N, typename U, typename T, ValidAccessor<U> Accessor1, typename Ownership1, ValidAccessor<T> Accessor2,
          typename Ownership2>
KOKKOS_FUNCTION constexpr bool is_approx_close(
    const Vector<U, N, Accessor1, Ownership1>& vec1, const Vector<T, N, Accessor2, Ownership2>& vec2,
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
KOKKOS_FUNCTION constexpr auto operator+(const U& scalar, const Vector<T, N, Accessor, OwnershipType>& vec)
    -> Vector<std::common_type_t<T, U>, N> {
  return vec + scalar;
}

/// \brief Scalar-vector subtraction
/// \param[in] scalar The scalar.
/// \param[in] vec The vector.
template <size_t N, typename U, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_FUNCTION constexpr auto operator-(const U& scalar, const Vector<T, N, Accessor, OwnershipType>& vec)
    -> Vector<std::common_type_t<T, U>, N> {
  return -vec + scalar;
}
//@}

//! \name Non-member multiplication and division operators
//@{

/// \brief Scalar-vector multiplication
/// \param[in] scalar The scalar.
/// \param[in] vec The vector.
template <size_t N, typename U, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_FUNCTION constexpr auto operator*(const U& scalar, const Vector<T, N, Accessor, OwnershipType>& vec)
    -> Vector<std::common_type_t<T, U>, N> {
  return vec * scalar;
}
//@}

//! \name Basic arithmetic reduction operations
//@{

/// \brief Sum of all elements
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_FUNCTION constexpr auto sum(const Vector<T, N, Accessor, OwnershipType>& vec) {
  return impl::sum_impl(std::make_index_sequence<N>{}, vec);
}

/// \brief Product of all elements
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_FUNCTION constexpr auto product(const Vector<T, N, Accessor, OwnershipType>& vec) {
  return impl::product_impl(std::make_index_sequence<N>{}, vec);
}

/// \brief Minimum element
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_FUNCTION constexpr auto min(const Vector<T, N, Accessor, OwnershipType>& vec) {
  return impl::min_impl(std::make_index_sequence<N>{}, vec);
}

/// \brief Maximum element
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_FUNCTION constexpr auto max(const Vector<T, N, Accessor, OwnershipType>& vec) {
  return impl::max_impl(std::make_index_sequence<N>{}, vec);
}

/// \brief Mean of all elements (returns a double if T is an integral type, otherwise returns T)
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, double, T>>
KOKKOS_FUNCTION constexpr OutputType mean(const Vector<T, N, Accessor, OwnershipType>& vec) {
  auto vec_sum = sum(vec);
  return static_cast<OutputType>(vec_sum) / OutputType(N);
}

/// \brief Mean of all elements (returns a float if T is an integral type, otherwise returns T)
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, float, T>>
KOKKOS_FUNCTION constexpr OutputType mean_f(const Vector<T, N, Accessor, OwnershipType>& vec) {
  return mean(vec);
}

/// \brief Variance of all elements (returns a double if T is an integral type, otherwise returns T)
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, double, T>>
KOKKOS_FUNCTION constexpr OutputType variance(const Vector<T, N, Accessor, OwnershipType>& vec) {
  return impl::variance_impl(std::make_index_sequence<N>{}, vec);
}

/// \brief Variance of all elements (returns a float if T is an integral type, otherwise returns T)
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, float, T>>
KOKKOS_FUNCTION constexpr OutputType variance_f(const Vector<T, N, Accessor, OwnershipType>& vec) {
  return variance(vec);
}

/// \brief Standard deviation of all elements (returns a double if T is an integral type, otherwise returns T)
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, double, T>>
KOKKOS_FUNCTION constexpr OutputType stddev(const Vector<T, N, Accessor, OwnershipType>& vec) {
  return impl::standard_deviation_impl(std::make_index_sequence<N>{}, vec);
}

/// \brief Standard deviation of all elements (returns a float if T is an integral type, otherwise returns T)
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, float, T>>
KOKKOS_FUNCTION constexpr OutputType stddev_f(const Vector<T, N, Accessor, OwnershipType>& vec) {
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
KOKKOS_FUNCTION constexpr auto dot(const Vector<U, N, Accessor1, Ownership1>& a,
                                   const Vector<T, N, Accessor2, Ownership2>& b) -> std::common_type_t<T, U> {
  return impl::dot_product_impl(std::make_index_sequence<N>{}, a, b);
}
//@}

//! \name Vector norms
//@{

/// \brief Vector infinity norm
/// \param[in] vec The vector.
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_FUNCTION constexpr auto infinity_norm(const Vector<T, N, Accessor, OwnershipType>& vec) {
  return max(vec);
}

/// \brief Vector 1-norm
/// \param[in] vec The vector.
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_FUNCTION constexpr auto one_norm(const Vector<T, N, Accessor, OwnershipType>& vec) {
  return sum(vec);
}

/// \brief Vector 2-norm (Returns a double if T is an integral type, otherwise returns T)
/// \param[in] vec The vector.
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, double, T>>
KOKKOS_FUNCTION constexpr OutputType two_norm(const Vector<T, N, Accessor, OwnershipType>& vec) {
  return std::sqrt(static_cast<OutputType>(dot(vec, vec)));
}

/// \brief Vector 2-norm (Returns a float if T is an integral type, otherwise returns T)
/// \param[in] vec The vector.
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, float, T>>
KOKKOS_FUNCTION constexpr OutputType two_norm_f(const Vector<T, N, Accessor, OwnershipType>& vec) {
  return two_norm(vec);
}

/// \brief Vector squared 2-norm
/// \param[in] vec The vector.
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_FUNCTION constexpr auto two_norm_squared(const Vector<T, N, Accessor, OwnershipType>& vec) {
  return dot(vec, vec);
}

/// \brief Default vector norm (2-norm, returns a double if T is an integral type, otherwise returns T)
/// \param[in] vec The vector.
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, double, T>>
KOKKOS_FUNCTION constexpr OutputType norm(const Vector<T, N, Accessor, OwnershipType>& vec) {
  return two_norm(vec);
}

/// \brief Default vector norm (2-norm, returns a float if T is an integral type, otherwise returns T)
/// \param[in] vec The vector.
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType,
          typename OutputType = std::conditional_t<std::is_integral_v<T>, float, T>>
KOKKOS_FUNCTION constexpr OutputType norm_f(const Vector<T, N, Accessor, OwnershipType>& vec) {
  return norm(vec);
}

/// \brief Default vector norm squared (2-norm)
/// \param[in] vec The vector.
template <size_t N, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_FUNCTION constexpr auto norm_squared(const Vector<T, N, Accessor, OwnershipType>& vec) {
  return two_norm_squared(vec);
}

/// \brief Minor angle between two vectors (returns a double if T is an integral type, otherwise returns T)
/// \param[in] a The first vector.
/// \param[in] b The second vector.
template <size_t N, typename U, typename T, ValidAccessor<U> Accessor1, typename Ownership1, ValidAccessor<T> Accessor2,
          typename Ownership2, typename OutputType = std::conditional_t<std::is_integral_v<T>, double, T>>
KOKKOS_FUNCTION constexpr OutputType minor_angle(const Vector<U, N, Accessor1, Ownership1>& a,
                                                 const Vector<T, N, Accessor2, Ownership2>& b) {
  return std::acos(static_cast<OutputType>(dot(a, b)) /
                   (static_cast<OutputType>(two_norm(a)) * static_cast<OutputType>(two_norm(b))));
}

/// \brief Minor angle between two vectors (returns a float if T is an integral type, otherwise returns T)
/// \param[in] a The first vector.
/// \param[in] b The second vector.
template <size_t N, typename U, typename T, ValidAccessor<U> Accessor1, typename Ownership1, ValidAccessor<T> Accessor2,
          typename Ownership2, typename OutputType = std::conditional_t<std::is_integral_v<T>, float, T>>
KOKKOS_FUNCTION constexpr OutputType minor_angle_f(const Vector<U, N, Accessor1, Ownership1>& a,
                                                   const Vector<T, N, Accessor2, Ownership2>& b) {
  return minor_angle(a, b);
}

/// \brief Major angle between two vectors (returns a double if T is an integral type, otherwise returns T)
/// \param[in] a The first vector.
/// \param[in] b The second vector.
template <size_t N, typename U, typename T, ValidAccessor<U> Accessor1, typename Ownership1, ValidAccessor<T> Accessor2,
          typename Ownership2, typename OutputType = std::conditional_t<std::is_integral_v<T>, double, T>>
KOKKOS_FUNCTION constexpr OutputType major_angle(const Vector<U, N, Accessor1, Ownership1>& a,
                                                 const Vector<T, N, Accessor2, Ownership2>& b) {
  return OutputType(M_PI) - minor_angle(a, b);
}

/// \brief Major angle between two vectors (returns a float if T is an integral type, otherwise returns T)
/// \param[in] a The first vector.
/// \param[in] b The second vector.
template <size_t N, typename U, typename T, ValidAccessor<U> Accessor1, typename Ownership1, ValidAccessor<T> Accessor2,
          typename Ownership2, typename OutputType = std::conditional_t<std::is_integral_v<T>, float, T>>
KOKKOS_FUNCTION constexpr OutputType major_angle_f(const Vector<U, N, Accessor1, Ownership1>& a,
                                                   const Vector<T, N, Accessor2, Ownership2>& b) {
  return major_angle(a, b);
}
//@}

//! \name Vector views
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
KOKKOS_FUNCTION constexpr auto get_vector_view(Accessor& data) {
  return VectorView<T, N, Accessor>(data);
}

template <typename T, size_t N, ValidAccessor<T> Accessor>
KOKKOS_FUNCTION constexpr auto get_vector_view(Accessor&& data) {
  return VectorView<T, N, Accessor>(std::forward<Accessor>(data));
}

template <typename T, size_t N, ValidAccessor<T> Accessor>
KOKKOS_FUNCTION constexpr auto get_owning_vector(Accessor& data) {
  return OwningVector<T, N, Accessor>(data);
}

template <typename T, size_t N, ValidAccessor<T> Accessor>
KOKKOS_FUNCTION constexpr auto get_owning_vector(Accessor&& data) {
  return OwningVector<T, N, Accessor>(std::forward<Accessor>(data));
}
//@}

//@}

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_VECTOR_HPP_
