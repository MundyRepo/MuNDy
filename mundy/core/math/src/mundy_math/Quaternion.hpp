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

#ifndef MUNDY_MATH_QUATERNION_HPP_
#define MUNDY_MATH_QUATERNION_HPP_

// External
#include <Kokkos_Core.hpp>

// C++ core
#include <type_traits>  // for std::decay_t
#include <utility>

// Mundy
#include <mundy_math/Accessor.hpp>              // for mundy::ValidAccessor
#include <mundy_math/Array.hpp>                 // for mundy::Array
#include <mundy_math/Matrix3.hpp>               // for mundy::Matrix3
#include <mundy_math/NumTraits.hpp>             // for mundy::ValidScalarType, mundy::NumTraits
#include <mundy_math/ScalarBinaryOpTraits.hpp>  // for mundy::scalar_*_result_t
#include <mundy_math/Tolerance.hpp>             // for mundy::get_zero_tolerance
#include <mundy_math/Vector3.hpp>               // for mundy::Vector3
#include <mundy_math/cmath.hpp>
#include <mundy_math/impl/QuaternionImpl.hpp>
#include <mundy_utils/requires.hpp>
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_ASSERT

namespace mundy {

/// \brief (Implementation) Type trait to determine if a type is an AQuaternion
template <typename TypeToCheck>
struct is_quaternion_impl : std::false_type {};
//
template <typename T, typename Accessor>
struct is_quaternion_impl<AQuaternion<T, Accessor>> : std::true_type {};

/// \brief Type trait to determine if a type is an AQuaternion
template <typename T>
struct is_quaternion : is_quaternion_impl<std::decay_t<T>> {};
//
template <typename TypeToCheck>
constexpr bool is_quaternion_v = is_quaternion<TypeToCheck>::value;

/// \brief A temporary concept to check if a type is a valid AQuaternion type
/// TODO(palmerb4): Extend this concept to contain all shared setters and getters for our quaternions.
template <typename QuaternionType>
concept ValidQuaternionType =
    is_quaternion_v<std::decay_t<QuaternionType>> &&
    requires(std::decay_t<QuaternionType> quaternion, const std::decay_t<QuaternionType> const_quaternion) {
      typename std::decay_t<QuaternionType>::value_type;
      { quaternion[0] } -> std::convertible_to<typename std::decay_t<QuaternionType>::value_type>;
      { quaternion[1] } -> std::convertible_to<typename std::decay_t<QuaternionType>::value_type>;
      { quaternion[2] } -> std::convertible_to<typename std::decay_t<QuaternionType>::value_type>;
      { quaternion[3] } -> std::convertible_to<typename std::decay_t<QuaternionType>::value_type>;

      { quaternion(0) } -> std::convertible_to<typename std::decay_t<QuaternionType>::value_type>;
      { quaternion(1) } -> std::convertible_to<typename std::decay_t<QuaternionType>::value_type>;
      { quaternion(2) } -> std::convertible_to<typename std::decay_t<QuaternionType>::value_type>;
      { quaternion(3) } -> std::convertible_to<typename std::decay_t<QuaternionType>::value_type>;

      { const_quaternion[0] } -> std::convertible_to<const typename std::decay_t<QuaternionType>::value_type>;
      { const_quaternion[1] } -> std::convertible_to<const typename std::decay_t<QuaternionType>::value_type>;
      { const_quaternion[2] } -> std::convertible_to<const typename std::decay_t<QuaternionType>::value_type>;
      { const_quaternion[3] } -> std::convertible_to<const typename std::decay_t<QuaternionType>::value_type>;

      { const_quaternion(0) } -> std::convertible_to<const typename std::decay_t<QuaternionType>::value_type>;
      { const_quaternion(1) } -> std::convertible_to<const typename std::decay_t<QuaternionType>::value_type>;
      { const_quaternion(2) } -> std::convertible_to<const typename std::decay_t<QuaternionType>::value_type>;
      { const_quaternion(3) } -> std::convertible_to<const typename std::decay_t<QuaternionType>::value_type>;
    };  // ValidQuaternionType

//! \name Forward declare AQuaternion functions that also require AQuaternion to be defined
//@{

/// \brief Get the norm of a quaternion
/// \param[in] quat The quaternion.
template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto norm(const AQuaternion<T, Accessor>& quat);
//@}

/// \brief AQuaternion class with floating point entries (an integer-valued quaternion doesn't make much sense)
/// \tparam T The type of the entries.
/// \tparam Accessor The type of the accessor.
///
/// This class is designed to be used with Kokkos. It is a simple quaternion with arithmetic entries. It is templated
/// on the type of the entries and Accessor type. See Accessor.hpp for more details on the Accessor type requirements.
///
/// The goal of AQuaternion is to be a lightweight class that can be used with Kokkos to perform mathematical operations
/// on vectors in R3. It does not own the data, but rather it is templated on an Accessor type that provides access to
/// the underlying data. This allows us to use AQuaternion with Kokkos Views, raw pointers, or any other type that meets
/// the ValidAccessor requirements without copying the data. This is especially important for GPU-compatible code.
/// The underlying data is stored in Eigen's coefficient order `(x, y, z, w)`, even though the semantic quaternion
/// constructor and setters accept values in `(w, x, y, z)` order.
///
/// Quaternions can be constructed by passing an accessor to the constructor. However, if the accessor has a 4-argument
/// constructor, then the AQuaternion can also be constructed by passing the elements directly to the constructor.
/// Similarly, if the accessor has an initializer list constructor, then the AQuaternion can be constructed by passing
/// an initializer list to the constructor. This is a convenience feature which makes working with the default accessor
/// (Array<T, 4>) easier. For example, the following are all valid ways to construct an AQuaternion:
///
/// \code{.cpp}
///   // Constructs an AQuaternion with the default accessor (Array<double, 4>)
///   AQuaternion<double> quat1({1.0, 2.0, 3.0, 4.0});
///   AQuaternion<double> quat2(1.0, 2.0, 3.0, 4.0);
///   AQuaternion<double> quat3(Array<double, 4>({2.0, 3.0, 4.0, 1.0}));
///   AQuaternion<double> quat4;
///   quat4.set(1.0, 2.0, 3.0, 4.0);
///
///   // Construct an AQuaternion view from raw Eigen-style coefficient storage
///   double data[4] = {2.0, 3.0, 4.0, 1.0};
///   AQuaternion<double, double*> quat5(data);
///   AQuaternion<double, double*> quat6{1.0, 2.0, 3.0, 4.0};
///   // Not allowed as double* doesn't have a 4-argument constructor
///   // AQuaternion<double, double*> quat7(1.0, 2.0, 3.0, 4.0);
/// \endcode
///
/// \note Accessors may be owning or non-owning, that is irrelevant to the AQuaternion class; however, these accessors
/// should be lightweight such that they can be copied around without much overhead. Furthermore, the lifetime of the
/// data underlying the accessor should be as long as the AQuaternion that use it.
template <typename T, ValidAccessor<T> Accessor>
MUNDY_REQUIRES(ValidScalarType<T> && !NumTraits<T>::IsInteger)
class AQuaternion {
 public:
  //! \name Internal data
  //@{

  static constexpr size_t x_storage_index = 0;
  static constexpr size_t y_storage_index = 1;
  static constexpr size_t z_storage_index = 2;
  static constexpr size_t w_storage_index = 3;

  /// \brief Stored accessor via storage.
  storage<Accessor> accessor_;
  //@}

  //! \name Type aliases
  //@{

  /// \brief The type of the entries
  using value_type = T;

  /// \brief The non-const type of the entries
  using non_const_value_type = std::remove_const_t<T>;

  /// \brief Deep copy type
  using deep_copy_t = AQuaternion<T>;
  //@}

 private:
  KOKKOS_INLINE_FUNCTION
  static constexpr Accessor make_storage_from_semantic_components(const T& w, const T& x, const T& y, const T& z)
      MUNDY_REQUIRES(HasNArgConstructor<Accessor, T, 4>) {
    return Accessor(x, y, z, w);
  }

 public:
  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor. Assume elements are uninitialized.
  /// \note This constructor is only enabled if the Accessor has a default constructor.
  KOKKOS_INLINE_FUNCTION constexpr AQuaternion() MUNDY_REQUIRES(HasDefaultConstructor<Accessor>) : accessor_() {
  }

  /// \brief Constructor from a given accessor
  /// \param[in] data The accessor.
  KOKKOS_INLINE_FUNCTION
  explicit constexpr AQuaternion(const Accessor& data) MUNDY_REQUIRES(std::is_copy_constructible_v<Accessor>)
      : accessor_(data) {
  }

  /// \brief Constructor to initialize all elements
  /// \param[in] w The scalar component.
  /// \param[in] x The x component.
  /// \param[in] y The y component.
  /// \param[in] z The z component.
  /// \note This constructor is only enabled if the Accessor has a 4-argument constructor.
  /// \note The underlying storage order is `(x, y, z, w)`.
  KOKKOS_INLINE_FUNCTION constexpr AQuaternion(const T& w, const T& x, const T& y, const T& z)
      MUNDY_REQUIRES(HasNArgConstructor<Accessor, T, 4>)
      : accessor_(make_storage_from_semantic_components(w, x, y, z)) {
  }

  /// \brief Destructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr ~AQuaternion() = default;

  // Default copy/move constructors and assignment operators when interacting with an AQuaternion of the same type

  /// \brief Default copy constructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr AQuaternion(const AQuaternion<T, Accessor>&) = default;

  /// \brief Default move constructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr AQuaternion(AQuaternion<T, Accessor>&&) = default;

  /// \brief Default copy assignment operator
  KOKKOS_DEFAULTED_FUNCTION
  constexpr AQuaternion<T, Accessor>& operator=(const AQuaternion<T, Accessor>&) = default;

  /// \brief Default move assignment operator
  KOKKOS_DEFAULTED_FUNCTION
  constexpr AQuaternion<T, Accessor>& operator=(AQuaternion<T, Accessor>&&) = default;

  // Custom copy/move constructors and assignment operators when interacting with an AQuaternion of a different type

  /// \brief Deep copy constructor with different accessor
  template <ValidQuaternionType OtherQuaternionType>
      KOKKOS_INLINE_FUNCTION constexpr AQuaternion(const OtherQuaternionType& other)
          MUNDY_REQUIRES(!std::is_same_v<OtherQuaternionType, AQuaternion<T, Accessor>>) &&
      (std::is_convertible_v<typename OtherQuaternionType::value_type, T>) : accessor_() {
    impl::deep_copy_impl(*this, other);
  }

  /// \brief Deep move constructor with different accessor
  template <ValidQuaternionType OtherQuaternionType>
      KOKKOS_INLINE_FUNCTION constexpr AQuaternion(OtherQuaternionType&& other)
          MUNDY_REQUIRES(!std::is_same_v<OtherQuaternionType, AQuaternion<T, Accessor>>) &&
      (std::is_convertible_v<typename OtherQuaternionType::value_type, T>) : accessor_() {
    impl::deep_copy_impl(*this, std::move(other));
  }

  /// \brief Deep copy assignment operator with different accessor
  /// \details Copies the data from the other vector to our data. This is only enabled if T is not const.
  template <ValidQuaternionType OtherQuaternionType>
  KOKKOS_INLINE_FUNCTION constexpr AQuaternion<T, Accessor>& operator=(const OtherQuaternionType& other)
      MUNDY_REQUIRES((!std::is_same_v<OtherQuaternionType, AQuaternion<T, Accessor>>) &&
                     (std::is_convertible_v<typename OtherQuaternionType::value_type, T>) &&
                     HasNonConstAccessOperator<Accessor, T>) {
    impl::deep_copy_impl(*this, other);
    return *this;
  }

  /// \brief Deep move assignment operator with different accessor
  /// \details Moves the data from the other vector to our data. This is only enabled if T is not const.
  template <ValidQuaternionType OtherQuaternionType>
  KOKKOS_INLINE_FUNCTION constexpr AQuaternion<T, Accessor>& operator=(OtherQuaternionType&& other)
      MUNDY_REQUIRES((!std::is_same_v<OtherQuaternionType, AQuaternion<T, Accessor>>) &&
                     (std::is_convertible_v<typename OtherQuaternionType::value_type, T>) &&
                     HasNonConstAccessOperator<Accessor, T>) {
    impl::deep_copy_impl(*this, std::move(other));
    return *this;
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Element access operator via a single index
  /// \param[in] index The index of the element.
  /// \note Indices follow the raw storage order `(x, y, z, w)`.
  KOKKOS_INLINE_FUNCTION
  constexpr T& operator[](size_t index) {
    MUNDY_THROW_ASSERT(index < 4, std::out_of_range, "AQuaternion index out of bounds.");
    return impl::access_at(accessor_, index);
  }

  /// \brief Const element access operator via a single index
  /// \param[in] index The index of the element.
  /// \note Indices follow the raw storage order `(x, y, z, w)`.
  KOKKOS_INLINE_FUNCTION
  constexpr const T& operator[](size_t index) const {
    MUNDY_THROW_ASSERT(index < 4, std::out_of_range, "AQuaternion index out of bounds.");
    return impl::access_at(accessor_, index);
  }

  /// \brief Element access operator via a single index
  /// \param[in] index The index of the element.
  /// \note Indices follow the raw storage order `(x, y, z, w)`.
  KOKKOS_INLINE_FUNCTION
  constexpr T& operator()(size_t index) {
    MUNDY_THROW_ASSERT(index < 4, std::out_of_range, "AQuaternion index out of bounds.");
    return impl::access_at(accessor_, index);
  }

  /// \brief Const element access operator via a single index
  /// \param[in] index The index of the element.
  /// \note Indices follow the raw storage order `(x, y, z, w)`.
  KOKKOS_INLINE_FUNCTION
  constexpr const T& operator()(size_t index) const {
    MUNDY_THROW_ASSERT(index < 4, std::out_of_range, "AQuaternion index out of bounds.");
    return impl::access_at(accessor_, index);
  }

  /// \brief Get a reference to the scalar component
  KOKKOS_INLINE_FUNCTION
  constexpr T& w() {
    return impl::access_at(accessor_, w_storage_index);
  }

  /// \brief Get a reference to the scalar component
  KOKKOS_INLINE_FUNCTION
  constexpr const T& w() const {
    return impl::access_at(accessor_, w_storage_index);
  }

  /// \brief Get a reference to the x component
  KOKKOS_INLINE_FUNCTION
  constexpr T& x() {
    return impl::access_at(accessor_, x_storage_index);
  }

  /// \brief Get a reference to the x component
  KOKKOS_INLINE_FUNCTION
  constexpr const T& x() const {
    return impl::access_at(accessor_, x_storage_index);
  }

  /// \brief Get a reference to the y component
  KOKKOS_INLINE_FUNCTION
  constexpr T& y() {
    return impl::access_at(accessor_, y_storage_index);
  }

  /// \brief Get a reference to the y component
  KOKKOS_INLINE_FUNCTION
  constexpr const T& y() const {
    return impl::access_at(accessor_, y_storage_index);
  }

  /// \brief Get a reference to the z component
  KOKKOS_INLINE_FUNCTION
  constexpr T& z() {
    return impl::access_at(accessor_, z_storage_index);
  }

  /// \brief Get a reference to the z component
  KOKKOS_INLINE_FUNCTION
  constexpr const T& z() const {
    return impl::access_at(accessor_, z_storage_index);
  }

  /// \brief Get the internal data accessor
  /// \note The returned accessor exposes the raw storage order `(x, y, z, w)`.
  KOKKOS_INLINE_FUNCTION
  constexpr decltype(auto) data() {
    return accessor_.get();
  }

  /// \brief Get the internal data accessor
  /// \note The returned accessor exposes the raw storage order `(x, y, z, w)`.
  KOKKOS_INLINE_FUNCTION
  constexpr decltype(auto) data() const {
    return accessor_.get();
  }

  /// \brief Get a view of the quaternion vector component
  KOKKOS_INLINE_FUNCTION
  constexpr const auto vector() const {
    auto shifted_accessor = get_shifted_view<T, x_storage_index>(accessor_);
    return get_owning_vector<T, 3>(std::move(shifted_accessor));
  }

  /// \brief Get a view of the quaternion vector component
  KOKKOS_INLINE_FUNCTION
  constexpr auto vector() {
    auto shifted_accessor = get_shifted_view<T, x_storage_index>(accessor_);
    return get_owning_vector<T, 3>(std::move(shifted_accessor));
  }

  /// \brief Get a deep copy of the quaternion
  KOKKOS_INLINE_FUNCTION
  constexpr deep_copy_t copy() const {
    return *this;
  }

  /// \brief Cast (and copy) the quaternion to a different type
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr auto cast() const {
    return AQuaternion<U>(static_cast<U>(w()), static_cast<U>(x()), static_cast<U>(y()), static_cast<U>(z()));
  }
  //@}

  //! \name Setters and modifiers
  //@{

  /// \brief Set all elements of the quaternion
  /// \param[in] w The scalar component.
  /// \param[in] x The x component.
  /// \param[in] y The y component.
  /// \param[in] z The z component.
  KOKKOS_INLINE_FUNCTION
  constexpr void set(const T& w, const T& x, const T& y, const T& z)
      MUNDY_REQUIRES(HasNonConstAccessOperator<Accessor, T>) {
    impl::access_at(accessor_, x_storage_index) = x;
    impl::access_at(accessor_, y_storage_index) = y;
    impl::access_at(accessor_, z_storage_index) = z;
    impl::access_at(accessor_, w_storage_index) = w;
  }

  /// \brief Set all elements of the quaternion
  /// \param[in] w The scalar component.
  /// \param[in] vec The vector component.
  KOKKOS_INLINE_FUNCTION
  constexpr void set(const T& w, const Vector3<T>& vec) MUNDY_REQUIRES(HasNonConstAccessOperator<Accessor, T>) {
    impl::access_at(accessor_, x_storage_index) = vec[0];
    impl::access_at(accessor_, y_storage_index) = vec[1];
    impl::access_at(accessor_, z_storage_index) = vec[2];
    impl::access_at(accessor_, w_storage_index) = w;
  }

  /// \brief Set all elements of the vector using an accessor
  /// \param[in] accessor A valid accessor.
  /// \note An AQuaternion is also a valid accessor.
  template <ValidAccessor<T> OtherAccessor>
  KOKKOS_INLINE_FUNCTION constexpr void set(const OtherAccessor& accessor)
      MUNDY_REQUIRES(HasNonConstAccessOperator<Accessor, T>) {
    impl::access_at(accessor_, 0) = impl::access_at(accessor, 0);
    impl::access_at(accessor_, 1) = impl::access_at(accessor, 1);
    impl::access_at(accessor_, 2) = impl::access_at(accessor, 2);
    impl::access_at(accessor_, 3) = impl::access_at(accessor, 3);
  }

  /// \brief Set the quaternion vector component
  /// \param[in] vec The vector.
  KOKKOS_INLINE_FUNCTION
  constexpr void set_vector(const Vector3<T>& vec) MUNDY_REQUIRES(HasNonConstAccessOperator<Accessor, T>) {
    impl::access_at(accessor_, x_storage_index) = vec[0];
    impl::access_at(accessor_, y_storage_index) = vec[1];
    impl::access_at(accessor_, z_storage_index) = vec[2];
  }

  /// \brief Normalize the quaternion in place
  KOKKOS_INLINE_FUNCTION
  constexpr void normalize() MUNDY_REQUIRES(HasNonConstAccessOperator<Accessor, T>) {
    const T quat_norm = norm(*this);
    MUNDY_THROW_ASSERT(!is_close(quat_norm, T(0)), std::runtime_error, "AQuaternion: Cannot normalize zero norm.");
    const T inv_norm = T(1) / quat_norm;
    impl::access_at(accessor_, 0) *= inv_norm;
    impl::access_at(accessor_, 1) *= inv_norm;
    impl::access_at(accessor_, 2) *= inv_norm;
    impl::access_at(accessor_, 3) *= inv_norm;
  }

  /// \brief Conjugate the quaternion in place
  KOKKOS_INLINE_FUNCTION
  constexpr void conjugate() MUNDY_REQUIRES(HasNonConstAccessOperator<Accessor, T>) {
    impl::access_at(accessor_, x_storage_index) = -impl::access_at(accessor_, x_storage_index);
    impl::access_at(accessor_, y_storage_index) = -impl::access_at(accessor_, y_storage_index);
    impl::access_at(accessor_, z_storage_index) = -impl::access_at(accessor_, z_storage_index);
  }

  /// \brief Invert the quaternion in place
  KOKKOS_INLINE_FUNCTION
  constexpr void invert() MUNDY_REQUIRES(HasNonConstAccessOperator<Accessor, T>) {
    const T quat_norm_squared = impl::access_at(accessor_, 0) * impl::access_at(accessor_, 0) +
                                impl::access_at(accessor_, 1) * impl::access_at(accessor_, 1) +
                                impl::access_at(accessor_, 2) * impl::access_at(accessor_, 2) +
                                impl::access_at(accessor_, 3) * impl::access_at(accessor_, 3);
    MUNDY_THROW_ASSERT(!is_close(quat_norm_squared, T(0)), std::runtime_error, "AQuaternion: Cannot invert zero norm.");
    const T inv_norm_squared = T(1) / quat_norm_squared;
    conjugate();
    impl::access_at(accessor_, 0) *= inv_norm_squared;
    impl::access_at(accessor_, 1) *= inv_norm_squared;
    impl::access_at(accessor_, 2) *= inv_norm_squared;
    impl::access_at(accessor_, 3) *= inv_norm_squared;
  }
  //@}

  //! \name Unary operators
  //@{

  /// \brief Unary plus operator
  KOKKOS_INLINE_FUNCTION
  constexpr AQuaternion<T> operator+() const {
    return AQuaternion<T>(+w(), +x(), +y(), +z());
  }

  /// \brief Unary minus operator
  KOKKOS_INLINE_FUNCTION
  constexpr AQuaternion<T> operator-() const {
    return AQuaternion<T>(-w(), -x(), -y(), -z());
  }
  //@}

  //! \name Addition and subtraction
  //@{

  /// \brief AQuaternion-quaternion addition
  /// \param[in] other The other quaternion.
  template <typename U, ValidAccessor<U> OtherAccessor>
  KOKKOS_INLINE_FUNCTION constexpr auto operator+(const AQuaternion<U, OtherAccessor>& other) const
      -> AQuaternion<scalar_sum_result_t<T, U>> {
    return impl::quat_quat_addition_impl(*this, other);
  }

  /// \brief AQuaternion-quaternion addition
  /// \param[in] other The other quaternion.
  template <typename U, ValidAccessor<U> OtherAccessor>
  KOKKOS_INLINE_FUNCTION constexpr AQuaternion<T, Accessor>& operator+=(const AQuaternion<U, OtherAccessor>& other)
      MUNDY_REQUIRES(HasNonConstAccessOperator<Accessor, T>) {
    impl::self_quat_addition_impl(*this, other);
    return *this;
  }

  /// \brief AQuaternion-quaternion subtraction
  /// \param[in] other The other quaternion.
  template <typename U, ValidAccessor<U> OtherAccessor>
  KOKKOS_INLINE_FUNCTION constexpr auto operator-(const AQuaternion<U, OtherAccessor>& other) const
      -> AQuaternion<scalar_difference_result_t<T, U>> {
    return impl::quat_quat_subtraction_impl(*this, other);
  }

  /// \brief AQuaternion-quaternion subtraction
  /// \param[in] other The other quaternion.
  template <typename U, ValidAccessor<U> OtherAccessor>
  KOKKOS_INLINE_FUNCTION constexpr AQuaternion<T, Accessor>& operator-=(const AQuaternion<U, OtherAccessor>& other)
      MUNDY_REQUIRES(HasNonConstAccessOperator<Accessor, T>) {
    impl::self_quat_subtraction_impl(*this, other);
    return *this;
  }
  //@}

  //! \name Multiplication and division
  //@{

  /// \brief AQuaternion-quaternion multiplication
  /// \param[in] other The other quaternion.
  template <typename U, ValidAccessor<U> OtherAccessor>
  KOKKOS_INLINE_FUNCTION constexpr auto operator*(const AQuaternion<U, OtherAccessor>& other) const
      -> AQuaternion<scalar_product_result_t<T, U>> {
    return impl::quat_quat_multiplication_impl(*this, other);
  }

  /// \brief AQuaternion-quaternion multiplication
  /// \param[in] other The other quaternion.
  template <typename U, ValidAccessor<U> OtherAccessor>
  KOKKOS_INLINE_FUNCTION constexpr AQuaternion<T, Accessor>& operator*=(const AQuaternion<U, OtherAccessor>& other)
      MUNDY_REQUIRES(HasNonConstAccessOperator<Accessor, T>) {
    impl::self_quat_multiplication_impl(*this, other);
    return *this;
  }

  /// \brief AQuaternion-vector multiplication (same as R * v)
  /// \param[in] vec The vector.
  template <typename U, ValidAccessor<U> OtherAccessor>
  KOKKOS_INLINE_FUNCTION constexpr auto operator*(const AVector3<U, OtherAccessor>& vec) const
      -> AVector3<scalar_product_result_t<T, U>> {
    return impl::quat_vec_multiplication_impl(*this, vec);
  }

  /// \brief AQuaternion-matrix multiplication
  /// \param[in] other The other matrix.
  template <typename U, ValidAccessor<U> OtherAccessor>
  KOKKOS_INLINE_FUNCTION constexpr auto operator*(const AMatrix3<U, OtherAccessor>& mat) const
      -> AMatrix3<scalar_product_result_t<T, U>> {
    return impl::quat_mat_multiplication_impl(*this, mat);
  }

  /// \brief AQuaternion-scalar multiplication
  /// \param[in] scalar The scalar.
  template <typename U>
  MUNDY_REQUIRES(!is_quaternion_v<U> && ValidScalarType<U>)
  KOKKOS_INLINE_FUNCTION constexpr auto operator*(const U& scalar) const -> AQuaternion<scalar_product_result_t<T, U>> {
    return impl::quat_scalar_multiplication_impl(*this, scalar);
  }

  /// \brief Self-scalar multiplication
  /// \param[in] scalar The scalar.
  template <typename U>
  MUNDY_REQUIRES(HasNonConstAccessOperator<Accessor, T> && !is_quaternion_v<U> && ValidScalarType<U>)
  KOKKOS_INLINE_FUNCTION constexpr AQuaternion<T, Accessor>& operator*=(const U& scalar) {
    impl::self_scalar_multiplication_impl(*this, scalar);
    return *this;
  }

  /// \brief AQuaternion-scalar division
  /// \param[in] scalar The scalar.
  template <typename U>
  MUNDY_REQUIRES(!is_quaternion_v<U> && ValidScalarType<U>)
  KOKKOS_INLINE_FUNCTION constexpr auto operator/(const U& scalar) const
      -> AQuaternion<scalar_quotient_result_t<T, U>> {
    return impl::quat_scalar_division_impl(*this, scalar);
  }

  /// \brief Self-scalar division
  /// \param[in] scalar The scalar.
  template <typename U>
  MUNDY_REQUIRES(HasNonConstAccessOperator<Accessor, T> && !is_quaternion_v<U> && ValidScalarType<U>)
  KOKKOS_INLINE_FUNCTION constexpr AQuaternion<T, Accessor>& operator/=(const U& scalar) {
    impl::self_scalar_division_impl(*this, scalar);
    return *this;
  }
  //@}

  //! \name Static methods
  //@{

  /// \brief Get the identity quaternion
  KOKKOS_INLINE_FUNCTION
  static constexpr AQuaternion<T> identity() {
    return AQuaternion<T>(T(1), T(0), T(0), T(0));
  }
  //@}

  //! \name Friends <3
  //@{

  // Declare the << operator as a friend
  template <typename U, ValidAccessor<U> OtherAccessor>
  friend std::ostream& operator<<(std::ostream& os, const AQuaternion<U, OtherAccessor>& quat);

  // We are friends with all Quaternions regardless of their Accessor or type
  template <typename U, ValidAccessor<U> OtherAccessor>
  MUNDY_REQUIRES(ValidScalarType<U> && !NumTraits<U>::IsInteger)
  friend class AQuaternion;
  //@}
};  // AQuaternion

static_assert(is_quaternion_v<AQuaternion<double>>, "Odd, default AQuaternion is not a quaternion.");
static_assert(is_quaternion_v<AQuaternion<double, Array<double, 4>>>,
              "Odd, default AQuaternion with Array accessor is not a quaternion.");

/// \brief Type alias for a quaternion with the default accessor
template <typename T>
using Quaternion = AQuaternion<T, Array<T, 4>>;

//! \name Non-member functions
//@{

//! \name Write to output stream
//@{

/// \brief Write the quaternion to an output stream
/// \param[in] os The output stream.
/// \param[in] quat The quaternion.
template <typename T, ValidAccessor<T> Accessor>
std::ostream& operator<<(std::ostream& os, const AQuaternion<T, Accessor>& quat) {
  os << "(" << quat.w() << ", " << quat.x() << ", " << quat.y() << ", " << quat.z() << ")";
  return os;
}
//@}

//! \name Non-member comparison functions
//@{

/// \brief AQuaternion-quaternion equality (element-wise within a tolerance)
/// \param[in] quat1 The first quaternion.
/// \param[in] quat2 The second quaternion.
/// \param[in] tol The tolerance.
template <typename U, typename T, ValidAccessor<U> Accessor1, ValidAccessor<T> Accessor2>
KOKKOS_INLINE_FUNCTION constexpr bool is_close(
    const AQuaternion<U, Accessor1>& quat1, const AQuaternion<T, Accessor2>& quat2,
    const decltype(get_comparison_tolerance<T, U>())& tol = get_comparison_tolerance<T, U>()) {
  return abs(quat1.w() - quat2.w()) <= tol && abs(quat1.x() - quat2.x()) <= tol && abs(quat1.y() - quat2.y()) <= tol &&
         abs(quat1.z() - quat2.z()) <= tol;
}

/// \brief AQuaternion-quaternion equality (element-wise within a relaxed tolerance)
/// \param[in] quat1 The first quaternion.
/// \param[in] quat2 The second quaternion.
/// \param[in] tol The tolerance.
template <typename U, typename T, ValidAccessor<U> Accessor1, ValidAccessor<T> Accessor2>
KOKKOS_INLINE_FUNCTION constexpr bool is_approx_close(
    const AQuaternion<U, Accessor1>& quat1, const AQuaternion<T, Accessor2>& quat2,
    const decltype(get_relaxed_comparison_tolerance<T, U>())& tol = get_relaxed_comparison_tolerance<T, U>()) {
  return is_close(quat1, quat2, tol);
}
//@}

//! \name Non-member multiplication and division operators
//@{

/// \brief Scalar-quaternion multiplication
/// \param[in] scalar The scalar.
/// \param[in] quat The quaternion.
template <typename U, typename T, ValidAccessor<T> Accessor>
MUNDY_REQUIRES(!is_quaternion_v<U> && ValidScalarType<U>)
KOKKOS_INLINE_FUNCTION constexpr auto operator*(const U& scalar, const AQuaternion<T, Accessor>& quat)
    -> AQuaternion<scalar_product_result_t<T, U>> {
  return quat * scalar;
}

/// \brief Vector-quaternion multiplication (same as v^T * R = transpose(R^T * v))
/// \param[in] vec The vector.
/// \param[in] quat The quaternion.
template <typename U, typename T, ValidAccessor<U> Accessor1, ValidAccessor<T> Accessor2>
KOKKOS_INLINE_FUNCTION constexpr auto operator*(const AVector3<U, Accessor1>& vec,
                                                const AQuaternion<T, Accessor2>& quat)
    -> Vector3<scalar_product_result_t<T, U>> {
  return impl::vec_quat_multiplication_impl(vec, quat);
}

/// \brief Matrix-quaternion multiplication (same as R * M)
/// \param[in] mat The matrix.
/// \param[in] quat The quaternion.
template <typename U, typename T, ValidAccessor<U> Accessor1, ValidAccessor<T> Accessor2>
KOKKOS_INLINE_FUNCTION constexpr auto operator*(const AMatrix3<U, Accessor1>& mat,
                                                const AQuaternion<T, Accessor2>& quat)
    -> AMatrix3<scalar_product_result_t<T, U>> {
  return impl::mat_quat_multiplication_impl(mat, quat);
}
//@}

//! \name Special quaternion operations
//@{

/// \brief Get a deep copy of the given quaternion
template <ValidQuaternionType QuaternionType>
KOKKOS_INLINE_FUNCTION constexpr auto copy(const QuaternionType& q) {
  return q.copy();
}

/// \brief Cast a quaternion to a different scalar type
template <typename U, ValidQuaternionType QuaternionType>
KOKKOS_INLINE_FUNCTION constexpr auto cast(const QuaternionType& q) {
  return q.template cast<U>();
}

/// \brief Get the dot product of two quaternions
/// \param[in] q1 The first quaternion.
/// \param[in] q2 The second quaternion.
template <typename U, typename T, ValidAccessor<U> Accessor1, ValidAccessor<T> Accessor2>
KOKKOS_INLINE_FUNCTION constexpr auto dot(const AQuaternion<U, Accessor1>& q1, const AQuaternion<T, Accessor2>& q2) {
  using CommonType = scalar_product_result_t<U, T>;
  return static_cast<CommonType>(q1.w()) * static_cast<CommonType>(q2.w()) +
         static_cast<CommonType>(q1.x()) * static_cast<CommonType>(q2.x()) +
         static_cast<CommonType>(q1.y()) * static_cast<CommonType>(q2.y()) +
         static_cast<CommonType>(q1.z()) * static_cast<CommonType>(q2.z());
}

/// \brief Get the conjugate of a quaternion
/// \param[in] quat The quaternion.
template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr AQuaternion<std::remove_const_t<T>> conjugate(const AQuaternion<T, Accessor>& quat) {
  return AQuaternion<std::remove_const_t<T>>(quat.w(), -quat.x(), -quat.y(), -quat.z());
}

/// \brief Get the inverse of a quaternion
/// \param[in] quat The quaternion.
template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr AQuaternion<std::remove_const_t<T>> inverse(const AQuaternion<T, Accessor>& quat) {
  const T quat_norm_squared = quat.w() * quat.w() + quat.x() * quat.x() + quat.y() * quat.y() + quat.z() * quat.z();
  MUNDY_THROW_ASSERT(!is_close(quat_norm_squared, T(0)), std::runtime_error, "AQuaternion: Cannot invert zero norm.");
  const T inv_norm_squared = T(1) / quat_norm_squared;
  return conjugate(quat) * inv_norm_squared;
}

/// \brief Get the norm of a quaternion
/// \param[in] quat The quaternion.
template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto norm(const AQuaternion<T, Accessor>& quat) {
  return sqrt(quat.w() * quat.w() + quat.x() * quat.x() + quat.y() * quat.y() + quat.z() * quat.z());
}

/// \brief Get the squared norm of a quaternion
/// \param[in] quat The quaternion.
template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto norm_squared(const AQuaternion<T, Accessor>& quat) {
  return quat.w() * quat.w() + quat.x() * quat.x() + quat.y() * quat.y() + quat.z() * quat.z();
}

/// \brief Get the normalized quaternion
/// \param[in] quat The quaternion.
template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr AQuaternion<std::remove_const_t<T>> normalize(const AQuaternion<T, Accessor>& quat) {
  const T quat_norm = norm(quat);
  MUNDY_THROW_ASSERT(!is_close(quat_norm, T(0)), std::runtime_error, "AQuaternion: Cannot normalize zero norm.");
  const T inv_norm = static_cast<T>(1) / quat_norm;
  return quat * inv_norm;
}

/// \brief Perform spherical linear interpolation between two quaternions
/// \param[in] q1 The first quaternion.
/// \param[in] q2 The second quaternion.
/// \param[in] t The interpolation parameter.
template <typename U, typename T, typename V, ValidAccessor<U> Accessor1, ValidAccessor<T> Accessor2>
MUNDY_REQUIRES(ValidScalarType<V>)
KOKKOS_INLINE_FUNCTION
    constexpr auto slerp(const AQuaternion<U, Accessor1>& q1, const AQuaternion<T, Accessor2>& q2, const V t)
        -> AQuaternion<scalar_product_result_t<scalar_product_result_t<U, T>, V>> {
  using CommonType = scalar_product_result_t<scalar_product_result_t<U, T>, V>;
  const CommonType epsilon = get_relaxed_zero_tolerance<CommonType>();  // Threshold for linear interpolation

  // Compute the dot product
  CommonType dot_q12 = dot(q1, q2);

  // Adjust second quaternion for negative dot product
  // Note, we cannot directly copy from q2 to q2_adjusted because the Accessor type may be different.
  AQuaternion<std::remove_const_t<T>> q2_adjusted;
  q2_adjusted.set(q2);
  if (dot_q12 < static_cast<CommonType>(0)) {
    dot_q12 = -dot_q12;
    q2_adjusted *= -1;
  }

  // Clamp the dot product to avoid numerical issues with acos
  if (dot_q12 > static_cast<CommonType>(1)) {
    dot_q12 = static_cast<CommonType>(1);
  } else if (dot_q12 < static_cast<CommonType>(-1)) {
    dot_q12 = static_cast<CommonType>(-1);
  }

  // Check for near-parallel case
  if (static_cast<CommonType>(1) - dot_q12 < epsilon) {
    // Linear Interpolation as fallback
    return AQuaternion<CommonType>{
        static_cast<CommonType>(q1.w()) +
            static_cast<CommonType>(t) * (static_cast<CommonType>(q2_adjusted.w()) - static_cast<CommonType>(q1.w())),
        static_cast<CommonType>(q1.x()) +
            static_cast<CommonType>(t) * (static_cast<CommonType>(q2_adjusted.x()) - static_cast<CommonType>(q1.x())),
        static_cast<CommonType>(q1.y()) +
            static_cast<CommonType>(t) * (static_cast<CommonType>(q2_adjusted.y()) - static_cast<CommonType>(q1.y())),
        static_cast<CommonType>(q1.z()) +
            static_cast<CommonType>(t) * (static_cast<CommonType>(q2_adjusted.z()) - static_cast<CommonType>(q1.z()))};
  } else {
    // Spherical Interpolation
    const CommonType theta = acos(dot_q12);
    const CommonType sin_theta = sin(theta);
    MUNDY_THROW_ASSERT(!is_close(sin_theta, CommonType(0)), std::runtime_error,
                       "AQuaternion: slerp undefined for sin(theta) near zero.");
    const CommonType inv_sin_theta = static_cast<CommonType>(1) / sin_theta;
    const CommonType s1 = sin((static_cast<CommonType>(1) - static_cast<CommonType>(t)) * theta) * inv_sin_theta;
    const CommonType s2 = sin(static_cast<CommonType>(t) * theta) * inv_sin_theta;

    return AQuaternion<CommonType>{(static_cast<CommonType>(s1) * static_cast<CommonType>(q1.w())) +
                                       (static_cast<CommonType>(s2) * static_cast<CommonType>(q2_adjusted.w())),
                                   (static_cast<CommonType>(s1) * static_cast<CommonType>(q1.x())) +
                                       (static_cast<CommonType>(s2) * static_cast<CommonType>(q2_adjusted.x())),
                                   (static_cast<CommonType>(s1) * static_cast<CommonType>(q1.y())) +
                                       (static_cast<CommonType>(s2) * static_cast<CommonType>(q2_adjusted.y())),
                                   (static_cast<CommonType>(s1) * static_cast<CommonType>(q1.z())) +
                                       (static_cast<CommonType>(s2) * static_cast<CommonType>(q2_adjusted.z()))};
  }
}

// /// \brief Perform spherical linear interpolation between two quaternions
// /// Source: https://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/slerp/index.htm
// /// \param[in] q1 The first quaternion.
// /// \param[in] q2 The second quaternion.
// /// \param[in] t The interpolation parameter.
// template <typename U, typename T, typename V>
//   requires std::is_arithmetic_v<V>
// template <typename U, typename T, typename V, ValidAccessor<U> Accessor1, ValidAccessor<T>
// Accessor2>
//   requires std::is_arithmetic_v<V>
// KOKKOS_INLINE_FUNCTION constexpr auto slerp(const AQuaternion<U, Accessor1> &q1, const AQuaternion<T,
// Accessor2> &q2,
//                                   const V t) -> AQuaternion<std::common_type_t<U, T, V>> {
//   using CommonType = decltype(U() * T() * V());

//   // quaternion to return
//   quat qm = new quat();
//   // Calculate angle between them.
//   double cosHalfTheta = qa.w * qb.w + qa.x * qb.x + qa.y * qb.y + qa.z * qb.z;
//   // if qa=qb or qa=-qb then theta = 0 and we can return qa
//   if (abs(cosHalfTheta) >= 1.0) {
//     qm.w = qa.w;
//     qm.x = qa.x;
//     qm.y = qa.y;
//     qm.z = qa.z;
//     return qm;
//   }
//   // Calculate temporary values.
//   double halfTheta = acos(cosHalfTheta);
//   double sinHalfTheta = sqrt(1.0 - cosHalfTheta * cosHalfTheta);
//   // if theta = 180 degrees then result is not fully defined
//   // we could rotate around any axis normal to qa or qb
//   if (fabs(sinHalfTheta) < 0.001) {  // fabs is floating point absolute
//     qm.w = (qa.w * 0.5 + qb.w * 0.5);
//     qm.x = (qa.x * 0.5 + qb.x * 0.5);
//     qm.y = (qa.y * 0.5 + qb.y * 0.5);
//     qm.z = (qa.z * 0.5 + qb.z * 0.5);
//     return qm;
//   }
//   double ratioA = sin((1 - t) * halfTheta) / sinHalfTheta;
//   double ratioB = sin(t * halfTheta) / sinHalfTheta;
//   // calculate AQuaternion.
//   qm.w = (qa.w * ratioA + qb.w * ratioB);
//   qm.x = (qa.x * ratioA + qb.x * ratioB);
//   qm.y = (qa.y * ratioA + qb.y * ratioB);
//   qm.z = (qa.z * ratioA + qb.z * ratioB);
//   return qm;
// }

/// \brief Rotate a quaternion by an angular velocity omega dt
///
/// Delong, JCP, 2015, Appendix A eq1, not linearized
///
/// \param q The quaternion to rotate
/// \param omega The angular velocity
/// \param dt The time
template <ValidQuaternionType QuaternionType, ValidVectorType VectorType>
MUNDY_REQUIRES(std::is_same_v<typename QuaternionType::value_type, typename VectorType::value_type>)
KOKKOS_INLINE_FUNCTION constexpr void rotate_quaternion(QuaternionType& quat, const VectorType& omega,
                                                        const typename QuaternionType::value_type& dt) {
  using Scalar = typename QuaternionType::value_type;
  const Scalar w = norm(omega);
  if (w < get_zero_tolerance<Scalar>()) {
    return;
  }
  const Scalar winv = Scalar(1) / w;
  const Scalar sw = sin(Scalar(0.5) * w * dt);
  const Scalar cw = cos(Scalar(0.5) * w * dt);
  const Scalar s = quat.w();
  const auto p = quat.vector();
  const auto xyz = s * sw * omega * winv + cw * p + sw * winv * cross(omega, p);
  quat.w() = s * cw - dot(omega, p) * sw * winv;
  quat.vector() = xyz;
  quat.normalize();
}
//@}

//! \name Non-member constructors and converters
//@{

/// \brief Get the quaternion from an axis-angle representation
/// \param[in] axis The axis.
/// \param[in] angle The angle.
template <typename T, typename U, ValidAccessor<T> Accessor>
MUNDY_REQUIRES(ValidScalarType<U>)
KOKKOS_INLINE_FUNCTION constexpr auto axis_angle_to_quaternion(const AVector3<T, Accessor>& axis, const U& angle)
    -> AQuaternion<scalar_product_result_t<T, U>> {
  using CommonType = scalar_product_result_t<T, U>;
  const auto half_angle = U(0.5) * angle;
  const auto sin_half_angle = sin(half_angle);
  const auto cos_half_angle = cos(half_angle);
  return AQuaternion<CommonType>(static_cast<CommonType>(cos_half_angle),
                                 static_cast<CommonType>(sin_half_angle) * static_cast<CommonType>(axis[0]),
                                 static_cast<CommonType>(sin_half_angle) * static_cast<CommonType>(axis[1]),
                                 static_cast<CommonType>(sin_half_angle) * static_cast<CommonType>(axis[2]));
}

/// \brief Get the quaternion from a rotation matrix
/// \param[in] rot_mat The rotation matrix.
template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr AQuaternion<T> rotation_matrix_to_quaternion(const AMatrix3<T, Accessor>& rot_mat) {
  // Source: https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
  AQuaternion<T> quat;

  // Computing the quaternion components
  quat.w() = sqrt(max(T(0), T(1) + rot_mat(0, 0) + rot_mat(1, 1) + rot_mat(2, 2))) / T(2);
  quat.x() = sqrt(max(T(0), T(1) + rot_mat(0, 0) - rot_mat(1, 1) - rot_mat(2, 2))) / T(2);
  quat.y() = sqrt(max(T(0), T(1) - rot_mat(0, 0) + rot_mat(1, 1) - rot_mat(2, 2))) / T(2);
  quat.z() = sqrt(max(T(0), T(1) - rot_mat(0, 0) - rot_mat(1, 1) + rot_mat(2, 2))) / T(2);

  // Correcting the signs
  quat.x() = copysign(quat.x(), rot_mat(2, 1) - rot_mat(1, 2));
  quat.y() = copysign(quat.y(), rot_mat(0, 2) - rot_mat(2, 0));
  quat.z() = copysign(quat.z(), rot_mat(1, 0) - rot_mat(0, 1));

  return quat;
}

/// \brief Get the rotation matrix from a quaternion
/// \param[in] quat The quaternion.
template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr Matrix3<std::remove_const_t<T>> quaternion_to_rotation_matrix(
    const AQuaternion<T, Accessor>& quat) {
  Matrix3<std::remove_const_t<T>> rot_mat;
  rot_mat(0, 0) = T(1) - T(2) * quat.y() * quat.y() - T(2) * quat.z() * quat.z();
  rot_mat(0, 1) = T(2) * quat.x() * quat.y() - T(2) * quat.w() * quat.z();
  rot_mat(0, 2) = T(2) * quat.x() * quat.z() + T(2) * quat.w() * quat.y();
  rot_mat(1, 0) = T(2) * quat.x() * quat.y() + T(2) * quat.w() * quat.z();
  rot_mat(1, 1) = T(1) - T(2) * quat.x() * quat.x() - T(2) * quat.z() * quat.z();
  rot_mat(1, 2) = T(2) * quat.y() * quat.z() - T(2) * quat.w() * quat.x();
  rot_mat(2, 0) = T(2) * quat.x() * quat.z() - T(2) * quat.w() * quat.y();
  rot_mat(2, 1) = T(2) * quat.y() * quat.z() + T(2) * quat.w() * quat.x();
  rot_mat(2, 2) = T(1) - T(2) * quat.x() * quat.x() - T(2) * quat.y() * quat.y();

  return rot_mat;
}

/// \brief Get the quaternion from Euler angles
/// https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions#Euler_angles_%E2%86%94_quaternion
/// \param[in] roll Roll angle.
/// \param[in] pitch Pitch angle.
/// \param[in] yaw Yaw angle.
template <typename T>
MUNDY_REQUIRES(ValidScalarType<T>)
KOKKOS_INLINE_FUNCTION
    constexpr AQuaternion<std::remove_const_t<T>> euler_to_quat(const T roll, const T pitch, const T yaw) {
  // Convert Euler angles to quaternion
  AQuaternion<std::remove_const_t<T>> quat;
  const T cha1 = cos(T(0.5) * roll);
  const T cha2 = cos(T(0.5) * pitch);
  const T cha3 = cos(T(0.5) * yaw);
  const T sha1 = sin(T(0.5) * roll);
  const T sha2 = sin(T(0.5) * pitch);
  const T sha3 = sin(T(0.5) * yaw);
  quat.w() = cha1 * cha2 * cha3 + sha1 * sha2 * sha3;
  quat.x() = sha1 * cha2 * cha3 - cha1 * sha2 * sha3;
  quat.y() = cha1 * sha2 * cha3 + sha1 * cha2 * sha3;
  quat.z() = cha1 * cha2 * sha3 - sha1 * sha2 * cha3;
  return quat;
}

/// \brief Get the quaternion that perform parallel transport from unit vector v1 to unit vector v2
/// \param[in] v1 The first vector.
/// \param[in] v2 The second vector.
///
/// The parallel transport quaternion from a to b is given by
///
/// p_a^b
///  = \frac{1}{\sqrt{2}} \sqrt{1 + a \cdot b} \left( 1 + \frac{a \times b}{1 + a \cdot b} \right)
///  = \frac{1}{\sqrt{2}} \left( \sqrt{1 + a \cdot b} + \frac{a \times b}{\sqrt{1 + a \cdot b}} \right)
///  = \sqrt{\frac{1 + a \cdot b}{2}} + \frac{1}{2} \frac{a \times b}{\sqrt{(1 + a \cdot b) / 2}}
///
/// This equation comes from J. Linn's 2020 "Discrete Cosserat rod kinematics constricted on the basis
/// of the difference geometry of framed curves," and as shown above, is identical to the equation given in K. Korner's
/// "Simple deformation measures for discrete elastic rods and ribbons."
///
/// \pre v_from, v_to are unit.
///
/// \note Antiparallel inputs (v_to == -v_from) are singular: any axis perpendicular to v_from gives a valid 180-deg
/// rotation. When 1 + v_from . v_to <= tol we pick one arbitrarily: 180 deg about v_from x (least-aligned world axis).
template <typename U, typename T, ValidAccessor<U> Accessor1, ValidAccessor<T> Accessor2>
MUNDY_REQUIRES(ValidScalarType<T>&& ValidScalarType<U>)
KOKKOS_INLINE_FUNCTION constexpr auto quat_from_parallel_transport(const AVector3<U, Accessor1>& v_from,
                                                                   const AVector3<T, Accessor2>& v_to)
    -> AQuaternion<decltype(U() * T())> {
  // Get the quaternion that performs parallel transport from vector v_from to vector v_to
  using CommonType = decltype(U() * T());
  AQuaternion<CommonType> quat;

  const auto dot_product = dot(v_from, v_to);
  const CommonType one_plus_dot = CommonType(1) + dot_product;

  // Antiparallel singularity (dot ~= -1): axis undefined.
  // Arbitrary tie-break: 180 deg about v_from x least-aligned world axis (smallest component, for stability).
  if (one_plus_dot <= get_comparison_tolerance<U, T>()) {
    const auto ax = abs(v_from[0]);
    const auto ay = abs(v_from[1]);
    const auto az = abs(v_from[2]);
    CommonType nx, ny, nz;
    if (ax <= ay && ax <= az) {  // cross(v_from, e_x) = (0, v_z, -v_y)
      nx = CommonType(0);
      ny = v_from[2];
      nz = -v_from[1];
    } else if (ay <= az) {  // cross(v_from, e_y) = (-v_z, 0, v_x)
      nx = -v_from[2];
      ny = CommonType(0);
      nz = v_from[0];
    } else {  // cross(v_from, e_z) = (v_y, -v_x, 0)
      nx = v_from[1];
      ny = -v_from[0];
      nz = CommonType(0);
    }
    const CommonType inv_len = CommonType(1) / Kokkos::sqrt(nx * nx + ny * ny + nz * nz);
    quat.w() = CommonType(0);
    quat.x() = nx * inv_len;
    quat.y() = ny * inv_len;
    quat.z() = nz * inv_len;
    return quat;
  }

  // Regular case
  const auto cross_product = cross(v_from, v_to);
  const CommonType sqrt_term = Kokkos::sqrt(CommonType(0.5) * one_plus_dot);
  const auto vec = CommonType(0.5) * cross_product / sqrt_term;
  quat.w() = sqrt_term;
  quat.x() = vec[0];
  quat.y() = vec[1];
  quat.z() = vec[2];
  return quat;
}
//@}

// Just to double check
static_assert(std::is_trivially_copyable_v<AQuaternion<double>>);
static_assert(std::is_trivially_destructible_v<AQuaternion<double>>);
static_assert(std::is_copy_constructible_v<AQuaternion<double>>);
static_assert(std::is_move_constructible_v<AQuaternion<double>>);

//! \name Type specializations
//@{

#define MUNDY_MATH_QUATERNION_TYPE_SPECIALIZATION(alias, alias_lower, T)                  \
  template <ValidAccessor<T> Accessor = Array<T, 4>>                                      \
  using A##alias = AQuaternion<T, Accessor>;                                              \
  using alias = A##alias<>;                                                               \
  template <typename TypeToCheck>                                                         \
  struct is_##alias_lower##_impl : std::false_type {};                                    \
  template <typename Accessor>                                                            \
  struct is_##alias_lower##_impl<A##alias<Accessor>> : std::true_type {};                 \
  template <typename TypeToCheck>                                                         \
  struct is_##alias_lower : public is_##alias_lower##_impl<std::decay_t<TypeToCheck>> {}; \
  template <typename TypeToCheck>                                                         \
  constexpr bool is_##alias_lower##_v = is_##alias_lower<TypeToCheck>::value;

// Eigen convention.
MUNDY_MATH_QUATERNION_TYPE_SPECIALIZATION(Quaterniond, quaterniond, double)
MUNDY_MATH_QUATERNION_TYPE_SPECIALIZATION(Quaternionf, quaternionf, float)
//@}

//! \name AQuaternion<T, Accessor> views
//@{

/// \brief A helper function to create a AQuaternion<T, Accessor> based on a given accessor.
/// \param[in] data The data accessor.
/// \note The accessor is interpreted in the raw storage order `(x, y, z, w)`.
///
/// In practice, this function is syntactic sugar to avoid having to specify the template parameters
/// when creating a AQuaternion<T, Accessor> from a data accessor.
/// Instead of writing
/// \code
///   AQuaternion<T, Accessor> quat(data);
/// \endcode
/// you can write
/// \code
///   auto quat = get_quaternion_view<T>(data);
/// \endcode
template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_quaternion_view(Accessor&& data) {
  using accessor_t = typename storage<Accessor>::stored_type;
  return AQuaternion<T, accessor_t>(accessor_t(std::forward<Accessor>(data)));
}

template <typename T, ValidAccessor<T> Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_owning_quaternion(Accessor&& data) {
  using accessor_t = typename storage<std::remove_reference_t<Accessor>>::stored_type;
  return AQuaternion<T, accessor_t>(accessor_t(std::forward<Accessor>(data)));
}

#define MUNDY_MATH_GET_QUATERNION_TYPE_SPECIALIZATION(alias, alias_lower, T)         \
  template <ValidAccessor<T> Accessor>                                               \
  KOKKOS_INLINE_FUNCTION constexpr auto get_##alias_lower##_view(Accessor&& data) {  \
    return get_quaternion_view<T>(std::forward<Accessor>(data));                     \
  }                                                                                  \
                                                                                     \
  template <ValidAccessor<T> Accessor>                                               \
  KOKKOS_INLINE_FUNCTION constexpr auto get_owning_##alias_lower(Accessor&& data) {  \
    return get_owning_quaternion<T>(std::forward<Accessor>(data));                   \
  }

/// \brief Accessor helpers for each AQuaternion specialization, mirroring the type specializations above.
MUNDY_MATH_GET_QUATERNION_TYPE_SPECIALIZATION(Quaterniond, quaterniond, double)
MUNDY_MATH_GET_QUATERNION_TYPE_SPECIALIZATION(Quaternionf, quaternionf, float)
//@}

//@}

}  // namespace mundy

#endif  // MUNDY_MATH_QUATERNION_HPP_
