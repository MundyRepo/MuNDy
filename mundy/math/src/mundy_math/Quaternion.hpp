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
#include <initializer_list>  // for std::initializer_list
#include <type_traits>       // for std::decay_t
#include <utility>

// Mundy
#include <mundy_core/throw_assert.hpp>  // for MUNDY_THROW_ASSERT
#include <mundy_math/Accessor.hpp>      // for mundy::math::ValidAccessor
#include <mundy_math/Array.hpp>         // for mundy::math::Array
#include <mundy_math/Matrix3.hpp>       // for mundy::math::AMatrix3
#include <mundy_math/Tolerance.hpp>     // for mundy::math::get_zero_tolerance
#include <mundy_math/Vector3.hpp>       // for mundy::math::AVector3
#include <mundy_math/impl/QuaternionImpl.hpp>

namespace mundy {

namespace math {

/// \brief (Implementation) Type trait to determine if a type is a AQuaternion
template <typename TypeToCheck>
struct is_quaternion_impl : std::false_type {};
//
template <typename T, typename Accessor, typename OwnershipType>
struct is_quaternion_impl<AQuaternion<T, Accessor, OwnershipType>> : std::true_type {};

/// \brief Type trait to determine if a type is a AQuaternion
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
      typename std::decay_t<QuaternionType>::scalar_t;
      { quaternion[0] } -> std::convertible_to<typename std::decay_t<QuaternionType>::scalar_t>;
      { quaternion[1] } -> std::convertible_to<typename std::decay_t<QuaternionType>::scalar_t>;
      { quaternion[2] } -> std::convertible_to<typename std::decay_t<QuaternionType>::scalar_t>;
      { quaternion[3] } -> std::convertible_to<typename std::decay_t<QuaternionType>::scalar_t>;

      { quaternion(0) } -> std::convertible_to<typename std::decay_t<QuaternionType>::scalar_t>;
      { quaternion(1) } -> std::convertible_to<typename std::decay_t<QuaternionType>::scalar_t>;
      { quaternion(2) } -> std::convertible_to<typename std::decay_t<QuaternionType>::scalar_t>;
      { quaternion(3) } -> std::convertible_to<typename std::decay_t<QuaternionType>::scalar_t>;

      { const_quaternion[0] } -> std::convertible_to<const typename std::decay_t<QuaternionType>::scalar_t>;
      { const_quaternion[1] } -> std::convertible_to<const typename std::decay_t<QuaternionType>::scalar_t>;
      { const_quaternion[2] } -> std::convertible_to<const typename std::decay_t<QuaternionType>::scalar_t>;
      { const_quaternion[3] } -> std::convertible_to<const typename std::decay_t<QuaternionType>::scalar_t>;

      { const_quaternion(0) } -> std::convertible_to<const typename std::decay_t<QuaternionType>::scalar_t>;
      { const_quaternion(1) } -> std::convertible_to<const typename std::decay_t<QuaternionType>::scalar_t>;
      { const_quaternion(2) } -> std::convertible_to<const typename std::decay_t<QuaternionType>::scalar_t>;
      { const_quaternion(3) } -> std::convertible_to<const typename std::decay_t<QuaternionType>::scalar_t>;
    };  // ValidQuaternionType

//! \name Forward declare AQuaternion functions that also require AQuaternion to be defined
//@{

/// \brief Get the norm of a quaternion
/// \param[in] quat The quaternion.
template <typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto norm(const AQuaternion<T, Accessor, OwnershipType> &quat);
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
/// the ValidAccessor requirements without copying the data. This is especially important for GPU-compatable code.
///
/// AQuaternions can be constructed by passing an accessor to the constructor. However, if the accessor has a 4-argument
/// constructor, then the AQuaternion can also be constructed by passing the elements directly to the constructor.
/// Similarly, if the accessor has an initializer list constructor, then the AQuaternion can be constructed by passing
/// an initializer list to the constructor. This is a convenience feature which makes working with the default accessor
/// (Array<T, 4>) easier. For example, the following are all valid ways to construct a AQuaternion:
///
/// \code{.cpp}
///   // Constructs a AQuaternion with the default accessor (Array<int, 4>)
///   AQuaternion<double> quat1({1.0, 2.0, 3.0, 4.0});
///   AQuaternion<double> quat2(1.0, 2.0, 3.0, 4.0);
///   AQuaternion<double> quat3(Array<int, 3>({1.0, 2.0, 3.0, 4.0}));
///   AQuaternion<double> quat4;
///   quat4.set(1.0, 2.0, 3.0, 4.0);
///
///   // Construct a AQuaternion from a double array
///   double data[4] = {1.0, 2.0, 3.0, 4.0};
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
  requires std::is_floating_point_v<T>
class AQuaternion<T, Accessor, Ownership::Views> {
 public:
  //! \name Internal data
  //@{

  /// \brief A reference or a pointer to an external data accessor.
  std::conditional_t<std::is_pointer_v<Accessor>, Accessor, Accessor &> accessor_;
  //@}

  //! \name Type aliases
  //@{

  /// \brief The type of the entries
  using scalar_t = T;

  /// \brief The non-const type of the entries
  using non_const_scalar_t = std::remove_const_t<T>;

  /// \brief Our ownership type
  using ownership_t = Ownership::Views;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief No default constructor since we don't own the data.
  KOKKOS_INLINE_FUNCTION AQuaternion() = delete;

  /// \brief Constructor for reference accessors
  KOKKOS_INLINE_FUNCTION
  explicit AQuaternion(Accessor &data)
    requires(!std::is_pointer_v<Accessor>)
      : accessor_(data) {
  }

  /// \brief Constructor for pointer accessors
  KOKKOS_INLINE_FUNCTION
  explicit AQuaternion(Accessor data)
    requires std::is_pointer_v<Accessor>
      : accessor_(data) {
  }

  /// \brief Destructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr ~AQuaternion() = default;

  // Default copy/move constructors and assignment operators when interacting with a AQuaternion of the same type

  /// \brief Default copy constructor (shallow copy)
  KOKKOS_DEFAULTED_FUNCTION
  constexpr AQuaternion(const AQuaternion<T, Accessor, Ownership::Views> &) = default;

  /// \brief Default move constructor (shallow move)
  KOKKOS_DEFAULTED_FUNCTION
  constexpr AQuaternion(AQuaternion<T, Accessor, Ownership::Views> &&) = default;

  /// \brief Default copy assignment operator (shallow copy)
  KOKKOS_DEFAULTED_FUNCTION
  constexpr AQuaternion<T, Accessor, Ownership::Views> &operator=(const AQuaternion<T, Accessor, Ownership::Views> &) =
      default;

  /// \brief Default move assignment operator (shallow move)
  KOKKOS_DEFAULTED_FUNCTION
  constexpr AQuaternion<T, Accessor, Ownership::Views> &operator=(AQuaternion<T, Accessor, Ownership::Views> &&) =
      default;

  // Custom copy/move constructors and assignment operators when interacting with a AQuaternion of a different type
  // We do not allow copy/move construction from a AQuaternion of a different type. This is undefined behavior.

  /// \brief Deep copy assignment operator with different accessor or ownership
  /// \details Copies the data from the other vector to our data. This is only enabled if T is not const.
  template <ValidQuaternionType OtherQuaternionType>
  KOKKOS_INLINE_FUNCTION constexpr AQuaternion<T, Accessor, Ownership::Views> &operator=(
      const OtherQuaternionType &other)
    requires(!std::is_same_v<OtherQuaternionType, AQuaternion<T, Accessor, Ownership::Views>>) &&
            (std::is_same_v<typename OtherQuaternionType::scalar_t, T>) && HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(*this, other);
    return *this;
  }

  /// \brief Deep move assignment operator with different accessor or ownership
  /// \details Moves the data from the other vector to our data. This is only enabled if T is not const.
  template <ValidQuaternionType OtherQuaternionType>
  KOKKOS_INLINE_FUNCTION constexpr AQuaternion<T, Accessor, Ownership::Views> &operator=(OtherQuaternionType &&other)
    requires(!std::is_same_v<OtherQuaternionType, AQuaternion<T, Accessor, Ownership::Views>>) &&
            (std::is_same_v<typename OtherQuaternionType::scalar_t, T>) && HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(*this, std::move(other));
    return *this;
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Element access operator via a single index
  /// \param[in] index The index of the element.
  KOKKOS_INLINE_FUNCTION
  constexpr T &operator[](int index) {
    return accessor_[index];
  }

  /// \brief Const element access operator via a single index
  /// \param[in] index The index of the element.
  KOKKOS_INLINE_FUNCTION
  constexpr const T &operator[](int index) const {
    return accessor_[index];
  }

  /// \brief Element access operator via a single index
  /// \param[in] index The index of the element.
  KOKKOS_INLINE_FUNCTION
  constexpr T &operator()(int index) {
    return accessor_[index];
  }

  /// \brief Const element access operator via a single index
  /// \param[in] index The index of the element.
  KOKKOS_INLINE_FUNCTION
  constexpr const T &operator()(int index) const {
    return accessor_[index];
  }

  /// \brief Get a reference to the scalar component
  KOKKOS_INLINE_FUNCTION
  constexpr T &w() {
    return accessor_[0];
  }

  /// \brief Get a reference to the scalar component
  KOKKOS_INLINE_FUNCTION
  constexpr const T &w() const {
    return accessor_[0];
  }

  /// \brief Get a reference to the x component
  KOKKOS_INLINE_FUNCTION
  constexpr T &x() {
    return accessor_[1];
  }

  /// \brief Get a reference to the x component
  KOKKOS_INLINE_FUNCTION
  constexpr const T &x() const {
    return accessor_[1];
  }

  /// \brief Get a reference to the y component
  KOKKOS_INLINE_FUNCTION
  constexpr T &y() {
    return accessor_[2];
  }

  /// \brief Get a reference to the y component
  KOKKOS_INLINE_FUNCTION
  constexpr const T &y() const {
    return accessor_[2];
  }

  /// \brief Get a reference to the z component
  KOKKOS_INLINE_FUNCTION
  constexpr T &z() {
    return accessor_[3];
  }

  /// \brief Get a reference to the z component
  KOKKOS_INLINE_FUNCTION
  constexpr const T &z() const {
    return accessor_[3];
  }

  /// \brief Get the internal data accessor
  KOKKOS_INLINE_FUNCTION
  constexpr Accessor data() {
    return accessor_;
  }

  /// \brief Get the internal data accessor
  KOKKOS_INLINE_FUNCTION
  constexpr const Accessor data() const {
    return accessor_;
  }

  /// \brief Get a view of the quaternion vector component
  KOKKOS_INLINE_FUNCTION
  constexpr const auto vector() const {
    auto shifted_accessor = get_shifted_view<T, 1>(accessor_);
    return get_owning_vector<T, 3>(std::move(shifted_accessor));
  }

  /// \brief Get a view of the quaternion vector component
  KOKKOS_INLINE_FUNCTION
  constexpr auto vector() {
    auto shifted_accessor = get_shifted_view<T, 1>(accessor_);
    return get_owning_vector<T, 3>(std::move(shifted_accessor));
  }

  /// \brief Cast (and copy) the quaternion to a different type
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr auto cast() const {
    return AQuaternion<U>{static_cast<U>(accessor_[0]),  //
                          static_cast<U>(accessor_[1]),  //
                          static_cast<U>(accessor_[2]),  //
                          static_cast<U>(accessor_[3])};
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
  constexpr void set(const T &w, const T &x, const T &y, const T &z)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    accessor_[0] = w;
    accessor_[1] = x;
    accessor_[2] = y;
    accessor_[3] = z;
  }

  /// \brief Set all elements of the quaternion
  /// \param[in] w The scalar component.
  /// \param[in] vec The vector component.
  KOKKOS_INLINE_FUNCTION
  constexpr void set(const T &w, const AVector3<T> &vec)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    accessor_[0] = w;
    accessor_[1] = vec[0];
    accessor_[2] = vec[1];
    accessor_[3] = vec[2];
  }

  /// \brief Set all elements of the vector using an accessor
  /// \param[in] accessor A valid accessor.
  /// \note A AQuaternion is also a valid accessor.
  template <ValidAccessor<T> OtherAccessor>
  KOKKOS_INLINE_FUNCTION constexpr void set(const OtherAccessor &accessor)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    accessor_[0] = accessor[0];
    accessor_[1] = accessor[1];
    accessor_[2] = accessor[2];
    accessor_[3] = accessor[3];
  }

  /// \brief Set the quaternion vector component
  /// \param[in] vec The vector.
  KOKKOS_INLINE_FUNCTION
  constexpr void set_vector(const AVector3<T> &vec)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    accessor_[1] = vec[0];
    accessor_[2] = vec[1];
    accessor_[3] = vec[2];
  }

  /// \brief Normalize the quaternion in place
  KOKKOS_INLINE_FUNCTION
  constexpr void normalize()
    requires HasNonConstAccessOperator<Accessor, T>
  {
    const T inv_norm = T(1) / norm(*this);
    accessor_[0] *= inv_norm;
    accessor_[1] *= inv_norm;
    accessor_[2] *= inv_norm;
    accessor_[3] *= inv_norm;
  }

  /// \brief Conjugate the quaternion in place
  KOKKOS_INLINE_FUNCTION
  constexpr void conjugate()
    requires HasNonConstAccessOperator<Accessor, T>
  {
    accessor_[1] = -accessor_[1];
    accessor_[2] = -accessor_[2];
    accessor_[3] = -accessor_[3];
  }

  /// \brief Invert the quaternion in place
  KOKKOS_INLINE_FUNCTION
  constexpr void invert()
    requires HasNonConstAccessOperator<Accessor, T>
  {
    const T inv_norm_squared = T(1) / (accessor_[0] * accessor_[0] + accessor_[1] * accessor_[1] +
                                       accessor_[2] * accessor_[2] + accessor_[3] * accessor_[3]);
    conjugate();
    accessor_[0] *= inv_norm_squared;
    accessor_[1] *= inv_norm_squared;
    accessor_[2] *= inv_norm_squared;
    accessor_[3] *= inv_norm_squared;
  }
  //@}

  //! \name Unary operators
  //@{

  /// \brief Unary plus operator
  KOKKOS_INLINE_FUNCTION
  constexpr AQuaternion<T> operator+() const {
    return AQuaternion<T>(+accessor_[0], +accessor_[1], +accessor_[2], +accessor_[3]);
  }

  /// \brief Unary minus operator
  KOKKOS_INLINE_FUNCTION
  constexpr AQuaternion<T> operator-() const {
    return AQuaternion<T>(-accessor_[0], -accessor_[1], -accessor_[2], -accessor_[3]);
  }
  //@}

  //! \name Addition and subtraction
  //@{

  /// \brief AQuaternion-quaternion addition
  /// \param[in] other The other quaternion.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr auto operator+(
      const AQuaternion<U, OtherAccessor, OtherOwnershipType> &other) const {
    return impl::quat_quat_addition_impl(*this, other);
  }

  /// \brief AQuaternion-quaternion addition
  /// \param[in] other The other quaternion.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr AQuaternion<T, Accessor, Ownership::Views> &operator+=(
      const AQuaternion<U, OtherAccessor, OtherOwnershipType> &other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_quat_addition_impl(*this, other);
    return *this;
  }

  /// \brief AQuaternion-quaternion subtraction
  /// \param[in] other The other quaternion.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr auto operator-(
      const AQuaternion<U, OtherAccessor, OtherOwnershipType> &other) const {
    return impl::quat_quat_subtraction_impl(*this, other);
  }

  /// \brief AQuaternion-quaternion subtraction
  /// \param[in] other The other quaternion.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr AQuaternion<T, Accessor, Ownership::Views> &operator-=(
      const AQuaternion<U, OtherAccessor, OtherOwnershipType> &other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_quat_subtraction_impl(*this, other);
    return *this;
  }
  //@}

  //! \name Multiplication and division
  //@{

  /// \brief AQuaternion-quaternion multiplication
  /// \param[in] other The other quaternion.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr auto operator*(
      const AQuaternion<U, OtherAccessor, OtherOwnershipType> &other) const {
    return impl::quat_quat_multiplication_impl(*this, other);
  }

  /// \brief AQuaternion-quaternion multiplication
  /// \param[in] other The other quaternion.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr AQuaternion<T, Accessor, Ownership::Views> &operator*=(
      const AQuaternion<U, OtherAccessor, OtherOwnershipType> &other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_quat_multiplication_impl(*this, other);
    return *this;
  }

  /// \brief AQuaternion-vector multiplication (same as R * v)
  /// \param[in] vec The vector.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr auto operator*(const AVector3<U, OtherAccessor, OtherOwnershipType> &vec) const {
    return impl::quat_vec_multiplication_impl(*this, vec);
  }

  /// \brief AQuaternion-matrix multiplication
  /// \param[in] other The other matrix.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr auto operator*(const AMatrix3<U, OtherAccessor, OtherOwnershipType> &mat) const {
    return impl::quat_mat_multiplication_impl(*this, mat);
  }

  /// \brief AQuaternion-scalar multiplication
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr auto operator*(const U &scalar) const {
    return impl::quat_scalar_multiplication_impl(*this, scalar);
  }

  /// \brief Self-scalar multiplication
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr AQuaternion<T, Accessor, Ownership::Views> &operator*=(const U &scalar)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_scalar_multiplication_impl(*this, scalar);
    return *this;
  }

  /// \brief AQuaternion-scalar division
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr auto operator/(const U &scalar) const {
    return impl::quat_scalar_division_impl(*this, scalar);
  }

  /// \brief Self-scalar division
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr AQuaternion<T, Accessor, Ownership::Views> &operator/=(const U &scalar)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_scalar_division_impl(*this, scalar);
    return *this;
  }
  //@}

  //! \name Friends <3
  //@{

  // Declare the << operator as a friend
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  friend std::ostream &operator<<(std::ostream &os, const AQuaternion<U, OtherAccessor, OtherOwnershipType> &quat);

  // We are friends with all AQuaternions regardless of their Accessor or type
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
    requires std::is_floating_point_v<U>
  friend class AQuaternion;
  //@}
};  // AQuaternion (non-owning)

template <typename T, ValidAccessor<T> Accessor>
  requires std::is_floating_point_v<T>
class AQuaternion<T, Accessor, Ownership::Owns> {
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
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor. Assume elements are uninitialized.
  /// \note This constructor is only enabled if the Accessor has a default constructor.
  KOKKOS_INLINE_FUNCTION constexpr AQuaternion()
    requires HasDefaultConstructor<Accessor>
      : accessor_() {
  }

  /// \brief Constructor from a given accessor
  /// \param[in] data The accessor.
  KOKKOS_INLINE_FUNCTION
  explicit constexpr AQuaternion(const Accessor &data)
    requires std::is_copy_constructible_v<Accessor>
      : accessor_(data) {
  }

  /// \brief Constructor to initialize all elements
  /// \param[in] w The scalar component.
  /// \param[in] x The x component.
  /// \param[in] y The y component.
  /// \param[in] z The z component.
  /// \note This constructor is only enabled if the Accessor has a 3-argument constructor.
  KOKKOS_INLINE_FUNCTION constexpr AQuaternion(const T &w, const T &x, const T &y, const T &z)
    requires HasNArgConstructor<Accessor, T, 4>
      : accessor_(w, x, y, z) {
  }

  /// \brief Constructor to initialize all elements via initializer list
  /// \param[in] list The initializer list.
  KOKKOS_INLINE_FUNCTION constexpr AQuaternion(const std::initializer_list<T> &list)
    requires HasInitializerListConstructor<Accessor, T>
      : accessor_(list) {
    MUNDY_THROW_ASSERT(list.size() == 4, std::invalid_argument, "AQuaternion: Initializer list must have 4 elements.");
  }

  /// \brief Destructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr ~AQuaternion() = default;

  // Default copy/move constructors and assignment operators when interacting with a AQuaternion of the same type

  /// \brief Default copy constructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr AQuaternion(const AQuaternion<T, Accessor, Ownership::Owns> &) = default;

  /// \brief Default move constructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr AQuaternion(AQuaternion<T, Accessor, Ownership::Owns> &&) = default;

  /// \brief Default copy assignment operator
  KOKKOS_DEFAULTED_FUNCTION
  constexpr AQuaternion<T, Accessor, Ownership::Owns> &operator=(const AQuaternion<T, Accessor, Ownership::Owns> &) =
      default;

  /// \brief Default move assignment operator
  KOKKOS_DEFAULTED_FUNCTION
  constexpr AQuaternion<T, Accessor, Ownership::Owns> &operator=(AQuaternion<T, Accessor, Ownership::Owns> &&) =
      default;

  // Custom copy/move constructors and assignment operators when interacting with a AQuaternion of a different type

  /// \brief Deep copy constructor with different accessor or ownership
  template <ValidQuaternionType OtherQuaternionType>
  KOKKOS_INLINE_FUNCTION constexpr AQuaternion(const OtherQuaternionType &other)
    requires(!std::is_same_v<OtherQuaternionType, AQuaternion<T, Accessor, Ownership::Owns>>) &&
            (std::is_same_v<typename OtherQuaternionType::scalar_t, T>)
      : accessor_() {
    impl::deep_copy_impl(*this, other);
  }

  /// \brief Deep move constructor with different accessor or ownership
  template <ValidQuaternionType OtherQuaternionType>
  KOKKOS_INLINE_FUNCTION constexpr AQuaternion(OtherQuaternionType &&other)
    requires(!std::is_same_v<OtherQuaternionType, AQuaternion<T, Accessor, Ownership::Owns>>) &&
            (std::is_same_v<typename OtherQuaternionType::scalar_t, T>)
      : accessor_() {
    impl::deep_copy_impl(*this, std::move(other));
  }

  /// \brief Deep copy assignment operator with different accessor or ownership
  /// \details Copies the data from the other vector to our data. This is only enabled if T is not const.
  template <ValidQuaternionType OtherQuaternionType>
  KOKKOS_INLINE_FUNCTION constexpr AQuaternion<T, Accessor, Ownership::Owns> &operator=(
      const OtherQuaternionType &other)
    requires(!std::is_same_v<OtherQuaternionType, AQuaternion<T, Accessor, Ownership::Owns>>) &&
            (std::is_same_v<typename OtherQuaternionType::scalar_t, T>) && HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(*this, other);
    return *this;
  }

  /// \brief Deep move assignment operator with different accessor or ownership
  /// \details Moves the data from the other vector to our data. This is only enabled if T is not const.
  template <ValidQuaternionType OtherQuaternionType>
  KOKKOS_INLINE_FUNCTION constexpr AQuaternion<T, Accessor, Ownership::Owns> &operator=(OtherQuaternionType &&other)
    requires(!std::is_same_v<OtherQuaternionType, AQuaternion<T, Accessor, Ownership::Owns>>) &&
            (std::is_same_v<typename OtherQuaternionType::scalar_t, T>) && HasNonConstAccessOperator<Accessor, T>
  {
    impl::deep_copy_impl(*this, std::move(other));
    return *this;
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Element access operator via a single index
  /// \param[in] index The index of the element.
  KOKKOS_INLINE_FUNCTION
  constexpr T &operator[](int index) {
    return accessor_[index];
  }

  /// \brief Const element access operator via a single index
  /// \param[in] index The index of the element.
  KOKKOS_INLINE_FUNCTION
  constexpr const T &operator[](int index) const {
    return accessor_[index];
  }

  /// \brief Element access operator via a single index
  /// \param[in] index The index of the element.
  KOKKOS_INLINE_FUNCTION
  constexpr T &operator()(int index) {
    return accessor_[index];
  }

  /// \brief Const element access operator via a single index
  /// \param[in] index The index of the element.
  KOKKOS_INLINE_FUNCTION
  constexpr const T &operator()(int index) const {
    return accessor_[index];
  }

  /// \brief Get a reference to the scalar component
  KOKKOS_INLINE_FUNCTION
  constexpr T &w() {
    return accessor_[0];
  }

  /// \brief Get a reference to the scalar component
  KOKKOS_INLINE_FUNCTION
  constexpr const T &w() const {
    return accessor_[0];
  }

  /// \brief Get a reference to the x component
  KOKKOS_INLINE_FUNCTION
  constexpr T &x() {
    return accessor_[1];
  }

  /// \brief Get a reference to the x component
  KOKKOS_INLINE_FUNCTION
  constexpr const T &x() const {
    return accessor_[1];
  }

  /// \brief Get a reference to the y component
  KOKKOS_INLINE_FUNCTION
  constexpr T &y() {
    return accessor_[2];
  }

  /// \brief Get a reference to the y component
  KOKKOS_INLINE_FUNCTION
  constexpr const T &y() const {
    return accessor_[2];
  }

  /// \brief Get a reference to the z component
  KOKKOS_INLINE_FUNCTION
  constexpr T &z() {
    return accessor_[3];
  }

  /// \brief Get a reference to the z component
  KOKKOS_INLINE_FUNCTION
  constexpr const T &z() const {
    return accessor_[3];
  }

  /// \brief Get the internal data accessor
  KOKKOS_INLINE_FUNCTION
  constexpr Accessor &data() {
    return accessor_;
  }

  /// \brief Get the internal data accessor
  KOKKOS_INLINE_FUNCTION
  constexpr const Accessor &data() const {
    return accessor_;
  }

  /// \brief Get a view of the quaternion vector component
  KOKKOS_INLINE_FUNCTION
  constexpr const auto vector() const {
    auto shifted_accessor = get_shifted_view<T, 1>(accessor_);
    return get_owning_vector<T, 3>(std::move(shifted_accessor));
  }

  /// \brief Get a view of the quaternion vector component
  KOKKOS_INLINE_FUNCTION
  constexpr auto vector() {
    auto shifted_accessor = get_shifted_view<T, 1>(accessor_);
    return get_owning_vector<T, 3>(std::move(shifted_accessor));
  }

  /// \brief Cast (and copy) the quaternion to a different type
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr auto cast() const {
    return AQuaternion<U>{static_cast<U>(accessor_[0]),  //
                          static_cast<U>(accessor_[1]),  //
                          static_cast<U>(accessor_[2]),  //
                          static_cast<U>(accessor_[3])};
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
  constexpr void set(const T &w, const T &x, const T &y, const T &z)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    accessor_[0] = w;
    accessor_[1] = x;
    accessor_[2] = y;
    accessor_[3] = z;
  }

  /// \brief Set all elements of the quaternion
  /// \param[in] w The scalar component.
  /// \param[in] vec The vector component.
  KOKKOS_INLINE_FUNCTION
  constexpr void set(const T &w, const AVector3<T> &vec)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    accessor_[0] = w;
    accessor_[1] = vec[0];
    accessor_[2] = vec[1];
    accessor_[3] = vec[2];
  }

  /// \brief Set all elements of the vector using an accessor
  /// \param[in] accessor A valid accessor.
  /// \note A AQuaternion is also a valid accessor.
  template <ValidAccessor<T> OtherAccessor>
  KOKKOS_INLINE_FUNCTION constexpr void set(const OtherAccessor &accessor)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    accessor_[0] = accessor[0];
    accessor_[1] = accessor[1];
    accessor_[2] = accessor[2];
    accessor_[3] = accessor[3];
  }

  /// \brief Set the quaternion vector component
  /// \param[in] vec The vector.
  KOKKOS_INLINE_FUNCTION
  constexpr void set_vector(const AVector3<T> &vec)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    accessor_[1] = vec[0];
    accessor_[2] = vec[1];
    accessor_[3] = vec[2];
  }

  /// \brief Normalize the quaternion in place
  KOKKOS_INLINE_FUNCTION
  constexpr void normalize()
    requires HasNonConstAccessOperator<Accessor, T>
  {
    const T inv_norm = T(1) / norm(*this);
    accessor_[0] *= inv_norm;
    accessor_[1] *= inv_norm;
    accessor_[2] *= inv_norm;
    accessor_[3] *= inv_norm;
  }

  /// \brief Conjugate the quaternion in place
  KOKKOS_INLINE_FUNCTION
  constexpr void conjugate()
    requires HasNonConstAccessOperator<Accessor, T>
  {
    accessor_[1] = -accessor_[1];
    accessor_[2] = -accessor_[2];
    accessor_[3] = -accessor_[3];
  }

  /// \brief Invert the quaternion in place
  KOKKOS_INLINE_FUNCTION
  constexpr void invert()
    requires HasNonConstAccessOperator<Accessor, T>
  {
    const T inv_norm_squared = T(1) / (accessor_[0] * accessor_[0] + accessor_[1] * accessor_[1] +
                                       accessor_[2] * accessor_[2] + accessor_[3] * accessor_[3]);
    conjugate();
    accessor_[0] *= inv_norm_squared;
    accessor_[1] *= inv_norm_squared;
    accessor_[2] *= inv_norm_squared;
    accessor_[3] *= inv_norm_squared;
  }
  //@}

  //! \name Unary operators
  //@{

  /// \brief Unary plus operator
  KOKKOS_INLINE_FUNCTION
  constexpr AQuaternion<T> operator+() const {
    return AQuaternion<T>(+accessor_[0], +accessor_[1], +accessor_[2], +accessor_[3]);
  }

  /// \brief Unary minus operator
  KOKKOS_INLINE_FUNCTION
  constexpr AQuaternion<T> operator-() const {
    return AQuaternion<T>(-accessor_[0], -accessor_[1], -accessor_[2], -accessor_[3]);
  }
  //@}

  //! \name Addition and subtraction
  //@{

  /// \brief AQuaternion-quaternion addition
  /// \param[in] other The other quaternion.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr auto operator+(
      const AQuaternion<U, OtherAccessor, OtherOwnershipType> &other) const {
    return impl::quat_quat_addition_impl(*this, other);
  }

  /// \brief AQuaternion-quaternion addition
  /// \param[in] other The other quaternion.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr AQuaternion<T, Accessor, Ownership::Owns> &operator+=(
      const AQuaternion<U, OtherAccessor, OtherOwnershipType> &other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_quat_addition_impl(*this, other);
    return *this;
  }

  /// \brief AQuaternion-quaternion subtraction
  /// \param[in] other The other quaternion.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr auto operator-(
      const AQuaternion<U, OtherAccessor, OtherOwnershipType> &other) const {
    return impl::quat_quat_subtraction_impl(*this, other);
  }

  /// \brief AQuaternion-quaternion subtraction
  /// \param[in] other The other quaternion.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr AQuaternion<T, Accessor, Ownership::Owns> &operator-=(
      const AQuaternion<U, OtherAccessor, OtherOwnershipType> &other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_quat_subtraction_impl(*this, other);
    return *this;
  }
  //@}

  //! \name Multiplication and division
  //@{

  /// \brief AQuaternion-quaternion multiplication
  /// \param[in] other The other quaternion.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr auto operator*(
      const AQuaternion<U, OtherAccessor, OtherOwnershipType> &other) const {
    return impl::quat_quat_multiplication_impl(*this, other);
  }

  /// \brief AQuaternion-quaternion multiplication
  /// \param[in] other The other quaternion.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr AQuaternion<T, Accessor, Ownership::Owns> &operator*=(
      const AQuaternion<U, OtherAccessor, OtherOwnershipType> &other)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_quat_multiplication_impl(*this, other);
    return *this;
  }

  /// \brief AQuaternion-vector multiplication (same as R * v)
  /// \param[in] vec The vector.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr auto operator*(const AVector3<U, OtherAccessor, OtherOwnershipType> &vec) const {
    return impl::quat_vec_multiplication_impl(*this, vec);
  }

  /// \brief AQuaternion-matrix multiplication
  /// \param[in] other The other matrix.
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  KOKKOS_INLINE_FUNCTION constexpr auto operator*(const AMatrix3<U, OtherAccessor, OtherOwnershipType> &mat) const {
    return impl::quat_mat_multiplication_impl(*this, mat);
  }

  /// \brief AQuaternion-scalar multiplication
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr auto operator*(const U &scalar) const {
    return impl::quat_scalar_multiplication_impl(*this, scalar);
  }

  /// \brief Self-scalar multiplication
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr AQuaternion<T, Accessor, Ownership::Owns> &operator*=(const U &scalar)
    requires HasNonConstAccessOperator<Accessor, T>
  {
    impl::self_scalar_multiplication_impl(*this, scalar);
    return *this;
  }

  /// \brief AQuaternion-scalar division
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr auto operator/(const U &scalar) const {
    return impl::quat_scalar_division_impl(*this, scalar);
  }

  /// \brief Self-scalar division
  /// \param[in] scalar The scalar.
  template <typename U>
  KOKKOS_INLINE_FUNCTION constexpr AQuaternion<T, Accessor, Ownership::Owns> &operator/=(const U &scalar)
    requires HasNonConstAccessOperator<Accessor, T>
  {
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
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
  friend std::ostream &operator<<(std::ostream &os, const AQuaternion<U, OtherAccessor, OtherOwnershipType> &quat);

  // We are friends with all AQuaternions regardless of their Accessor or type
  template <typename U, ValidAccessor<U> OtherAccessor, typename OtherOwnershipType>
    requires std::is_floating_point_v<U>
  friend class AQuaternion;
  //@}
};  // AQuaternion (owning)

template <typename T, ValidAccessor<T> Accessor = Array<T, 4>>
  requires std::is_floating_point_v<T>
using QuaternionView = AQuaternion<T, Accessor, Ownership::Views>;

template <typename T, ValidAccessor<T> Accessor = Array<T, 4>>
  requires std::is_floating_point_v<T>
using OwningQuaternion = AQuaternion<T, Accessor, Ownership::Owns>;

template <typename T>
  requires std::is_floating_point_v<T>
using Quaternion = OwningQuaternion<T, Array<T, 4>>;

static_assert(is_quaternion_v<AQuaternion<double>>, "Odd, default AQuaternion is not a quaternion.");
static_assert(is_quaternion_v<AQuaternion<double, Array<double, 4>>>,
              "Odd, default AQuaternion with Array accessor is not a quaternion.");
static_assert(is_quaternion_v<QuaternionView<double>>, "Odd, QuaternionView is not a quaternion.");
static_assert(is_quaternion_v<Quaternion<double>>, "Odd, AQuaternionAVector is not a quaternion.");

//! \name Non-member functions
//@{

//! \name Write to output stream
//@{

/// \brief Write the quaternion to an output stream
/// \param[in] os The output stream.
/// \param[in] quat The quaternion.
template <typename T, ValidAccessor<T> Accessor, typename OwnershipType>
std::ostream &operator<<(std::ostream &os, const AQuaternion<T, Accessor, OwnershipType> &quat) {
  os << "(" << quat[0] << ", " << quat[1] << ", " << quat[2] << ", " << quat[3] << ")";
  return os;
}
//@}

//! \name Non-member comparison functions
//@{

/// \brief AQuaternion-quaternion equality (element-wise within a tolerance)
/// \param[in] quat1 The first quaternion.
/// \param[in] quat2 The second quaternion.
/// \param[in] tol The tolerance.
template <typename U, typename T, ValidAccessor<U> Accessor1, typename OwnershipType1, ValidAccessor<T> Accessor2,
          typename OwnershipType2>
KOKKOS_INLINE_FUNCTION constexpr bool is_close(
    const AQuaternion<U, Accessor1, OwnershipType1> &quat1, const AQuaternion<T, Accessor2, OwnershipType2> &quat2,
     const decltype(get_comparison_tolerance<T, U>())& tol = get_comparison_tolerance<T, U>()) {
  using Tol = decltype(tol);
  return Kokkos::abs(static_cast<Tol>(quat1[0]) - static_cast<Tol>(quat2[0])) <= tol &&
         Kokkos::abs(static_cast<Tol>(quat1[1]) - static_cast<Tol>(quat2[1])) <= tol &&
         Kokkos::abs(static_cast<Tol>(quat1[2]) - static_cast<Tol>(quat2[2])) <= tol &&
         Kokkos::abs(static_cast<Tol>(quat1[3]) - static_cast<Tol>(quat2[3])) <= tol;
}

/// \brief AQuaternion-quaternion equality (element-wise within a relaxed tolerance)
/// \param[in] quat1 The first quaternion.
/// \param[in] quat2 The second quaternion.
/// \param[in] tol The tolerance.
template <typename U, typename T, ValidAccessor<U> Accessor1, typename OwnershipType1, ValidAccessor<T> Accessor2,
          typename OwnershipType2>
KOKKOS_INLINE_FUNCTION constexpr bool is_approx_close(
    const AQuaternion<U, Accessor1, OwnershipType1> &quat1, const AQuaternion<T, Accessor2, OwnershipType2> &quat2,
     const decltype(get_relaxed_comparison_tolerance<T, U>())& tol = get_relaxed_comparison_tolerance<T, U>()) {
  return is_close(quat1, quat2, tol);
}
//@}

//! \name Non-member multiplication and division operators
//@{

/// \brief Scalar-quaternion multiplication
/// \param[in] scalar The scalar.
/// \param[in] quat The quaternion.
template <typename U, typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto operator*(const U &scalar, const AQuaternion<T, Accessor, OwnershipType> &quat)
    -> AQuaternion<std::common_type_t<T, U>> {
  return quat * scalar;
}

/// \brief AVector-quaternion multiplication (same as v^T * R = transpose(R^T * v))
/// \param[in] vec The vector.
/// \param[in] quat The quaternion.
template <typename U, typename T, ValidAccessor<U> Accessor1, typename OwnershipType1, ValidAccessor<T> Accessor2,
          typename OwnershipType2>
KOKKOS_INLINE_FUNCTION constexpr auto operator*(const AVector3<U, Accessor1, OwnershipType1> &vec,
                                                const AQuaternion<T, Accessor2, OwnershipType2> &quat)
    -> AVector3<std::common_type_t<T, U>> {
  return impl::vec_quat_multiplication_impl(vec, quat);
}

/// \brief AMatrix-quaternion multiplication (same as R * M)
/// \param[in] mat The matrix.
/// \param[in] quat The quaternion.
template <typename U, typename T, ValidAccessor<U> Accessor1, typename OwnershipType1, ValidAccessor<T> Accessor2,
          typename OwnershipType2>
KOKKOS_INLINE_FUNCTION constexpr auto operator*(const AMatrix3<U, Accessor1, OwnershipType1> &mat,
                                                const AQuaternion<T, Accessor2, OwnershipType2> &quat) {
  return impl::mat_quat_multiplication_impl(mat, quat);
}
//@}

//! \name Special quaternion operations
//@{

/// \brief Get the dot product of two quaternions
/// \param[in] q1 The first quaternion.
/// \param[in] q2 The second quaternion.
template <typename U, typename T, ValidAccessor<U> Accessor1, typename OwnershipType1, ValidAccessor<T> Accessor2,
          typename OwnershipType2>
KOKKOS_INLINE_FUNCTION constexpr auto dot(const AQuaternion<U, Accessor1, OwnershipType1> &q1,
                                          const AQuaternion<T, Accessor2, OwnershipType2> &q2) {
  using CommonType = std::common_type_t<U, T>;
  return static_cast<CommonType>(q1[0]) * static_cast<CommonType>(q2[0]) +
         static_cast<CommonType>(q1[1]) * static_cast<CommonType>(q2[1]) +
         static_cast<CommonType>(q1[2]) * static_cast<CommonType>(q2[2]) +
         static_cast<CommonType>(q1[3]) * static_cast<CommonType>(q2[3]);
}

/// \brief Get the conjugate of a quaternion
/// \param[in] quat The quaternion.
template <typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr AQuaternion<std::remove_const_t<T>> conjugate(
    const AQuaternion<T, Accessor, OwnershipType> &quat) {
  AQuaternion<std::remove_const_t<T>> result;
  result[0] = quat[0];
  result[1] = -quat[1];
  result[2] = -quat[2];
  result[3] = -quat[3];
  return result;
}

/// \brief Get the inverse of a quaternion
/// \param[in] quat The quaternion.
template <typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr AQuaternion<std::remove_const_t<T>> inverse(
    const AQuaternion<T, Accessor, OwnershipType> &quat) {
  const T inv_norm_squared = T(1) / (quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]);
  return conjugate(quat) * inv_norm_squared;
}

/// \brief Get the norm of a quaternion
/// \param[in] quat The quaternion.
template <typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto norm(const AQuaternion<T, Accessor, OwnershipType> &quat) {
  return std::sqrt(quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]);
}

/// \brief Get the squared norm of a quaternion
/// \param[in] quat The quaternion.
template <typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr auto norm_squared(const AQuaternion<T, Accessor, OwnershipType> &quat) {
  return quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3];
}

/// \brief Get the normalized quaternion
/// \param[in] quat The quaternion.
template <typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr AQuaternion<std::remove_const_t<T>> normalize(
    const AQuaternion<T, Accessor, OwnershipType> &quat) {
  const T inv_norm = static_cast<T>(1) / norm(quat);
  return quat * inv_norm;
}

/// \brief Perform spherical linear interpolation between two quaternions
/// \param[in] q1 The first quaternion.
/// \param[in] q2 The second quaternion.
/// \param[in] t The interpolation parameter.
template <typename U, typename T, typename V, ValidAccessor<U> Accessor1, typename OwnershipType1,
          ValidAccessor<T> Accessor2, typename OwnershipType2>
  requires std::is_arithmetic_v<V>
KOKKOS_INLINE_FUNCTION constexpr auto slerp(const AQuaternion<U, Accessor1, OwnershipType1> &q1,
                                            const AQuaternion<T, Accessor2, OwnershipType2> &q2, const V t)
    -> AQuaternion<std::common_type_t<U, T, V>> {
  using CommonType = std::common_type_t<U, T, V>;
  const CommonType epsilon = get_relaxed_zero_tolerance<CommonType>();  // Threshold for linear interpolation

  // Compute the dot product
  CommonType dot_q12 = dot(q1, q2);

  // Adjust second quaternion for negative dot product
  // Note, we cannot directly copy from q2 to q2_adjusted because the Accessor type may be different.
  AQuaternion<std::remove_const_t<T>> q2_adjusted;
  q2_adjusted.set(q2);
  if (dot_q12 < 0) {
    dot_q12 = -dot_q12;
    q2_adjusted *= -1;
  }

  // Check for near-parallel case
  if (static_cast<CommonType>(1) - dot_q12 < epsilon) {
    // Linear Interpolation as fallback
    return AQuaternion<CommonType>{
        static_cast<CommonType>(q1[0]) +
            static_cast<CommonType>(t) * (static_cast<CommonType>(q2_adjusted[0]) - static_cast<CommonType>(q1[0])),
        static_cast<CommonType>(q1[1]) +
            static_cast<CommonType>(t) * (static_cast<CommonType>(q2_adjusted[1]) - static_cast<CommonType>(q1[1])),
        static_cast<CommonType>(q1[2]) +
            static_cast<CommonType>(t) * (static_cast<CommonType>(q2_adjusted[2]) - static_cast<CommonType>(q1[2])),
        static_cast<CommonType>(q1[3]) +
            static_cast<CommonType>(t) * (static_cast<CommonType>(q2_adjusted[3]) - static_cast<CommonType>(q1[3]))};
  } else {
    // Spherical Interpolation
    const CommonType theta = std::acos(dot_q12);
    const CommonType sin_theta = std::sin(theta);
    const CommonType inv_sin_theta = static_cast<CommonType>(1) / sin_theta;
    const CommonType s1 = std::sin((static_cast<CommonType>(1) - static_cast<CommonType>(t)) * theta) * inv_sin_theta;
    const CommonType s2 = std::sin(static_cast<CommonType>(t) * theta) * inv_sin_theta;

    return AQuaternion<CommonType>{(static_cast<CommonType>(s1) * static_cast<CommonType>(q1[0])) +
                                       (static_cast<CommonType>(s2) * static_cast<CommonType>(q2_adjusted[0])),
                                   (static_cast<CommonType>(s1) * static_cast<CommonType>(q1[1])) +
                                       (static_cast<CommonType>(s2) * static_cast<CommonType>(q2_adjusted[1])),
                                   (static_cast<CommonType>(s1) * static_cast<CommonType>(q1[2])) +
                                       (static_cast<CommonType>(s2) * static_cast<CommonType>(q2_adjusted[2])),
                                   (static_cast<CommonType>(s1) * static_cast<CommonType>(q1[3])) +
                                       (static_cast<CommonType>(s2) * static_cast<CommonType>(q2_adjusted[3]))};
  }
}

// /// \brief Perform spherical linear interpolation between two quaternions
// /// Source: https://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/slerp/index.htm
// /// \param[in] q1 The first quaternion.
// /// \param[in] q2 The second quaternion.
// /// \param[in] t The interpolation parameter.
// template <typename U, typename T, typename V>
//   requires std::is_arithmetic_v<V>
// template <typename U, typename T, typename V, ValidAccessor<U> Accessor1, typename OwnershipType1, ValidAccessor<T>
// Accessor2,
//           typename OwnershipType2>
//   requires std::is_arithmetic_v<V>
// KOKKOS_INLINE_FUNCTION constexpr auto slerp(const AQuaternion<U, Accessor1, OwnershipType1> &q1, const AQuaternion<T,
// Accessor2, OwnershipType2> &q2,
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
KOKKOS_INLINE_FUNCTION constexpr void rotate_quaternion(QuaternionType &quat, const VectorType &omega,
                                                        const double &dt) {
  const double w = norm(omega);
  if (w < get_zero_tolerance<double>()) {
    // Omega is zero, no rotation
    return;
  }
  const double winv = 1.0 / w;
  const double sw = Kokkos::sin(0.5 * w * dt);
  const double cw = Kokkos::cos(0.5 * w * dt);
  const double s = quat.w();
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
template <typename T, typename U, ValidAccessor<T> Accessor, typename OwnershipType>
  requires std::is_arithmetic_v<U>
KOKKOS_INLINE_FUNCTION constexpr auto axis_angle_to_quaternion(const AVector3<T, Accessor, OwnershipType> &axis,
                                                               const U &angle)
    -> AQuaternion<std::common_type_t<T, U>> {
  using CommonType = std::common_type_t<T, U>;
  const auto half_angle = U(0.5) * angle;
  const auto sin_half_angle = std::sin(half_angle);
  const auto cos_half_angle = std::cos(half_angle);
  return AQuaternion<CommonType>(static_cast<CommonType>(cos_half_angle),
                                 static_cast<CommonType>(sin_half_angle) * static_cast<CommonType>(axis[0]),
                                 static_cast<CommonType>(sin_half_angle) * static_cast<CommonType>(axis[1]),
                                 static_cast<CommonType>(sin_half_angle) * static_cast<CommonType>(axis[2]));
}

/// \brief Get the quaternion from a rotation matrix
/// \param[in] rot_mat The rotation matrix.
template <typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr AQuaternion<T> rotation_matrix_to_quaternion(
    const AMatrix3<T, Accessor, OwnershipType> &rot_mat) {
  // Source: https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToAQuaternion/
  AQuaternion<T> quat;

  // Computing the quaternion components
  quat.w() = Kokkos::sqrt(Kokkos::max(T(0), T(1) + rot_mat(0, 0) + rot_mat(1, 1) + rot_mat(2, 2))) / T(2);
  quat.x() = Kokkos::sqrt(Kokkos::max(T(0), T(1) + rot_mat(0, 0) - rot_mat(1, 1) - rot_mat(2, 2))) / T(2);
  quat.y() = Kokkos::sqrt(Kokkos::max(T(0), T(1) - rot_mat(0, 0) + rot_mat(1, 1) - rot_mat(2, 2))) / T(2);
  quat.z() = Kokkos::sqrt(Kokkos::max(T(0), T(1) - rot_mat(0, 0) - rot_mat(1, 1) + rot_mat(2, 2))) / T(2);

  // Correcting the signs
  quat.x() = std::copysign(quat[1], rot_mat(2, 1) - rot_mat(1, 2));
  quat.y() = std::copysign(quat[2], rot_mat(0, 2) - rot_mat(2, 0));
  quat.z() = std::copysign(quat[3], rot_mat(1, 0) - rot_mat(0, 1));

  return quat;
}

/// \brief Get the rotation matrix from a quaternion
/// \param[in] quat The quaternion.
template <typename T, ValidAccessor<T> Accessor, typename OwnershipType>
KOKKOS_INLINE_FUNCTION constexpr AMatrix3<std::remove_const_t<T>> quaternion_to_rotation_matrix(
    const AQuaternion<T, Accessor, OwnershipType> &quat) {
  AMatrix3<std::remove_const_t<T>> rot_mat;
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
  requires std::is_arithmetic_v<T>
KOKKOS_INLINE_FUNCTION constexpr AQuaternion<std::remove_const_t<T>> euler_to_quat(const T roll, const T pitch,
                                                                                   const T yaw) {
  // Convert Euler angles to quaternion
  AQuaternion<std::remove_const_t<T>> quat;
  const T cha1 = std::cos(T(0.5) * roll);
  const T cha2 = std::cos(T(0.5) * pitch);
  const T cha3 = std::cos(T(0.5) * yaw);
  const T sha1 = std::sin(T(0.5) * roll);
  const T sha2 = std::sin(T(0.5) * pitch);
  const T sha3 = std::sin(T(0.5) * yaw);
  quat.w() = cha1 * cha2 * cha3 + sha1 * sha2 * sha3;
  quat.x() = sha1 * cha2 * cha3 - cha1 * sha2 * sha3;
  quat.y() = cha1 * sha2 * cha3 + sha1 * cha2 * sha3;
  quat.z() = cha1 * cha2 * sha3 - sha1 * sha2 * cha3;
  return quat;
}

/// \brief Get the quaternion that perform parallel transport from vector v1 to vector v2
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
template <typename U, typename T, ValidAccessor<U> Accessor1, typename OwnershipType1, ValidAccessor<T> Accessor2,
          typename OwnershipType2>
  requires(std::is_arithmetic_v<T> && std::is_arithmetic_v<U>)
KOKKOS_INLINE_FUNCTION constexpr auto quat_from_parallel_transport(const AVector3<U, Accessor1, OwnershipType1> &v_from,
                                                                   const AVector3<T, Accessor2, OwnershipType2> &v_to)
    -> AQuaternion<decltype(U() * T())> {
  // Get the quaternion that performs parallel transport from vector v_from to vector v_to
  using CommonType = decltype(U() * T());
  AQuaternion<CommonType> quat;

  // Compute the dot product and cross product
  const auto dot_product = dot(v_from, v_to);
  const auto cross_product = cross(v_from, v_to);
  const double sqrt_term = std::sqrt(0.5 * (1.0 + dot_product));
  const auto vec = 0.5 * cross_product / sqrt_term;
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

#define MUNDY_MATH_QUATERNION_TYPE_SPECIALIZATION(alias, alias_lower, T)                       \
  template <ValidAccessor<T> Accessor = Array<T, 4>, typename OwnershipType = Ownership::Owns> \
  using A##alias = AQuaternion<T, Accessor, OwnershipType>;                                    \
  template <ValidAccessor<T> Accessor = Array<T, 4>>                                           \
  using alias##View = AQuaternion<T, Accessor, Ownership::Views>;                              \
  template <ValidAccessor<T> Accessor = Array<T, 4>>                                           \
  using Owning##alias = AQuaternion<T, Accessor, Ownership::Owns>;                             \
  using alias = Owning##alias<>;                                                               \
  template <typename TypeToCheck>                                                              \
  struct is_##alias_lower##_impl : std::false_type {};                                         \
  template <typename Accessor, typename OwnershipType>                                         \
  struct is_##alias_lower##_impl<A##alias<Accessor, OwnershipType>> : std::true_type {};       \
  template <typename TypeToCheck>                                                              \
  struct is_##alias_lower : public is_##alias_lower##_impl<std::decay_t<TypeToCheck>> {};      \
  template <typename TypeToCheck>                                                              \
  constexpr bool is_##alias_lower##_v = is_##alias_lower<TypeToCheck>::value;

// Eigen convention.
MUNDY_MATH_QUATERNION_TYPE_SPECIALIZATION(Quaterniond, quaterniond, double)
MUNDY_MATH_QUATERNION_TYPE_SPECIALIZATION(Quaternionf, quaternionf, float)
//@}

//! \name AQuaternion<T, Accessor> views
//@{

/// \brief A helper function to create a AQuaternion<T, Accessor> based on a given accessor.
/// \param[in] data The data accessor.
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
template <typename T, typename Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_quaternion_view(const Accessor &data) {
  return QuaternionView<T, Accessor>(data);
}

template <typename T, typename Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_quaternion_view(Accessor &&data) {
  return QuaternionView<T, Accessor>(std::move(data));
}

template <typename T, typename Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_owning_quaternion(const Accessor &data) {
  return OwningQuaternion<T, Accessor>(data);
}

template <typename T, typename Accessor>
KOKKOS_INLINE_FUNCTION constexpr auto get_owning_quaternion(Accessor &&data) {
  return OwningQuaternion<T, Accessor>(std::move(data));
}
//@}

//@}

}  // namespace math

}  // namespace mundy

#endif  // MUNDY_MATH_QUATERNION_HPP_
