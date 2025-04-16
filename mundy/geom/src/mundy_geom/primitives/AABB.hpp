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

#ifndef MUNDY_GEOM_PRIMITIVES_AABB_HPP_
#define MUNDY_GEOM_PRIMITIVES_AABB_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core
#include <iostream>
#include <stdexcept>
#include <utility>

// Our libs
#include <mundy_core/throw_assert.hpp>      // for MUNDY_THROW_ASSERT
#include <mundy_geom/primitives/Point.hpp>  // for mundy::geom::Point

namespace mundy {

namespace geom {

template <typename Scalar, ValidPointType MinPointType = Point<Scalar>, ValidPointType MaxPointType = Point<Scalar>>
class AABB {
  static_assert(std::is_same_v<typename MinPointType::scalar_t, Scalar> &&
                    std::is_same_v<typename MaxPointType::scalar_t, Scalar>,
                "The scalar type of the PointType must match the Scalar type.");

 public:
  //! \name Type aliases
  //@{

  /// \brief Our scalar type
  using scalar_t = Scalar;

  /// \brief Our point type for the min corner
  using min_point_t = MinPointType;

  /// \brief Our point type for the max corner
  using max_point_t = MaxPointType;

  /// \brief Our ownership type.
  /// One of three values
  ///  - mundy::math::Ownership::Owns: If both the min and max points own their data.
  ///  - mundy::math::Ownership::Views: If both the min and max points are views into other data.
  ///  - mundy::math::Ownership::Mixed: If the min and max points have different ownership types.
  using OwnershipType = std::conditional_t<
      std::is_same_v<typename min_point_t::ownership_t, mundy::math::Ownership::Owns> &&
          std::is_same_v<typename max_point_t::ownership_t, mundy::math::Ownership::Owns>,
      mundy::math::Ownership::Owns,
      std::conditional_t<std::is_same_v<typename min_point_t::ownership_t, mundy::math::Ownership::Views> &&
                             std::is_same_v<typename max_point_t::ownership_t, mundy::math::Ownership::Views>,
                         mundy::math::Ownership::Views, mundy::math::Ownership::Mixed>>;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor for owning AABBs. Initializes the aabb inside out and as large as possible.
  /// Nothing can be inside this aabb.
  KOKKOS_FUNCTION
  constexpr AABB()
    requires std::is_same_v<OwnershipType, mundy::math::Ownership::Owns>
      : min_corner_(scalar_max(), scalar_max(), scalar_max()), max_corner_(scalar_min(), scalar_min(), scalar_min()) {
  }

  /// \brief No default constructor for viewing/mixed AABBs.
  KOKKOS_FUNCTION
  constexpr AABB()
    requires(!std::is_same_v<OwnershipType, mundy::math::Ownership::Owns>)
  = delete;

  /// \brief Constructor to directly set the min and max corners.
  /// \param[in] min_corner The minimum corner of the aabb.
  /// \param[in] max_corner The maximum corner of the aabb.
  KOKKOS_FUNCTION
  constexpr AABB(const min_point_t& min_corner, const max_point_t& max_corner)
      : min_corner_(min_corner), max_corner_(max_corner) {
  }

  /// \brief Constructor to directly set the min and max corners.
  /// \param[in] min_corner The minimum corner of the aabb.
  /// \param[in] max_corner The maximum corner of the aabb.
  template <ValidPointType OtherPointType1, ValidPointType OtherPointType2>
  KOKKOS_FUNCTION constexpr AABB(const OtherPointType1& min_corner, const OtherPointType2& max_corner)
    requires(!std::is_same_v<OtherPointType1, min_point_t> || !std::is_same_v<OtherPointType2, max_point_t>)
      : min_corner_(min_corner), max_corner_(max_corner) {
  }

  /// \brief Constructor to directly set the min and max corners.
  /// \param[in] x_min The minimum x-coordinate.
  /// \param[in] y_min The minimum y-coordinate.
  /// \param[in] z_min The minimum z-coordinate.
  /// \param[in] x_max The maximum x-coordinate.
  /// \param[in] y_max The maximum y-coordinate.
  /// \param[in] z_max The maximum z-coordinate.
  KOKKOS_FUNCTION
  constexpr AABB(scalar_t x_min, scalar_t y_min, scalar_t z_min, scalar_t x_max, scalar_t y_max, scalar_t z_max)
    requires std::is_same_v<OwnershipType, mundy::math::Ownership::Owns>
      : min_corner_(x_min, y_min, z_min), max_corner_(x_max, y_max, z_max) {
  }

  /// \brief Destructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr ~AABB() = default;

  /// \brief Deep copy constructor
  KOKKOS_FUNCTION
  constexpr AABB(const AABB<scalar_t, min_point_t, max_point_t>& other)
      : min_corner_(other.min_corner_), max_corner_(other.max_corner_) {
  }

  /// \brief Deep copy constructor
  template <typename OtherAABBType>
  KOKKOS_FUNCTION constexpr AABB(const OtherAABBType& other)
    requires(!std::is_same_v<OtherAABBType, AABB<scalar_t, min_point_t, max_point_t>>)
      : min_corner_(other.min_corner_), max_corner_(other.max_corner_) {
  }

  /// \brief Deep move constructor
  KOKKOS_FUNCTION
  constexpr AABB(AABB<scalar_t, min_point_t, max_point_t>&& other)
      : min_corner_(std::move(other.min_corner_)), max_corner_(std::move(other.max_corner_)) {
  }

  /// \brief Deep move constructor
  template <typename OtherAABBType>
  KOKKOS_FUNCTION constexpr AABB(OtherAABBType&& other)
    requires(!std::is_same_v<OtherAABBType, AABB<scalar_t, min_point_t, max_point_t>>)
      : min_corner_(std::move(other.min_corner_)), max_corner_(std::move(other.max_corner_)) {
  }
  //@}

  //! \name Operators
  //@{

  /// \brief Copy assignment operator
  KOKKOS_FUNCTION
  constexpr AABB<scalar_t, min_point_t, max_point_t>& operator=(const AABB<scalar_t, min_point_t, max_point_t>& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    min_corner_ = other.min_corner_;
    max_corner_ = other.max_corner_;
    return *this;
  }

  /// \brief Copy assignment operator
  template <typename OtherAABBType>
  KOKKOS_FUNCTION constexpr AABB<scalar_t, min_point_t, max_point_t>& operator=(const OtherAABBType& other)
    requires(!std::is_same_v<OtherAABBType, AABB<scalar_t, min_point_t, max_point_t>>)
  {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    min_corner_ = other.min_corner_;
    max_corner_ = other.max_corner_;
    return *this;
  }

  /// \brief Move assignment operator
  KOKKOS_FUNCTION
  constexpr AABB<scalar_t, min_point_t, max_point_t>& operator=(AABB<scalar_t, min_point_t, max_point_t>&& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    min_corner_ = std::move(other.min_corner_);
    max_corner_ = std::move(other.max_corner_);
    return *this;
  }

  /// \brief Move assignment operator
  template <typename OtherAABBType>
  KOKKOS_FUNCTION constexpr AABB<scalar_t, min_point_t, max_point_t>& operator=(OtherAABBType&& other)
    requires(!std::is_same_v<OtherAABBType, AABB<scalar_t, min_point_t, max_point_t>>)
  {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    min_corner_ = std::move(other.min_corner_);
    max_corner_ = std::move(other.max_corner_);
    return *this;
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Accesses the AABB as though it were an array of size 6 with the min then max corners.
  KOKKOS_FUNCTION const scalar_t& operator[](const size_t& i) const {
    MUNDY_THROW_ASSERT(i < 6, std::out_of_range, "Index out of range");
    return i < 3 ? min_corner_[i] : max_corner_[i - 3];
  }

  /// \brief Accesses the AABB as though it were an array of size 6 with the min then max corners.
  KOKKOS_FUNCTION scalar_t& operator[](const size_t& i) {
    MUNDY_THROW_ASSERT(i < 6, std::out_of_range, "Index out of range");
    return i < 3 ? min_corner_[i] : max_corner_[i - 3];
  }

  /// \brief Accessor for the minimum corner
  KOKKOS_FUNCTION
  const min_point_t& min_corner() const {
    return min_corner_;
  }

  /// \brief Accessor for the minimum corner
  KOKKOS_FUNCTION
  min_point_t& min_corner() {
    return min_corner_;
  }

  /// \brief Accessor for the maximum corner
  KOKKOS_FUNCTION
  const max_point_t& max_corner() const {
    return max_corner_;
  }

  /// \brief Accessor for the maximum corner
  KOKKOS_FUNCTION
  max_point_t& max_corner() {
    return max_corner_;
  }

  /// \brief Accessor for x_min
  KOKKOS_FUNCTION
  const scalar_t& x_min() const {
    return min_corner_[0];
  }

  /// \brief Accessor for x_min
  KOKKOS_FUNCTION
  scalar_t& x_min() {
    return min_corner_[0];
  }

  /// \brief Accessor for y_min
  KOKKOS_FUNCTION
  const scalar_t& y_min() const {
    return min_corner_[1];
  }

  /// \brief Accessor for y_min
  KOKKOS_FUNCTION
  scalar_t& y_min() {
    return min_corner_[1];
  }

  /// \brief Accessor for z_min
  KOKKOS_FUNCTION
  const scalar_t& z_min() const {
    return min_corner_[2];
  }

  /// \brief Accessor for z_min
  KOKKOS_FUNCTION
  scalar_t& z_min() {
    return min_corner_[2];
  }

  /// \brief Accessor for x_max
  KOKKOS_FUNCTION
  const scalar_t& x_max() const {
    return max_corner_[0];
  }

  /// \brief Accessor for x_max
  KOKKOS_FUNCTION
  scalar_t& x_max() {
    return max_corner_[0];
  }

  /// \brief Accessor for y_max
  KOKKOS_FUNCTION
  const scalar_t& y_max() const {
    return max_corner_[1];
  }

  /// \brief Accessor for y_max
  KOKKOS_FUNCTION
  scalar_t& y_max() {
    return max_corner_[1];
  }

  /// \brief Accessor for z_max
  KOKKOS_FUNCTION
  const scalar_t& z_max() const {
    return max_corner_[2];
  }

  /// \brief Accessor for z_max
  KOKKOS_FUNCTION
  scalar_t& z_max() {
    return max_corner_[2];
  }
  //@}

  //! \name Setters
  //@{

  /// \brief Set the minimum corner
  /// \param[in] min_corner The new minimum corner.
  template <ValidPointType OtherPointType>
  KOKKOS_FUNCTION void set_min_corner(const OtherPointType& min_corner) {
    min_corner_ = min_corner;
  }

  /// \brief Set the minimum corner
  /// \param[in] x The new x-coordinate.
  /// \param[in] y The new y-coordinate.
  /// \param[in] z The new z-coordinate.
  KOKKOS_FUNCTION
  void set_min_corner(const scalar_t& x, const scalar_t& y, const scalar_t& z) {
    min_corner_[0] = x;
    min_corner_[1] = y;
    min_corner_[2] = z;
  }

  /// \brief Set the maximum corner
  /// \param[in] max_corner The new maximum corner.
  template <ValidPointType OtherPointType>
  KOKKOS_FUNCTION void set_max_corner(const OtherPointType& max_corner) {
    max_corner_ = max_corner;
  }

  /// \brief Set the maximum corner
  /// \param[in] x The new x-coordinate.
  /// \param[in] y The new y-coordinate.
  /// \param[in] z The new z-coordinate.
  KOKKOS_FUNCTION
  void set_max_corner(const scalar_t& x, const scalar_t& y, const scalar_t& z) {
    max_corner_[0] = x;
    max_corner_[1] = y;
    max_corner_[2] = z;
  }
  //@}

 private:
  //! \name Private helpers
  //@{

  /// \brief Get the maximum possible scalar value
  static KOKKOS_FUNCTION constexpr scalar_t scalar_max() {
    return Kokkos::Experimental::finite_max_v<scalar_t>;
  }

  /// \brief Get the minimum possible scalar value
  static KOKKOS_FUNCTION constexpr scalar_t scalar_min() {
    // finite_min_v<T> returns the most negative real value (equivalent to numeric_limits<T>::lowest).
    // it is the 'lowest' value that we want here.
    return Kokkos::Experimental::finite_min_v<scalar_t>;
  }
  //@}

  min_point_t min_corner_;
  max_point_t max_corner_;
};  // AABB

/// \brief Deduction guide for AABB
template <typename Scalar>
AABB(Scalar, Scalar, Scalar, Scalar, Scalar, Scalar) -> AABB<Scalar>;
//
template <typename Scalar, ValidPointType MinPointType, ValidPointType MaxPointType>
AABB(MinPointType, MaxPointType) -> AABB<Scalar, MinPointType, MaxPointType>;

/// @brief (Implementation) Type trait to determine if a type is an AABB
template <typename T>
struct is_aabb_impl : std::false_type {};
//
template <typename Scalar, ValidPointType MinPointType, ValidPointType MaxPointType>
struct is_aabb_impl<AABB<Scalar, MinPointType, MaxPointType>> : std::true_type {};

/// @brief Type trait to determine if a type is an AABB
template <typename T>
struct is_aabb : is_aabb_impl<std::remove_cv_t<T>> {};
//
template <typename T>
constexpr bool is_aabb_v = is_aabb<T>::value;

/// @brief Concept to determine if a type is a valid AABB type
template <typename AABBType>
concept ValidAABBType = is_aabb_v<std::remove_cv_t<AABBType>> &&
                        is_point_v<decltype(std::declval<std::remove_cv_t<AABBType>>().min_corner())> &&
                        is_point_v<decltype(std::declval<std::remove_cv_t<AABBType>>().max_corner())> &&
                        is_point_v<decltype(std::declval<const std::remove_cv_t<AABBType>>().min_corner())> &&
                        is_point_v<decltype(std::declval<const std::remove_cv_t<AABBType>>().max_corner())> &&
                        requires(std::remove_cv_t<AABBType> aabb, const std::remove_cv_t<AABBType> const_aabb) {
                          typename std::remove_cv_t<AABBType>::scalar_t;

                          { aabb.x_min() } -> std::convertible_to<typename std::remove_cv_t<AABBType>::scalar_t&>;
                          { aabb.y_min() } -> std::convertible_to<typename std::remove_cv_t<AABBType>::scalar_t&>;
                          { aabb.z_min() } -> std::convertible_to<typename std::remove_cv_t<AABBType>::scalar_t&>;
                          { aabb.x_max() } -> std::convertible_to<typename std::remove_cv_t<AABBType>::scalar_t&>;
                          { aabb.y_max() } -> std::convertible_to<typename std::remove_cv_t<AABBType>::scalar_t&>;
                          { aabb.z_max() } -> std::convertible_to<typename std::remove_cv_t<AABBType>::scalar_t&>;

                          {
                            const_aabb.x_min()
                          } -> std::convertible_to<const typename std::remove_cv_t<AABBType>::scalar_t&>;
                          {
                            const_aabb.y_min()
                          } -> std::convertible_to<const typename std::remove_cv_t<AABBType>::scalar_t&>;
                          {
                            const_aabb.z_min()
                          } -> std::convertible_to<const typename std::remove_cv_t<AABBType>::scalar_t&>;
                          {
                            const_aabb.x_max()
                          } -> std::convertible_to<const typename std::remove_cv_t<AABBType>::scalar_t&>;
                          {
                            const_aabb.y_max()
                          } -> std::convertible_to<const typename std::remove_cv_t<AABBType>::scalar_t&>;
                          {
                            const_aabb.z_max()
                          } -> std::convertible_to<const typename std::remove_cv_t<AABBType>::scalar_t&>;
                        };  // ValidAABBType

static_assert(ValidAABBType<AABB<float>> && ValidAABBType<const AABB<float>> && ValidAABBType<AABB<double>> &&
                  ValidAABBType<const AABB<double>>,
              "AABB should satisfy the ValidAABBType concept.");

//! \name Non-member functions for ValidAABBType objects
//@{

/// \brief Equality operator
template <ValidAABBType AABBType1, ValidAABBType AABBType2>
KOKKOS_FUNCTION bool operator==(const AABBType1& aabb1, const AABBType2& aabb2) {
  return (aabb1.min_corner() == aabb2.min_corner()) && (aabb1.max_corner() == aabb2.max_corner());
}

/// \brief Inequality operator
template <ValidAABBType AABBType1, ValidAABBType AABBType2>
KOKKOS_FUNCTION bool operator!=(const AABBType1& aabb1, const AABBType2& aabb2) {
  return (aabb1.min_corner() != aabb2.min_corner()) || (aabb1.max_corner() != aabb2.max_corner());
}

template <typename Scalar>
std::ostream& operator<<(std::ostream& os, const AABB<Scalar>& aabb) {
  os << "{" << aabb.min_corner() << "->" << aabb.max_corner() << "}";
  return os;
}

/// \brief Check if two AABBs intersect
template <ValidAABBType AABBType1, ValidAABBType AABBType2>
KOKKOS_FUNCTION bool intersects(const AABBType1& aabb1, const AABBType2& aabb2) {
  const auto& amax = aabb1.max_corner();
  const auto& bmin = aabb2.min_corner();
  if (amax[0] < bmin[0] || amax[1] < bmin[1] || amax[2] < bmin[2]) {
    return false;
  }

  const auto& bmax = aabb2.max_corner();
  const auto& amin = aabb1.min_corner();
  const bool disjoint2 = bmax[0] < amin[0] || bmax[1] < amin[1] || bmax[2] < amin[2];
  return !disjoint2;
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_PRIMITIVES_AABB_HPP_
