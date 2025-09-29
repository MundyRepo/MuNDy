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

#ifndef MUNDY_GEOM_PRIMITIVES_RING_HPP_
#define MUNDY_GEOM_PRIMITIVES_RING_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core
#include <iostream>
#include <stdexcept>
#include <utility>

// Our libs
#include <mundy_core/throw_assert.hpp>         // for MUNDY_THROW_ASSERT
#include <mundy_geom/primitives/Circle3D.hpp>  // for mundy::geom::Circle3D
#include <mundy_geom/primitives/Point.hpp>     // for mundy::geom::Point
#include <mundy_math/Quaternion.hpp>           // for mundy::math::Quaternion
#include <mundy_math/Vector3.hpp>              // for mundy::math::Vector3

namespace mundy {

namespace geom {

template <typename Scalar, ValidPointType PointType = Point<Scalar>,
          math::ValidQuaternionType QuaternionType = math::Quaternion<Scalar>,
          typename OwnershipType = math::Ownership::Owns>
class Ring {
  static_assert(std::is_same_v<typename PointType::scalar_t, Scalar> &&
                    std::is_same_v<typename QuaternionType::scalar_t, Scalar>,
                "The scalar type of the PointType and QuaternionType must match the scalar type of the Ring.");
  static_assert(std::is_same_v<typename PointType::ownership_t, OwnershipType> &&
                    std::is_same_v<typename QuaternionType::ownership_t, OwnershipType>,
                "The ownership type of the PointType and QuaternionType must match the ownership type of the Ring.\n"
                "This is somewhat restrictive, and we may want to relax this constraint in the future.\n"
                "If you need to use a different ownership type, please let us know and we'll remove this restriction.");

 public:
  //! \name Type aliases
  //@{

  /// \brief Our scalar type
  using scalar_t = Scalar;

  /// \brief Our point type
  using point_t = PointType;

  /// \brief Our orientation type
  using orientation_t = QuaternionType;

  /// \brief Our ownership type
  using ownership_t = OwnershipType;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor for owning Rings. Default initializes the center and sets the radius to an
  /// invalid value of -1
  KOKKOS_FUNCTION
  constexpr Ring()
    requires std::is_same_v<OwnershipType, math::Ownership::Owns>
      : center_circle_(), minor_radius_(static_cast<scalar_t>(-1)) {
  }

  /// \brief No default constructor for viewing Rings.
  KOKKOS_FUNCTION
  constexpr Ring()
    requires std::is_same_v<OwnershipType, math::Ownership::Views>
  = delete;

  /// \brief Constructor to initialize the ring.
  /// \param[in] center The center of the Ring.
  /// \param[in] orientation The orientation of the Ring (as a quaternion).
  /// \param[in] major_radius The radius of the center circle of the Ring.
  /// \param[in] minor_radius The radius of the tube around said circle.
  KOKKOS_FUNCTION
  constexpr Ring(const point_t& center, const orientation_t& orientation, const scalar_t& major_radius,
                 const scalar_t& minor_radius)
      : center_circle_(center, orientation, major_radius), minor_radius_(minor_radius) {
  }

  /// \brief Constructor to initialize the ring.
  /// \param[in] center The center of the Ring.
  /// \param[in] orientation The orientation of the Ring (as a quaternion).
  /// \param[in] major_radius The radius of the center circle of the Ring.
  /// \param[in] minor_radius The radius of the tube around said circle.
  template <ValidPointType OtherPointType, math::ValidQuaternionType OtherQuaternionType>
  KOKKOS_FUNCTION constexpr Ring(const OtherPointType& center, const OtherQuaternionType& orientation,
                                 const scalar_t& major_radius, const scalar_t& minor_radius)
    requires(!std::is_same_v<OtherPointType, point_t> || !std::is_same_v<OtherQuaternionType, orientation_t>)
      : center_circle_(center, orientation, major_radius), minor_radius_(minor_radius) {
  }

  /// \brief Destructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr ~Ring() = default;

  /// \brief Deep copy constructor
  KOKKOS_FUNCTION
  constexpr Ring(const Ring<scalar_t, point_t, orientation_t, ownership_t>& other)
      : center_circle_(other.center_circle_), minor_radius_(other.minor_radius_) {
  }

  /// \brief Deep copy constructor with different ring type
  template <typename OtherRingType>
  KOKKOS_FUNCTION constexpr Ring(const OtherRingType& other)
    requires(!std::is_same_v<OtherRingType, Ring<scalar_t, point_t, orientation_t, ownership_t>>)
      : center_circle_(other.center_circle_), minor_radius_(other.minor_radius_) {
  }

  /// \brief Deep move constructor
  KOKKOS_FUNCTION
  constexpr Ring(Ring<scalar_t, point_t, orientation_t, ownership_t>&& other)
      : center_circle_(std::move(other.center_circle_)), minor_radius_(std::move(other.minor_radius_)) {
  }

  /// \brief Deep move constructor
  template <typename OtherRingType>
  KOKKOS_FUNCTION constexpr Ring(OtherRingType&& other)
    requires(!std::is_same_v<OtherRingType, Ring<scalar_t, point_t, orientation_t, ownership_t>>)
      : center_circle_(std::move(other.center_circle_)), minor_radius_(std::move(other.minor_radius_)) {
  }
  //@}

  //! \name Operators
  //@{

  /// \brief Copy assignment operator
  KOKKOS_FUNCTION
  constexpr Ring<scalar_t, point_t, orientation_t, ownership_t>& operator=(
      const Ring<scalar_t, point_t, orientation_t, ownership_t>& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_circle_ = other.center_circle_;
    minor_radius_ = other.minor_radius_;
    return *this;
  }

  /// \brief Copy assignment operator
  template <typename OtherRingType>
  KOKKOS_FUNCTION constexpr Ring<scalar_t, point_t, orientation_t, ownership_t>& operator=(const OtherRingType& other)
    requires(!std::is_same_v<OtherRingType, Ring<scalar_t, point_t, orientation_t, ownership_t>>)
  {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_circle_ = other.center_circle_;
    minor_radius_ = other.minor_radius_;
    return *this;
  }

  /// \brief Move assignment operator
  KOKKOS_FUNCTION
  constexpr Ring<scalar_t, point_t, orientation_t, ownership_t>& operator=(
      Ring<scalar_t, point_t, orientation_t, ownership_t>&& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_circle_ = std::move(other.center_circle_);
    minor_radius_ = std::move(other.minor_radius_);
    return *this;
  }

  /// \brief Move assignment operator
  template <typename OtherRingType>
  KOKKOS_FUNCTION constexpr Ring<scalar_t, point_t, orientation_t, ownership_t>& operator=(OtherRingType&& other)
    requires(!std::is_same_v<OtherRingType, Ring<scalar_t, point_t, orientation_t, ownership_t>>)
  {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_circle_ = std::move(other.center_circle_);
    minor_radius_ = std::move(other.minor_radius_);
    return *this;
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Accessor for the center line (a Circle3D)
  KOKKOS_FUNCTION
  constexpr const Circle3D<scalar_t, point_t, orientation_t, ownership_t>& center_circle() const {
    return center_circle_;
  }

  /// \brief Accessor for the center
  KOKKOS_FUNCTION
  constexpr const point_t& center() const {
    return center_circle_.center();
  }

  /// \brief Accessor for the center
  KOKKOS_FUNCTION
  constexpr point_t& center() {
    return center_circle_.center();
  }

  /// \brief Accessor for the orientation
  KOKKOS_FUNCTION
  constexpr const orientation_t& orientation() const {
    return center_circle_.orientation();
  }

  /// \brief Accessor for the orientation
  KOKKOS_FUNCTION
  constexpr orientation_t& orientation() {
    return center_circle_.orientation();
  }

  /// \brief Accessor for the major radius
  KOKKOS_FUNCTION
  constexpr const scalar_t& major_radius() const {
    return center_circle_.major_radius();
  }

  /// \brief Accessor for the major radius
  KOKKOS_FUNCTION
  constexpr scalar_t& major_radius() {
    return center_circle_.major_radius();
  }

  /// \brief Accessor for the minor radius
  KOKKOS_FUNCTION
  constexpr const scalar_t& minor_radius() const {
    return minor_radius_;
  }

  /// \brief Accessor for the minor radius
  KOKKOS_FUNCTION
  constexpr scalar_t& minor_radius() {
    return minor_radius_;
  }
  //@}

  //! \name Setters
  //@{

  /// \brief Set the center
  /// \param[in] center The new center.
  template <ValidPointType OtherPointType>
  KOKKOS_FUNCTION constexpr void set_center(const OtherPointType& center) {
    center_circle_.set_center(center);
  }

  /// \brief Set the center
  /// \param[in] x The x-coordinate.
  /// \param[in] y The y-coordinate.
  /// \param[in] z The z-coordinate.
  KOKKOS_FUNCTION
  constexpr void set_center(const scalar_t& x, const scalar_t& y, const scalar_t& z) {
    center_circle_.set_center(x, y, z);
  }

  /// \brief Set the orientation
  /// \param[in] orientation The new orientation.
  KOKKOS_FUNCTION
  constexpr void set_orientation(const orientation_t& orientation) {
    center_circle_.set_orientation(orientation);
  }

  /// \brief Set the orientation
  /// \param[in] qw The scalar-component of the orientation quaternion.
  /// \param[in] qx The x-component of the orientation quaternion.
  /// \param[in] qy The y-component of the orientation quaternion.
  /// \param[in] qz The z-component of the orientation quaternion.
  KOKKOS_FUNCTION
  constexpr void set_orientation(const scalar_t& qw, const scalar_t& qx, const scalar_t& qy, const scalar_t& qz) {
    center_circle_.set_orientation(qw, qx, qy, qz);
  }

  /// \brief Set the major radius
  /// \param[in] major_radius The new major radius.
  KOKKOS_FUNCTION
  constexpr void set_major_radius(const scalar_t& major_radius) {
    center_circle_.set_major_radius(major_radius);
  }

  /// \brief Set the minor radius
  /// \param[in] minor_radius The new minor radius.
  KOKKOS_FUNCTION
  constexpr void set_minor_radius(const scalar_t& minor_radius) {
    minor_radius_ = minor_radius;
  }
  //@}

 private:
  Circle3D<scalar_t, point_t, orientation_t, ownership_t> center_circle_;
  std::conditional_t<std::is_same_v<OwnershipType, math::Ownership::Owns>, scalar_t, scalar_t&> minor_radius_;
};

/// @brief (Implementation) Type trait to determine if a type is a Ring
template <typename T>
struct impl_is_ring : std::false_type {};
//
template <typename Scalar, ValidPointType PointType, math::ValidQuaternionType QuaternionType, typename OwnershipType>
struct impl_is_ring<Ring<Scalar, PointType, QuaternionType, OwnershipType>> : std::true_type {};

/// @brief Type trait to determine if a type is a Ring
template <typename T>
struct is_ring : impl_is_ring<std::remove_cv_t<T>> {};
//
template <typename T>
constexpr bool is_ring_v = is_ring<T>::value;

/// @brief Concept to check if a type is a valid Ring type
template <typename RingType>
concept ValidRingType = is_ring_v<RingType>;

static_assert(ValidRingType<Ring<float>> && ValidRingType<const Ring<float>> && ValidRingType<Ring<double>> &&
                  ValidRingType<const Ring<double>>,
              "Ring should satisfy the ValidRingType concept.");

//! \name Non-member functions for ValidRingType objects
//@{

/// \brief Equality operator
template <ValidRingType RingType1, ValidRingType RingType2>
KOKKOS_FUNCTION constexpr bool operator==(const RingType1& ring1, const RingType2& ring2) {
  return (ring1.major_radius() == ring2.major_radius()) && (ring1.minor_radius() == ring2.minor_radius()) &&
         (ring1.center() == ring2.center()) && (ring1.orientation() == ring2.orientation());
}

/// \brief Inequality operator
template <ValidRingType RingType1, ValidRingType RingType2>
KOKKOS_FUNCTION constexpr bool operator!=(const RingType1& ring1, const RingType2& ring2) {
  return (ring1.major_radius() != ring2.major_radius()) || (ring1.minor_radius() != ring2.minor_radius()) ||
         (ring1.center() != ring2.center()) || (ring1.orientation() != ring2.orientation());
}

/// \brief OStream operator
template <ValidRingType RingType>
std::ostream& operator<<(std::ostream& os, const RingType& ring) {
  os << "{" << ring.center() << ":" << ring.orientation() << ":" << ring.major_radius() << ":" << ring.minor_radius()
     << "}";
  return os;
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_PRIMITIVES_RING_HPP_
