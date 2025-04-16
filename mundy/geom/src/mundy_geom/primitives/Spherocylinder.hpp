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

#ifndef MUNDY_GEOM_PRIMITIVES_SPHEROCYLINDER_HPP_
#define MUNDY_GEOM_PRIMITIVES_SPHEROCYLINDER_HPP_

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

template <typename Scalar, ValidPointType PointType = Point<Scalar>,
          mundy::math::ValidQuaternionType QuaternionType = mundy::math::Quaternion<Scalar>,
          typename OwnershipType = mundy::math::Ownership::Owns>
class Spherocylinder {
  static_assert(
      std::is_same_v<typename PointType::scalar_t, Scalar> && std::is_same_v<typename QuaternionType::scalar_t, Scalar>,
      "The scalar type of the PointType and QuaternionType must match the scalar type of the Spherocylinder.");
  static_assert(
      std::is_same_v<typename PointType::ownership_t, OwnershipType> &&
          std::is_same_v<typename QuaternionType::ownership_t, OwnershipType>,
      "The ownership type of the PointType and QuaternionType must match the ownership type of the Spherocylinder.\n"
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

  /// \brief Default constructor for owning Spherocylinders. Default initializes the center and sets the radius to an
  /// invalid value of -1
  KOKKOS_FUNCTION
  Spherocylinder()
    requires std::is_same_v<OwnershipType, mundy::math::Ownership::Owns>
      : center_(scalar_t(), scalar_t(), scalar_t()),
        orientation_(static_cast<scalar_t>(1), static_cast<scalar_t>(0), static_cast<scalar_t>(0),
                     static_cast<scalar_t>(0)),
        radius_(static_cast<scalar_t>(-1)),
        length_(static_cast<scalar_t>(-1)) {
  }

  /// \brief No default constructor for viewing Spherocylinders.
  KOKKOS_FUNCTION
  Spherocylinder()
    requires std::is_same_v<OwnershipType, mundy::math::Ownership::Views>
  = delete;

  /// \brief Constructor to initialize the center and radius.
  /// \param[in] center The center of the Spherocylinder.
  /// \param[in] orientation The orientation of the Spherocylinder (as a quaternion).
  /// \param[in] radius The radius of the Spherocylinder.
  /// \param[in] length The length of the Spherocylinder.
  KOKKOS_FUNCTION
  Spherocylinder(const point_t& center, const orientation_t& orientation, const scalar_t& radius,
                 const scalar_t& length)
      : center_(center), orientation_(orientation), radius_(radius), length_(length) {
  }

  /// \brief Constructor to initialize the center and radius.
  /// \param[in] center The center of the Spherocylinder.
  /// \param[in] orientation The orientation of the Spherocylinder (as a quaternion).
  /// \param[in] radius The radius of the Spherocylinder.
  /// \param[in] length The length of the Spherocylinder.
  template <ValidPointType OtherPointType, mundy::math::ValidQuaternionType OtherQuaternionType>
  KOKKOS_FUNCTION Spherocylinder(const OtherPointType& center, const OtherQuaternionType& orientation,
                                 const scalar_t& radius, const scalar_t& length)
    requires(!std::is_same_v<OtherPointType, point_t> || !std::is_same_v<OtherQuaternionType, orientation_t>)
      : center_(center), orientation_(orientation), radius_(radius), length_(length) {
  }

  /// \brief Destructor
  KOKKOS_DEFAULTED_FUNCTION
  ~Spherocylinder() = default;

  /// \brief Deep copy constructor
  KOKKOS_FUNCTION
  Spherocylinder(const Spherocylinder<scalar_t, point_t, orientation_t, ownership_t>& other)
      : center_(other.center_), orientation_(other.orientation_), radius_(other.radius_), length_(other.length_) {
  }

  /// \brief Deep copy constructor with different spherocylinder type
  template <typename OtherSpherocylinderType>
  KOKKOS_FUNCTION Spherocylinder(const OtherSpherocylinderType& other)
    requires(!std::is_same_v<OtherSpherocylinderType, Spherocylinder<scalar_t, point_t, orientation_t, ownership_t>>)
      : center_(other.center_), orientation_(other.orientation_), radius_(other.radius_), length_(other.length_) {
  }

  /// \brief Deep move constructor
  KOKKOS_FUNCTION
  Spherocylinder(Spherocylinder<scalar_t, point_t, orientation_t, ownership_t>&& other)
      : center_(std::move(other.center_)),
        orientation_{std::move(other.orientation_)},
        radius_(std::move(other.radius_)),
        length_(std::move(other.length_)) {
  }

  /// \brief Deep move constructor
  template <typename OtherSpherocylinderType>
  KOKKOS_FUNCTION Spherocylinder(OtherSpherocylinderType&& other)
    requires(!std::is_same_v<OtherSpherocylinderType, Spherocylinder<scalar_t, point_t, orientation_t, ownership_t>>)
      : center_(std::move(other.center_)),
        orientation_{std::move(other.orientation_)},
        radius_(std::move(other.radius_)),
        length_(std::move(other.length_)) {
  }
  //@}

  //! \name Operators
  //@{

  /// \brief Copy assignment operator
  KOKKOS_FUNCTION
  Spherocylinder<scalar_t, point_t, orientation_t, ownership_t>& operator=(
      const Spherocylinder<scalar_t, point_t, orientation_t, ownership_t>& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = other.center_;
    orientation_ = other.orientation_;
    radius_ = other.radius_;
    length_ = other.length_;
    return *this;
  }

  /// \brief Copy assignment operator
  template <typename OtherSpherocylinderType>
  KOKKOS_FUNCTION Spherocylinder<scalar_t, point_t, orientation_t, ownership_t>& operator=(
      const OtherSpherocylinderType& other)
    requires(!std::is_same_v<OtherSpherocylinderType, Spherocylinder<scalar_t, point_t, orientation_t, ownership_t>>)
  {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = other.center_;
    orientation_ = other.orientation_;
    radius_ = other.radius_;
    length_ = other.length_;
    return *this;
  }

  /// \brief Move assignment operator
  KOKKOS_FUNCTION
  Spherocylinder<scalar_t, point_t, orientation_t, ownership_t>& operator=(
      Spherocylinder<scalar_t, point_t, orientation_t, ownership_t>&& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = std::move(other.center_);
    orientation_ = std::move(other.orientation_);
    radius_ = std::move(other.radius_);
    length_ = std::move(other.length_);
    return *this;
  }

  /// \brief Move assignment operator
  template <typename OtherSpherocylinderType>
  KOKKOS_FUNCTION Spherocylinder<scalar_t, point_t, orientation_t, ownership_t>& operator=(
      OtherSpherocylinderType&& other)
    requires(!std::is_same_v<OtherSpherocylinderType, Spherocylinder<scalar_t, point_t, orientation_t, ownership_t>>)
  {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = std::move(other.center_);
    orientation_ = std::move(other.orientation_);
    radius_ = std::move(other.radius_);
    length_ = std::move(other.length_);
    return *this;
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Accessor for the center
  KOKKOS_FUNCTION
  const point_t& center() const {
    return center_;
  }

  /// \brief Accessor for the center
  KOKKOS_FUNCTION
  point_t& center() {
    return center_;
  }

  /// \brief Accessor for the orientation
  KOKKOS_FUNCTION
  const orientation_t& orientation() const {
    return orientation_;
  }

  /// \brief Accessor for the orientation
  KOKKOS_FUNCTION
  orientation_t& orientation() {
    return orientation_;
  }

  /// \brief Accessor for the radius
  KOKKOS_FUNCTION
  const scalar_t& radius() const {
    return radius_;
  }

  /// \brief Accessor for the radius
  KOKKOS_FUNCTION
  scalar_t& radius() {
    return radius_;
  }

  /// \brief Accessor for the length
  KOKKOS_FUNCTION
  const scalar_t& length() const {
    return length_;
  }

  /// \brief Accessor for the length
  KOKKOS_FUNCTION
  scalar_t& length() {
    return length_;
  }
  //@}

  //! \name Setters
  //@{

  /// \brief Set the center
  /// \param[in] center The new center.
  template <ValidPointType OtherPointType>
  KOKKOS_FUNCTION void set_center(const OtherPointType& center) {
    center_ = center;
  }

  /// \brief Set the center
  /// \param[in] x The x-coordinate.
  /// \param[in] y The y-coordinate.
  /// \param[in] z The z-coordinate.
  KOKKOS_FUNCTION
  void set_center(const scalar_t& x, const scalar_t& y, const scalar_t& z) {
    center_[0] = x;
    center_[1] = y;
    center_[2] = z;
  }

  /// \brief Set the orientation
  /// \param[in] orientation The new orientation.
  KOKKOS_FUNCTION
  void set_orientation(const orientation_t& orientation) {
    orientation_ = orientation;
  }

  /// \brief Set the orientation
  /// \param[in] qw The scalar-component of the orientation quaternion.
  /// \param[in] qx The x-component of the orientation quaternion.
  /// \param[in] qy The y-component of the orientation quaternion.
  /// \param[in] qz The z-component of the orientation quaternion.
  KOKKOS_FUNCTION
  void set_orientation(const scalar_t& qw, const scalar_t& qx, const scalar_t& qy, const scalar_t& qz) {
    orientation_[0] = qw;
    orientation_[1] = qx;
    orientation_[2] = qy;
    orientation_[3] = qz;
  }

  /// \brief Set the radius
  /// \param[in] radius The new radius.
  KOKKOS_FUNCTION
  void set_radius(const scalar_t& radius) {
    radius_ = radius;
  }

  /// \brief Set the length
  /// \param[in] length The new length.
  KOKKOS_FUNCTION
  void set_length(const scalar_t& length) {
    length_ = length;
  }
  //@}

 private:
  point_t center_;
  orientation_t orientation_;
  std::conditional_t<std::is_same_v<OwnershipType, mundy::math::Ownership::Owns>, scalar_t, scalar_t&> radius_;
  std::conditional_t<std::is_same_v<OwnershipType, mundy::math::Ownership::Owns>, scalar_t, scalar_t&> length_;
};

/// @brief Type trait to determine if a type is a Spherocylinder
template <typename T>
struct is_spherocylinder : std::false_type {};
//
template <typename Scalar, ValidPointType PointType, mundy::math::ValidQuaternionType QuaternionType,
          typename OwnershipType>
struct is_spherocylinder<Spherocylinder<Scalar, PointType, QuaternionType, OwnershipType>> : std::true_type {};
//
template <typename Scalar, ValidPointType PointType, mundy::math::ValidQuaternionType QuaternionType,
          typename OwnershipType>
struct is_spherocylinder<const Spherocylinder<Scalar, PointType, QuaternionType, OwnershipType>> : std::true_type {};
//
template <typename T>
constexpr bool is_spherocylinder_v = is_spherocylinder<T>::value;

/// @brief Concept to check if a type is a valid Spherocylinder type
template <typename SpherocylinderType>
concept ValidSpherocylinderType =
    is_spherocylinder_v<std::decay_t<SpherocylinderType>> &&
    is_point_v<typename std::decay_t<SpherocylinderType>::point_t> &&
    mundy::math::is_quaternion_v<typename std::decay_t<SpherocylinderType>::orientation_t> &&
    is_point_v<decltype(std::declval<std::decay_t<SpherocylinderType>>().center())> &&
    is_point_v<decltype(std::declval<const std::decay_t<SpherocylinderType>>().center())> &&
    mundy::math::is_quaternion_v<decltype(std::declval<std::decay_t<SpherocylinderType>>().orientation())> &&
    mundy::math::is_quaternion_v<decltype(std::declval<const std::decay_t<SpherocylinderType>>().orientation())> &&
    requires(std::decay_t<SpherocylinderType> spherocylinder,
             const std::decay_t<SpherocylinderType> const_spherocylinder) {
      typename std::decay_t<SpherocylinderType>::scalar_t;
      typename std::decay_t<SpherocylinderType>::point_t;
      typename std::decay_t<SpherocylinderType>::orientation_t;
      { spherocylinder.radius() } -> std::same_as<typename std::decay_t<SpherocylinderType>::scalar_t&>;
      { const_spherocylinder.radius() } -> std::same_as<const typename std::decay_t<SpherocylinderType>::scalar_t&>;
      { spherocylinder.length() } -> std::same_as<typename std::decay_t<SpherocylinderType>::scalar_t&>;
      { const_spherocylinder.length() } -> std::same_as<const typename std::decay_t<SpherocylinderType>::scalar_t&>;
    };  // ValidSpherocylinderType

static_assert(ValidSpherocylinderType<Spherocylinder<float>> && ValidSpherocylinderType<const Spherocylinder<float>> &&
                  ValidSpherocylinderType<Spherocylinder<double>> &&
                  ValidSpherocylinderType<const Spherocylinder<double>>,
              "Spherocylinder should satisfy the ValidSpherocylinderType concept.");

//! \name Non-member functions for ValidSpherocylinderType objects
//@{

/// \brief Equality operator
template <ValidSpherocylinderType SpherocylinderType1, ValidSpherocylinderType SpherocylinderType2>
KOKKOS_FUNCTION bool operator==(const SpherocylinderType1& spherocylinder1,
                                const SpherocylinderType2& spherocylinder2) {
  return (spherocylinder1.radius() == spherocylinder2.radius()) &&
         (spherocylinder1.length() == spherocylinder2.length()) &&
         (spherocylinder1.center() == spherocylinder2.center()) &&
         (spherocylinder1.orientation() == spherocylinder2.orientation());
}

/// \brief Inequality operator
template <ValidSpherocylinderType SpherocylinderType1, ValidSpherocylinderType SpherocylinderType2>
KOKKOS_FUNCTION bool operator!=(const SpherocylinderType1& spherocylinder1,
                                const SpherocylinderType2& spherocylinder2) {
  return (spherocylinder1.radius() != spherocylinder2.radius()) ||
         (spherocylinder1.length() != spherocylinder2.length()) ||
         (spherocylinder1.center() != spherocylinder2.center()) ||
         (spherocylinder1.orientation() != spherocylinder2.orientation());
}

/// \brief OStream operator
template <ValidSpherocylinderType SpherocylinderType>
std::ostream& operator<<(std::ostream& os, const SpherocylinderType& spherocylinder) {
  os << "{" << spherocylinder.center() << ":" << spherocylinder.orientation() << ":" << spherocylinder.radius() << ":"
     << spherocylinder.length() << "}";
  return os;
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_PRIMITIVES_SPHEROCYLINDER_HPP_
