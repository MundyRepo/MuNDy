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

#ifndef MUNDY_GEOM_PRIMITIVES_CIRCLE3D_HPP_
#define MUNDY_GEOM_PRIMITIVES_CIRCLE3D_HPP_

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
class Circle3D {
  static_assert(std::is_same_v<typename PointType::scalar_t, Scalar> &&
                    std::is_same_v<typename QuaternionType::scalar_t, Scalar>,
                "The scalar type of the PointType and QuaternionType must match the scalar type of the Circle3D.");
  static_assert(
      std::is_same_v<typename PointType::ownership_t, OwnershipType> &&
          std::is_same_v<typename QuaternionType::ownership_t, OwnershipType>,
      "The ownership type of the PointType and QuaternionType must match the ownership type of the Circle3D.\n"
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

  /// \brief Default constructor for owning Circle3Ds. Initializes as invalid.
  KOKKOS_FUNCTION
  Circle3D()
    requires std::is_same_v<OwnershipType, mundy::math::Ownership::Owns>
      : center_(scalar_t(), scalar_t(), scalar_t()),
        orientation_(static_cast<scalar_t>(1), static_cast<scalar_t>(0), static_cast<scalar_t>(0),
                     static_cast<scalar_t>(0)),
        radius_(static_cast<scalar_t>(-1)) {
  }

  /// \brief No default constructor for viewing Circle3Ds.
  KOKKOS_FUNCTION
  Circle3D()
    requires std::is_same_v<OwnershipType, mundy::math::Ownership::Views>
  = delete;

  /// \brief Constructor to initialize the circle3d.
  /// \param[in] center The center (in the lab frame) of the Circle3D.
  /// \param[in] orientation The quaternion orientation mapping a circle with normal in the z-direction to the lab
  /// frame.
  /// \param[in] radius The radius of the circle.
  KOKKOS_FUNCTION
  Circle3D(const point_t& center, const orientation_t& orientation, const scalar_t& radius)
      : center_(center), orientation_(orientation), radius_(radius) {
  }

  /// \brief Constructor to initialize the circle3d.
  /// \param[in] center The center (in the lab frame) of the Circle3D.
  /// \param[in] orientation The quaternion orientation mapping a circle with normal in the z-direction to the lab
  /// frame.
  /// \param[in] radius The radius of the circle.
  template <ValidPointType OtherPointType, mundy::math::ValidQuaternionType OtherQuaternionType>
  KOKKOS_FUNCTION Circle3D(const OtherPointType& center, const OtherQuaternionType& orientation, const scalar_t& radius)
    requires(!std::is_same_v<OtherPointType, point_t> || !std::is_same_v<OtherQuaternionType, orientation_t>)
      : center_(center), orientation_(orientation), radius_(radius) {
  }

  /// \brief Destructor
  KOKKOS_DEFAULTED_FUNCTION
  ~Circle3D() = default;

  /// \brief Deep copy constructor
  KOKKOS_FUNCTION
  Circle3D(const Circle3D<scalar_t, point_t, orientation_t, ownership_t>& other)
      : center_(other.center_), orientation_(other.orientation_), radius_(other.radius_) {
  }

  /// \brief Deep copy constructor with different circle3d type
  template <typename OtherCircle3DType>
  KOKKOS_FUNCTION Circle3D(const OtherCircle3DType& other)
    requires(!std::is_same_v<OtherCircle3DType, Circle3D<scalar_t, point_t, orientation_t, ownership_t>>)
      : center_(other.center_), orientation_(other.orientation_), radius_(other.radius_) {
  }

  /// \brief Deep move constructor
  KOKKOS_FUNCTION
  Circle3D(Circle3D<scalar_t, point_t, orientation_t, ownership_t>&& other)
      : center_(std::move(other.center_)),
        orientation_{std::move(other.orientation_)},
        radius_(std::move(other.radius_)) {
  }

  /// \brief Deep move constructor
  template <typename OtherCircle3DType>
  KOKKOS_FUNCTION Circle3D(OtherCircle3DType&& other)
    requires(!std::is_same_v<OtherCircle3DType, Circle3D<scalar_t, point_t, orientation_t, ownership_t>>)
      : center_(std::move(other.center_)),
        orientation_{std::move(other.orientation_)},
        radius_(std::move(other.radius_)) {
  }
  //@}

  //! \name Operators
  //@{

  /// \brief Copy assignment operator
  KOKKOS_FUNCTION
  Circle3D<scalar_t, point_t, orientation_t, ownership_t>& operator=(
      const Circle3D<scalar_t, point_t, orientation_t, ownership_t>& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = other.center_;
    orientation_ = other.orientation_;
    radius_ = other.radius_;
    return *this;
  }

  /// \brief Copy assignment operator
  template <typename OtherCircle3DType>
  KOKKOS_FUNCTION Circle3D<scalar_t, point_t, orientation_t, ownership_t>& operator=(const OtherCircle3DType& other)
    requires(!std::is_same_v<OtherCircle3DType, Circle3D<scalar_t, point_t, orientation_t, ownership_t>>)
  {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = other.center_;
    orientation_ = other.orientation_;
    radius_ = other.radius_;
    return *this;
  }

  /// \brief Move assignment operator
  KOKKOS_FUNCTION
  Circle3D<scalar_t, point_t, orientation_t, ownership_t>& operator=(
      Circle3D<scalar_t, point_t, orientation_t, ownership_t>&& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = std::move(other.center_);
    orientation_ = std::move(other.orientation_);
    radius_ = std::move(other.radius_);
    return *this;
  }

  /// \brief Move assignment operator
  template <typename OtherCircle3DType>
  KOKKOS_FUNCTION Circle3D<scalar_t, point_t, orientation_t, ownership_t>& operator=(OtherCircle3DType&& other)
    requires(!std::is_same_v<OtherCircle3DType, Circle3D<scalar_t, point_t, orientation_t, ownership_t>>)
  {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = std::move(other.center_);
    orientation_ = std::move(other.orientation_);
    radius_ = std::move(other.radius_);
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

  /// \brief Accessor for the major radius
  KOKKOS_FUNCTION
  const scalar_t& radius() const {
    return radius_;
  }

  /// \brief Accessor for the major radius
  KOKKOS_FUNCTION
  scalar_t& radius() {
    return radius_;
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

  /// \brief Set the major radius
  /// \param[in] radius The new major radius.
  KOKKOS_FUNCTION
  void set_radius(const scalar_t& radius) {
    radius_ = radius;
  }
  //@}

 private:
  point_t center_;
  orientation_t orientation_;
  std::conditional_t<std::is_same_v<OwnershipType, mundy::math::Ownership::Owns>, scalar_t, scalar_t&> radius_;
};

/// @brief Type trait to determine if a type is a Circle3D
template <typename T>
struct is_circle3d : std::false_type {};
//
template <typename Scalar, ValidPointType PointType, mundy::math::ValidQuaternionType QuaternionType,
          typename OwnershipType>
struct is_circle3d<Circle3D<Scalar, PointType, QuaternionType, OwnershipType>> : std::true_type {};
//
template <typename Scalar, ValidPointType PointType, mundy::math::ValidQuaternionType QuaternionType,
          typename OwnershipType>
struct is_circle3d<const Circle3D<Scalar, PointType, QuaternionType, OwnershipType>> : std::true_type {};
//
template <typename T>
constexpr bool is_circle3d_v = is_circle3d<T>::value;

/// @brief Concept to check if a type is a valid Circle3D type
template <typename Circle3DType>
concept ValidCircle3DType =
    is_circle3d_v<std::decay_t<Circle3DType>> && is_point_v<typename std::decay_t<Circle3DType>::point_t> &&
    mundy::math::is_quaternion_v<typename std::decay_t<Circle3DType>::orientation_t> &&
    is_point_v<decltype(std::declval<std::decay_t<Circle3DType>>().center())> &&
    is_point_v<decltype(std::declval<const std::decay_t<Circle3DType>>().center())> &&
    mundy::math::is_quaternion_v<decltype(std::declval<std::decay_t<Circle3DType>>().orientation())> &&
    mundy::math::is_quaternion_v<decltype(std::declval<const std::decay_t<Circle3DType>>().orientation())> &&
    requires(std::decay_t<Circle3DType> circle3d, const std::decay_t<Circle3DType> const_circle3d) {
      typename std::decay_t<Circle3DType>::scalar_t;
      typename std::decay_t<Circle3DType>::point_t;
      typename std::decay_t<Circle3DType>::orientation_t;
      { circle3d.radius() } -> std::same_as<typename std::decay_t<Circle3DType>::scalar_t&>;
      { const_circle3d.radius() } -> std::same_as<const typename std::decay_t<Circle3DType>::scalar_t&>;
    };  // ValidCircle3DType

static_assert(ValidCircle3DType<Circle3D<float>> && ValidCircle3DType<const Circle3D<float>> &&
                  ValidCircle3DType<Circle3D<double>> && ValidCircle3DType<const Circle3D<double>>,
              "Circle3D should satisfy the ValidCircle3DType concept.");

//! \name Non-member functions for ValidCircle3DType objects
//@{

/// \brief Equality operator
template <ValidCircle3DType Circle3DType1, ValidCircle3DType Circle3DType2>
KOKKOS_FUNCTION bool operator==(const Circle3DType1& circle3d1, const Circle3DType2& circle3d2) {
  return (circle3d1.radius() == circle3d2.radius()) && (circle3d1.center() == circle3d2.center()) &&
         (circle3d1.orientation() == circle3d2.orientation());
}

/// \brief Inequality operator
template <ValidCircle3DType Circle3DType1, ValidCircle3DType Circle3DType2>
KOKKOS_FUNCTION bool operator!=(const Circle3DType1& circle3d1, const Circle3DType2& circle3d2) {
  return (circle3d1.radius() != circle3d2.radius()) || (circle3d1.center() != circle3d2.center()) ||
         (circle3d1.orientation() != circle3d2.orientation());
}

/// \brief OStream operator
template <ValidCircle3DType Circle3DType>
std::ostream& operator<<(std::ostream& os, const Circle3DType& circle3d) {
  os << "{" << circle3d.center() << ":" << circle3d.orientation() << ":" << circle3d.radius() << "}";
  return os;
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_PRIMITIVES_CIRCLE3D_HPP_
