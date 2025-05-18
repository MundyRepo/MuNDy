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

#ifndef MUNDY_GEOM_PRIMITIVES_ELLIPSOID_HPP_
#define MUNDY_GEOM_PRIMITIVES_ELLIPSOID_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core
#include <iostream>
#include <stdexcept>
#include <utility>

// Our libs
#include <mundy_core/throw_assert.hpp>      // for MUNDY_THROW_ASSERT
#include <mundy_geom/primitives/Point.hpp>  // for mundy::geom::Point
#include <mundy_math/Quaternion.hpp>        // for mundy::math::Quaternion
#include <mundy_math/Tolerance.hpp>         // for mundy::math::get_zero_tolerance

namespace mundy {

namespace geom {

template <typename Scalar, ValidPointType PointType = Point<Scalar>,
          mundy::math::ValidQuaternionType OrientationType = mundy::math::Quaternion<Scalar>,
          typename OwnershipType = mundy::math::Ownership::Owns>
class Ellipsoid {
  static_assert(std::is_same_v<typename PointType::scalar_t, Scalar> &&
                    std::is_same_v<typename OrientationType::scalar_t, Scalar>,
                "The scalar_t of the PointType and OrientationType must match the Scalar type.");
  static_assert(std::is_same_v<typename PointType::ownership_t, OwnershipType> &&
                    std::is_same_v<typename OrientationType::ownership_t, OwnershipType>,
                "The ownership type of the PointType and OrientationType must match the OwnershipType.\n"
                "This is somewhat restrictive, and we may want to relax this constraint in the future.\n"
                "If you need to use a different ownership type, please let us know and we'll remove this restriction.");

 public:
  //! \name Type aliases
  //@{

  /// \brief Our scalar type
  using scalar_t = Scalar;

  /// \brief Our point type
  using point_t = PointType;

  /// \brief Our quaternion type
  using orientation_t = OrientationType;

  /// \brief Our ownership type
  using ownership_t = OwnershipType;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor for owning ellipsoids. Default initializes the center and sets the axis radii to an
  /// invalid value of -1
  KOKKOS_FUNCTION
  Ellipsoid()
    requires std::is_same_v<OwnershipType, mundy::math::Ownership::Owns>
      : center_(scalar_t(), scalar_t(), scalar_t()),
        orientation_{static_cast<scalar_t>(1), static_cast<scalar_t>(0), static_cast<scalar_t>(0),
                     static_cast<scalar_t>(0)},
        radii_{static_cast<scalar_t>(-1), static_cast<scalar_t>(-1), static_cast<scalar_t>(-1)} {
  }

  /// \brief No default constructor for viewing ellipsoids.
  KOKKOS_FUNCTION
  Ellipsoid()
    requires std::is_same_v<OwnershipType, mundy::math::Ownership::Views>
  = delete;

  /// \brief Constructor to initialize the center and radii.
  /// \param[in] center The center of the Ellipsoid.
  /// \param[in] radius_1 The first axis radius of the Ellipsoid.
  /// \param[in] radius_2 The second axis radius of the Ellipsoid.
  /// \param[in] radius_3 The third axis radius of the Ellipsoid.
  KOKKOS_FUNCTION
  Ellipsoid(const point_t& center, const scalar_t& radius_1, const scalar_t& radius_2, const scalar_t& radius_3)
    requires std::is_same_v<OwnershipType, mundy::math::Ownership::Owns>
      : center_(center),
        orientation_{static_cast<scalar_t>(1), static_cast<scalar_t>(0), static_cast<scalar_t>(0),
                     static_cast<scalar_t>(0)},
        radii_{radius_1, radius_2, radius_3} {
  }

  /// \brief Constructor to initialize the center and axis radii.
  /// \param[in] x The x-coordinate of the center.
  /// \param[in] y The y-coordinate of the center.
  /// \param[in] z The z-coordinate of the center.
  /// \param[in] qw The scalar-component of the orientation quaternion.
  /// \param[in] qx The x-component of the orientation quaternion.
  /// \param[in] qy The y-component of the orientation quaternion.
  /// \param[in] qz The z-component of the orientation quaternion.
  /// \param[in] radius_1 The first axis length of the Ellipsoid.
  /// \param[in] radius_2 The second axis length of the Ellipsoid.
  /// \param[in] radius_3 The third axis length of the Ellipsoid.
  KOKKOS_FUNCTION
  Ellipsoid(const scalar_t& x, const scalar_t& y, const scalar_t& z, const scalar_t& qw, const scalar_t& qx,
            const scalar_t& qy, const scalar_t& qz, const scalar_t& radius_1, const scalar_t& radius_2,
            const scalar_t& radius_3)
    requires std::is_same_v<OwnershipType, mundy::math::Ownership::Owns>
      : center_(x, y, z), orientation_(qw, qx, qy, qz), radii_{radius_1, radius_2, radius_3} {
  }

  /// \brief Constructor to initialize the center and axis radii.
  /// \param[in] center The center of the Ellipsoid.
  /// \param[in] radii The axis radii of the Ellipsoid.
  KOKKOS_FUNCTION
  Ellipsoid(const point_t& center, const orientation_t& orientation, const point_t& radii)
      : center_(center), orientation_(orientation), radii_(radii) {
  }

  /// \brief Constructor to initialize the center and axis radii.
  /// \param[in] center The center of the Ellipsoid.
  /// \param[in] radii The axis radii of the Ellipsoid.
  KOKKOS_FUNCTION
  Ellipsoid(const point_t& center, const orientation_t& orientation, const scalar_t& radius_1, const scalar_t& radius_2,
            const scalar_t& radius_3)
    requires std::is_same_v<OwnershipType, mundy::math::Ownership::Owns>
      : center_(center), orientation_(orientation), radii_(radius_1, radius_2, radius_3) {
  }

  /// \brief Destructor
  KOKKOS_DEFAULTED_FUNCTION
  ~Ellipsoid() = default;

  /// \brief Deep copy constructor
  KOKKOS_FUNCTION
  Ellipsoid(const Ellipsoid<scalar_t, point_t, orientation_t, ownership_t>& other)
      : center_(other.center_), orientation_{other.orientation_}, radii_{other.radii_} {
  }

  /// \brief Deep copy constructor with different ellipsoid type
  template <typename OtherEllipsoidType>
  KOKKOS_FUNCTION Ellipsoid(const OtherEllipsoidType& other)
    requires(!std::is_same_v<OtherEllipsoidType, Ellipsoid<scalar_t, point_t, orientation_t, ownership_t>>)
      : center_(other.center_), orientation_{other.orientation_}, radii_{other.radii_} {
  }

  /// \brief Deep move constructor
  KOKKOS_FUNCTION
  Ellipsoid(Ellipsoid<scalar_t, point_t, orientation_t, ownership_t>&& other)
      : center_(std::move(other.center_)),
        orientation_{std::move(other.orientation_)},
        radii_{std::move(other.radii_)} {
  }

  /// \brief Deep move constructor with different ellipsoid type
  template <typename OtherEllipsoidType>
  KOKKOS_FUNCTION Ellipsoid(OtherEllipsoidType&& other)
    requires(!std::is_same_v<OtherEllipsoidType, Ellipsoid<scalar_t, point_t, orientation_t, ownership_t>>)
      : center_(std::move(other.center_)),
        orientation_{std::move(other.orientation_)},
        radii_{std::move(other.radii_)} {
  }
  //@}

  //! \name Operators
  //@{

  /// \brief Copy assignment operator
  KOKKOS_FUNCTION
  Ellipsoid<scalar_t, point_t, orientation_t, ownership_t>& operator=(
      const Ellipsoid<scalar_t, point_t, orientation_t, ownership_t>& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = other.center_;
    orientation_ = other.orientation_;
    radii_ = other.radii_;
    return *this;
  }

  /// \brief Copy assignment operator with different ellipsoid type
  template <typename OtherEllipsoidType>
  KOKKOS_FUNCTION Ellipsoid<scalar_t, point_t, orientation_t, ownership_t>& operator=(const OtherEllipsoidType& other)
    requires(!std::is_same_v<OtherEllipsoidType, Ellipsoid<scalar_t, point_t, orientation_t, ownership_t>>)
  {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = other.center_;
    orientation_ = other.orientation_;
    radii_ = other.radii_;
    return *this;
  }

  /// \brief Move assignment operator
  KOKKOS_FUNCTION
  Ellipsoid<scalar_t, point_t, orientation_t, ownership_t>& operator=(
      Ellipsoid<scalar_t, point_t, orientation_t, ownership_t>&& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = std::move(other.center_);
    orientation_ = std::move(other.orientation_);
    radii_ = std::move(other.radii_);
    return *this;
  }

  /// \brief Move assignment operator with different ellipsoid type
  template <typename OtherEllipsoidType>
  KOKKOS_FUNCTION Ellipsoid<scalar_t, point_t, orientation_t, ownership_t>& operator=(OtherEllipsoidType&& other)
    requires(!std::is_same_v<OtherEllipsoidType, Ellipsoid<scalar_t, point_t, orientation_t, ownership_t>>)
  {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = std::move(other.center_);
    orientation_ = std::move(other.orientation_);
    radii_ = std::move(other.radii_);
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

  /// \brief Accessor for the radii
  KOKKOS_FUNCTION
  const point_t& radii() const {
    return radii_;
  }

  /// \brief Accessor for the radii
  KOKKOS_FUNCTION
  point_t& radii() {
    return radii_;
  }

  /// \brief Accessor for the first axis length
  KOKKOS_FUNCTION
  const scalar_t& radius_1() const {
    return radii_[0];
  }

  /// \brief Accessor for the first axis length
  KOKKOS_FUNCTION
  scalar_t& radius_1() {
    return radii_[0];
  }

  /// \brief Accessor for the second axis length
  KOKKOS_FUNCTION
  const scalar_t& radius_2() const {
    return radii_[1];
  }

  /// \brief Accessor for the second axis length
  KOKKOS_FUNCTION
  scalar_t& radius_2() {
    return radii_[1];
  }

  /// \brief Accessor for the third axis length
  KOKKOS_FUNCTION
  const scalar_t& radius_3() const {
    return radii_[2];
  }

  /// \brief Accessor for the third axis length
  KOKKOS_FUNCTION
  scalar_t& radius_3() {
    return radii_[2];
  }
  //@}

  //! \name Setters
  //@{

  /// \brief Set the center
  /// \param[in] center The new center.
  KOKKOS_FUNCTION
  void set_center(const point_t& center) {
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

  /// \brief Set the radii
  /// \param[in] radii The new radii.
  KOKKOS_FUNCTION
  void set_radii(const point_t& radii) {
    radii_ = radii;
  }

  /// \brief Set the radii
  /// \param[in] radius_1 The new first axis radius.
  /// \param[in] radius_2 The new second axis radius.
  /// \param[in] radius_3 The new third axis radius.
  KOKKOS_FUNCTION
  void set_radii(const scalar_t& radius_1, const scalar_t& radius_2, const scalar_t& radius_3) {
    radii_[0] = radius_1;
    radii_[1] = radius_2;
    radii_[2] = radius_3;
  }
  //@}

 private:
  point_t center_;
  orientation_t orientation_;
  point_t radii_;
};

/// @brief Type trait to determine if a type is am Ellipsoid
template <typename T>
struct is_ellipsoid : std::false_type {};
//
template <typename Scalar, ValidPointType PointType, mundy::math::ValidQuaternionType OrientationType,
          typename OwnershipType>
struct is_ellipsoid<Ellipsoid<Scalar, PointType, OrientationType, OwnershipType>> : std::true_type {};
//
template <typename Scalar, ValidPointType PointType, mundy::math::ValidQuaternionType OrientationType,
          typename OwnershipType>
struct is_ellipsoid<const Ellipsoid<Scalar, PointType, OrientationType, OwnershipType>> : std::true_type {};
//
template <typename T>
inline constexpr bool is_ellipsoid_v = is_ellipsoid<T>::value;

/// @brief Concept to check if a type is an Ellipsoid
template <typename EllipsoidType>
concept ValidEllipsoidType =
    is_ellipsoid_v<std::remove_cv_t<EllipsoidType>> &&
    is_point_v<decltype(std::declval<std::remove_cv_t<EllipsoidType>>().center())> &&
    mundy::math::is_quaternion_v<decltype(std::declval<std::remove_cv_t<EllipsoidType>>().orientation())> &&
    mundy::math::is_vector3_v<decltype(std::declval<std::remove_cv_t<EllipsoidType>>().radii())> &&
    is_point_v<decltype(std::declval<const std::remove_cv_t<EllipsoidType>>().center())> &&
    mundy::math::is_quaternion_v<decltype(std::declval<const std::remove_cv_t<EllipsoidType>>().orientation())> &&
    mundy::math::is_vector3_v<decltype(std::declval<const std::remove_cv_t<EllipsoidType>>().radii())> &&
    requires(std::remove_cv_t<EllipsoidType> ellipsoid) {
      typename std::remove_cv_t<EllipsoidType>::scalar_t;
    };  // ValidEllipsoidType

static_assert(ValidEllipsoidType<Ellipsoid<float>> && ValidEllipsoidType<const Ellipsoid<float>> &&
                  ValidEllipsoidType<Ellipsoid<double>> && ValidEllipsoidType<const Ellipsoid<double>>,
              "Ellipsoid should satisfy the ValidEllipsoidType concept.");

//! \name Non-member functions for ValidSphereType objects
//@{

/// \brief Equality operator
template <ValidEllipsoidType EllipsoidType1, ValidEllipsoidType EllipsoidType2>
KOKKOS_FUNCTION bool operator==(const EllipsoidType1& ellipsoid1, const EllipsoidType2& ellipsoid2) {
  return (ellipsoid1.center() == ellipsoid2.center()) && (ellipsoid1.orientation() == ellipsoid2.orientation()) &&
         (ellipsoid1.radii() == ellipsoid2.radii());
}

/// \brief Inequality operator
template <ValidEllipsoidType EllipsoidType1, ValidEllipsoidType EllipsoidType2>
KOKKOS_FUNCTION bool operator!=(const EllipsoidType1& ellipsoid1, const EllipsoidType2& ellipsoid2) {
  return (ellipsoid1.center() != ellipsoid2.center()) || (ellipsoid1.orientation() != ellipsoid2.orientation()) ||
         (ellipsoid1.radii() != ellipsoid2.radii());
}

/// \brief OStream operator
template <ValidEllipsoidType EllipsoidType>
std::ostream& operator<<(std::ostream& os, const EllipsoidType& ellipsoid) {
  os << "{" << ellipsoid.center() << ":" << ellipsoid.orientation() << ":" << ellipsoid.radii() << "}";
  return os;
}

template <ValidEllipsoidType EllipsoidType>
KOKKOS_FUNCTION Point<typename EllipsoidType::scalar_t> map_body_frame_normal_to_ellipsoid(
    const mundy::math::Vector3<typename EllipsoidType::scalar_t>& body_frame_nhat, const EllipsoidType& ellipsoid) {
  using Scalar = typename EllipsoidType::scalar_t;
  constexpr Scalar half = static_cast<Scalar>(0.5);
  constexpr Scalar one = static_cast<Scalar>(1.0);
  constexpr Scalar zero = static_cast<Scalar>(0.0);

  const Scalar r1 = ellipsoid.radius_1();
  const Scalar r2 = ellipsoid.radius_2();
  const Scalar r3 = ellipsoid.radius_3();

  const Scalar sign0 = Kokkos::copysign(one, body_frame_nhat[0]);
  const Scalar sign1 = Kokkos::copysign(one, body_frame_nhat[1]);
  const Scalar sign2 = Kokkos::copysign(one, body_frame_nhat[2]);

  Scalar alpha1, alpha2;
  if (sign0 * body_frame_nhat[0] > mundy::math::get_zero_tolerance<Scalar>()) {
    const Scalar tmp0 = one / (r1 * body_frame_nhat[0]);
    const Scalar tmp1 = tmp0 * r2 * body_frame_nhat[1];
    const Scalar tmp2 = tmp0 * r3 * body_frame_nhat[2];
    alpha1 = one / (one + tmp1 * tmp1);
    alpha2 = one / (one + tmp2 * tmp2 * alpha1);
  } else if (sign1 * body_frame_nhat[1] > mundy::math::get_zero_tolerance<Scalar>()) {
    const Scalar tmp = r3 * body_frame_nhat[2] / (r2 * body_frame_nhat[1]);
    alpha1 = zero;
    alpha2 = one / (one + tmp * tmp);
  } else {
    alpha1 = zero;
    alpha2 = zero;
  }

  const Scalar sqrt_alpha1 = Kokkos::sqrt(alpha1);
  const Scalar sqrt_alpha2 = Kokkos::sqrt(alpha2);

  const Scalar x = half * sign0 * ((one + sign0) * r1 + (one - sign0) * r1) * sqrt_alpha1 * sqrt_alpha2;
  const Scalar y = half * sign1 * ((one + sign1) * r2 + (one - sign1) * r2) * Kokkos::sqrt(one - alpha1) * sqrt_alpha2;
  const Scalar z = half * sign2 * ((one + sign2) * r3 + (one - sign2) * r3) * Kokkos::sqrt(one - alpha2);

  return Point<Scalar>(x, y, z);
}

template <typename Scalar, typename Accessor1, typename OwnershipType1, ValidEllipsoidType EllipsoidType>
KOKKOS_FUNCTION mundy::math::Vector3<Scalar> map_surface_normal_to_foot_point_on_ellipsoid(
    const mundy::math::AVector3<Scalar, Accessor1, OwnershipType1>& lab_frame_ellipsoid_nhat,
    const EllipsoidType& ellipsoid) {
  const auto body_frame_nhat = conjugate(ellipsoid.orientation()) * lab_frame_ellipsoid_nhat;
  const auto body_frame_foot_point = map_body_frame_normal_to_ellipsoid(body_frame_nhat, ellipsoid);
  return ellipsoid.orientation() * body_frame_foot_point + ellipsoid.center();
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_PRIMITIVES_ELLIPSOID_HPP_
