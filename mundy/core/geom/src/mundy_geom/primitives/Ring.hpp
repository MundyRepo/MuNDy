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
#include <mundy_geom/primitives/Circle3D.hpp>  // for mundy::Circle3D
#include <mundy_geom/primitives/Point.hpp>     // for mundy::Point
#include <mundy_math/Quaternion.hpp>           // for mundy::Quaternion
#include <mundy_math/Vector3.hpp>              // for mundy::Vector3
#include <mundy_utils/requires.hpp>
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_ASSERT

namespace mundy {

/// \addtogroup MundyGeomPrimitives
/// @{

template <typename Scalar, ValidPointType PointType = Point<Scalar>,
          ValidQuaternionType QuaternionType = Quaternion<Scalar>>
class Ring {
  static_assert(std::is_same_v<typename PointType::value_type, Scalar> &&
                    std::is_same_v<typename QuaternionType::value_type, Scalar>,
                "The scalar type of the PointType and QuaternionType must match the scalar type of the Ring.");

 public:
  //! \name Type aliases
  //@{

  /// \brief Our scalar type
  using value_type = Scalar;

  /// \brief Our point type
  using point_t = PointType;

  /// \brief Our orientation type
  using orientation_t = QuaternionType;

  static constexpr bool is_finite = true;

  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor for owning Rings. Default initializes the center and sets the radius to an
  /// invalid value of -1
  KOKKOS_FUNCTION
  constexpr Ring() MUNDY_REQUIRES(HasDefaultConstructor<point_t>&& HasDefaultConstructor<orientation_t>)
      : center_circle_(), minor_radius_(static_cast<value_type>(-1)) {
  }

  /// \brief Constructor to initialize the ring.
  /// \param[in] center The center of the Ring.
  /// \param[in] orientation The orientation of the Ring (as a quaternion).
  /// \param[in] major_radius The radius of the center circle of the Ring.
  /// \param[in] minor_radius The radius of the tube around said circle.
  KOKKOS_FUNCTION
  constexpr Ring(const point_t& center, const orientation_t& orientation, const value_type& major_radius,
                 const value_type& minor_radius)
      : center_circle_(center, orientation, major_radius), minor_radius_(minor_radius) {
  }

  /// \brief Constructor to initialize the ring.
  /// \param[in] center The center of the Ring.
  /// \param[in] orientation The orientation of the Ring (as a quaternion).
  /// \param[in] major_radius The radius of the center circle of the Ring.
  /// \param[in] minor_radius The radius of the tube around said circle.
  template <ValidPointType OtherPointType, ValidQuaternionType OtherQuaternionType>
  KOKKOS_FUNCTION constexpr Ring(const OtherPointType& center, const OtherQuaternionType& orientation,
                                 const value_type& major_radius, const value_type& minor_radius)
      MUNDY_REQUIRES(!std::is_same_v<OtherPointType, point_t> || !std::is_same_v<OtherQuaternionType, orientation_t>)
      : center_circle_(center, orientation, major_radius), minor_radius_(minor_radius) {
  }

  /// \brief Destructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr ~Ring() = default;

  /// \brief Deep copy constructor
  KOKKOS_FUNCTION
  constexpr Ring(const Ring<value_type, point_t, orientation_t>& other)
      : center_circle_(other.center_circle_), minor_radius_(other.minor_radius_) {
  }

  /// \brief Deep copy constructor with different ring type
  template <typename OtherRingType>
  KOKKOS_FUNCTION constexpr Ring(const OtherRingType& other)
      MUNDY_REQUIRES(!std::is_same_v<OtherRingType, Ring<value_type, point_t, orientation_t>>)
      : center_circle_(other.center_circle_), minor_radius_(other.minor_radius_) {
  }

  /// \brief Deep move constructor
  KOKKOS_FUNCTION
  constexpr Ring(Ring<value_type, point_t, orientation_t>&& other)
      : center_circle_(std::move(other.center_circle_)), minor_radius_(std::move(other.minor_radius_)) {
  }

  /// \brief Deep move constructor
  template <typename OtherRingType>
  KOKKOS_FUNCTION constexpr Ring(OtherRingType&& other)
      MUNDY_REQUIRES(!std::is_same_v<OtherRingType, Ring<value_type, point_t, orientation_t>>)
      : center_circle_(std::move(other.center_circle_)), minor_radius_(std::move(other.minor_radius_)) {
  }
  //@}

  //! \name Operators
  //@{

  /// \brief Copy assignment operator
  KOKKOS_FUNCTION
  constexpr Ring<value_type, point_t, orientation_t>& operator=(const Ring<value_type, point_t, orientation_t>& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_circle_ = other.center_circle_;
    minor_radius_ = other.minor_radius_;
    return *this;
  }

  /// \brief Copy assignment operator
  template <typename OtherRingType>
  KOKKOS_FUNCTION constexpr Ring<value_type, point_t, orientation_t>& operator=(const OtherRingType& other)
      MUNDY_REQUIRES(!std::is_same_v<OtherRingType, Ring<value_type, point_t, orientation_t>>) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_circle_ = other.center_circle_;
    minor_radius_ = other.minor_radius_;
    return *this;
  }

  /// \brief Move assignment operator
  KOKKOS_FUNCTION
  constexpr Ring<value_type, point_t, orientation_t>& operator=(Ring<value_type, point_t, orientation_t>&& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_circle_ = std::move(other.center_circle_);
    minor_radius_ = std::move(other.minor_radius_);
    return *this;
  }

  /// \brief Move assignment operator
  template <typename OtherRingType>
  KOKKOS_FUNCTION constexpr Ring<value_type, point_t, orientation_t>& operator=(OtherRingType&& other)
      MUNDY_REQUIRES(!std::is_same_v<OtherRingType, Ring<value_type, point_t, orientation_t>>) {
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
  constexpr const Circle3D<value_type, point_t, orientation_t>& center_circle() const {
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
  constexpr const value_type& major_radius() const {
    return center_circle_.radius();
  }

  /// \brief Accessor for the major radius
  KOKKOS_FUNCTION
  constexpr value_type& major_radius() {
    return center_circle_.radius();
  }

  /// \brief Accessor for the minor radius
  KOKKOS_FUNCTION
  constexpr const value_type& minor_radius() const {
    return minor_radius_;
  }

  /// \brief Accessor for the minor radius
  KOKKOS_FUNCTION
  constexpr value_type& minor_radius() {
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
  constexpr void set_center(const value_type& x, const value_type& y, const value_type& z) {
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
  constexpr void set_orientation(const value_type& qw, const value_type& qx, const value_type& qy, const value_type& qz) {
    center_circle_.set_orientation(qw, qx, qy, qz);
  }

  /// \brief Set the major radius
  /// \param[in] major_radius The new major radius.
  KOKKOS_FUNCTION
  constexpr void set_major_radius(const value_type& major_radius) {
    center_circle_.set_radius(major_radius);
  }

  /// \brief Set the minor radius
  /// \param[in] minor_radius The new minor radius.
  KOKKOS_FUNCTION
  constexpr void set_minor_radius(const value_type& minor_radius) {
    minor_radius_ = minor_radius;
  }
  //@}

 private:
  Circle3D<value_type, point_t, orientation_t> center_circle_;
  value_type minor_radius_;
};

/// @brief (Implementation) Type trait to determine if a type is a Ring
template <typename T>
struct impl_is_ring : std::false_type {};
//
template <typename Scalar, ValidPointType PointType, ValidQuaternionType QuaternionType>
struct impl_is_ring<Ring<Scalar, PointType, QuaternionType>> : std::true_type {};

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

/// \brief Element-wise approximate equality (within a tolerance)
template <ValidRingType T1, ValidRingType T2>
KOKKOS_FUNCTION constexpr bool is_close(
    const T1& r1, const T2& r2,
    typename T1::value_type tol = get_comparison_tolerance<typename T1::value_type, typename T2::value_type>()) {
  return is_close(r1.major_radius(), r2.major_radius(), tol) &&
         is_close(r1.minor_radius(), r2.minor_radius(), tol) && is_close(r1.center(), r2.center(), tol) &&
         is_close(r1.orientation(), r2.orientation(), tol);
}

/// \brief Element-wise approximate equality (within a relaxed tolerance)
template <ValidRingType T1, ValidRingType T2>
KOKKOS_FUNCTION constexpr bool is_approx_close(
    const T1& r1, const T2& r2,
    typename T1::value_type tol = get_relaxed_comparison_tolerance<typename T1::value_type, typename T2::value_type>()) {
  return is_close(r1, r2, tol);
}

/// \brief OStream operator
template <ValidRingType RingType>
std::ostream& operator<<(std::ostream& os, const RingType& ring) {
  os << "{" << ring.center() << ":" << ring.orientation() << ":" << ring.major_radius() << ":" << ring.minor_radius()
     << "}";
  return os;
}
//@}

/// @}

//! \name Point visitation
//@{

/// \brief Visit the geometric point of a Ring (its center).
template <ValidRingType T, typename Functor>
KOKKOS_INLINE_FUNCTION void for_each_point(const T& r, Functor&& f) {
  f(r.center());
}

/// \brief Visit and mutate the geometric point of a Ring.
template <ValidRingType T, typename Functor>
KOKKOS_INLINE_FUNCTION void for_each_point_mutable(T& r, Functor&& f) {
  f(r.center());
}

//@}

}  // namespace mundy

#endif  // MUNDY_GEOM_PRIMITIVES_RING_HPP_
