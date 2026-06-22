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
#include <mundy_geom/primitives/Point.hpp>  // for mundy::Point
#include <mundy_math/Quaternion.hpp>        // for mundy::Quaternion
#include <mundy_math/Vector3.hpp>           // for mundy::Vector3
#include <mundy_math/cmath.hpp>             // for mundy::impl::passive_value
#include <mundy_utils/requires.hpp>
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_ASSERT

namespace mundy {

/// \addtogroup MundyGeomPrimitives
/// @{

template <typename Scalar, ValidPointType PointType = Point<Scalar>,
          ValidQuaternionType QuaternionType = Quaternion<Scalar>>
class Circle3D {
  static_assert(std::is_same_v<typename PointType::value_type, Scalar> &&
                    std::is_same_v<typename QuaternionType::value_type, Scalar>,
                "The scalar type of the PointType and QuaternionType must match the scalar type of the Circle3D.");

 public:
  //! \name Type aliases
  //@{

  /// \brief Our scalar type
  using value_type = Scalar;

  /// \brief Our point type
  using point_t = PointType;

  /// \brief Our orientation type
  using orientation_t = QuaternionType;

  /// \brief Our deep-copy (owning) type
  using deep_copy_t = Circle3D<Scalar>;

  static constexpr bool is_finite = true;

  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor. Initializes as invalid.
  KOKKOS_FUNCTION
  constexpr Circle3D() MUNDY_REQUIRES(HasDefaultConstructor<point_t>&& HasDefaultConstructor<orientation_t>)
      : center_(), orientation_(), radius_(static_cast<value_type>(-1)) {
  }

  /// \brief Constructor to initialize the circle3d.
  /// \param[in] center The center (in the lab frame) of the Circle3D.
  /// \param[in] orientation The quaternion orientation mapping a circle with normal in the z-direction to the lab
  /// frame.
  /// \param[in] radius The radius of the circle.
  KOKKOS_FUNCTION
  constexpr Circle3D(const point_t& center, const orientation_t& orientation, const value_type& radius)
      : center_(center), orientation_(orientation), radius_(radius) {
  }

  /// \brief Constructor to initialize the circle3d.
  /// \param[in] center The center (in the lab frame) of the Circle3D.
  /// \param[in] orientation The quaternion orientation mapping a circle with normal in the z-direction to the lab
  /// frame.
  /// \param[in] radius The radius of the circle.
  template <ValidPointType OtherPointType, ValidQuaternionType OtherQuaternionType>
  KOKKOS_FUNCTION constexpr Circle3D(const OtherPointType& center, const OtherQuaternionType& orientation,
                                     const value_type& radius)
      MUNDY_REQUIRES(!std::is_same_v<OtherPointType, point_t> || !std::is_same_v<OtherQuaternionType, orientation_t>)
      : center_(center), orientation_(orientation), radius_(radius) {
  }

  /// \brief Destructor
  KOKKOS_DEFAULTED_FUNCTION
  ~Circle3D() = default;

  /// \brief Deep copy constructor
  KOKKOS_FUNCTION
  constexpr Circle3D(const Circle3D<value_type, point_t, orientation_t>& other)
      : center_(other.center_), orientation_(other.orientation_), radius_(other.radius_) {
  }

  /// \brief Deep copy constructor with different circle3d type
  template <typename OtherCircle3DType>
  KOKKOS_FUNCTION constexpr Circle3D(const OtherCircle3DType& other)
      MUNDY_REQUIRES(!std::is_same_v<OtherCircle3DType, Circle3D<value_type, point_t, orientation_t>>)
      : center_(other.center_), orientation_(other.orientation_), radius_(other.radius_) {
  }

  /// \brief Deep move constructor
  KOKKOS_FUNCTION
  constexpr Circle3D(Circle3D<value_type, point_t, orientation_t>&& other)
      : center_(std::move(other.center_)),
        orientation_{std::move(other.orientation_)},
        radius_(std::move(other.radius_)) {
  }

  /// \brief Deep move constructor
  template <typename OtherCircle3DType>
  KOKKOS_FUNCTION constexpr Circle3D(OtherCircle3DType&& other)
      MUNDY_REQUIRES(!std::is_same_v<OtherCircle3DType, Circle3D<value_type, point_t, orientation_t>>)
      : center_(std::move(other.center_)),
        orientation_{std::move(other.orientation_)},
        radius_(std::move(other.radius_)) {
  }
  //@}

  //! \name Operators
  //@{

  /// \brief Copy assignment operator
  KOKKOS_FUNCTION
  constexpr Circle3D<value_type, point_t, orientation_t>& operator=(
      const Circle3D<value_type, point_t, orientation_t>& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = other.center_;
    orientation_ = other.orientation_;
    radius_ = other.radius_;
    return *this;
  }

  /// \brief Copy assignment operator
  template <typename OtherCircle3DType>
  KOKKOS_FUNCTION constexpr Circle3D<value_type, point_t, orientation_t>& operator=(const OtherCircle3DType& other)
      MUNDY_REQUIRES(!std::is_same_v<OtherCircle3DType, Circle3D<value_type, point_t, orientation_t>>) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = other.center_;
    orientation_ = other.orientation_;
    radius_ = other.radius_;
    return *this;
  }

  /// \brief Move assignment operator
  KOKKOS_FUNCTION
  constexpr Circle3D<value_type, point_t, orientation_t>& operator=(Circle3D<value_type, point_t, orientation_t>&& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = std::move(other.center_);
    orientation_ = std::move(other.orientation_);
    radius_ = std::move(other.radius_);
    return *this;
  }

  /// \brief Move assignment operator
  template <typename OtherCircle3DType>
  KOKKOS_FUNCTION constexpr Circle3D<value_type, point_t, orientation_t>& operator=(OtherCircle3DType&& other)
      MUNDY_REQUIRES(!std::is_same_v<OtherCircle3DType, Circle3D<value_type, point_t, orientation_t>>) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = std::move(other.center_);
    orientation_ = std::move(other.orientation_);
    radius_ = std::move(other.radius_);
    return *this;
  }
  //@}

  //! \name Accessors
  //@{

  // clang-format off
  KOKKOS_FUNCTION constexpr const point_t& center() const { return center_; }
  KOKKOS_FUNCTION constexpr       point_t& center()       { return center_; }
  KOKKOS_FUNCTION constexpr const orientation_t& orientation() const { return orientation_; }
  KOKKOS_FUNCTION constexpr       orientation_t& orientation()       { return orientation_; }
  KOKKOS_FUNCTION constexpr const value_type& radius() const { return radius_; }
  KOKKOS_FUNCTION constexpr       value_type& radius()       { return radius_; }
  // clang-format on

  /// \brief Get a deep, owning copy of the Circle3D.
  KOKKOS_FUNCTION constexpr deep_copy_t copy() const { return *this; }
  //@}

  //! \name Setters
  //@{

  /// \brief Set the center
  /// \param[in] center The new center.
  template <ValidPointType OtherPointType>
  KOKKOS_FUNCTION constexpr void set_center(const OtherPointType& center) {
    center_ = center;
  }

  /// \brief Set the center
  /// \param[in] x The x-coordinate.
  /// \param[in] y The y-coordinate.
  /// \param[in] z The z-coordinate.
  KOKKOS_FUNCTION
  constexpr void set_center(const value_type& x, const value_type& y, const value_type& z) {
    center_[0] = x;
    center_[1] = y;
    center_[2] = z;
  }

  /// \brief Set the orientation
  /// \param[in] orientation The new orientation.
  KOKKOS_FUNCTION
  constexpr void set_orientation(const orientation_t& orientation) {
    orientation_ = orientation;
  }

  /// \brief Set the orientation
  /// \param[in] qw The scalar-component of the orientation quaternion.
  /// \param[in] qx The x-component of the orientation quaternion.
  /// \param[in] qy The y-component of the orientation quaternion.
  /// \param[in] qz The z-component of the orientation quaternion.
  KOKKOS_FUNCTION
  constexpr void set_orientation(const value_type& qw, const value_type& qx, const value_type& qy, const value_type& qz) {
    orientation_[0] = qw;
    orientation_[1] = qx;
    orientation_[2] = qy;
    orientation_[3] = qz;
  }

  /// \brief Set the major radius
  /// \param[in] radius The new major radius.
  KOKKOS_FUNCTION
  constexpr void set_radius(const value_type& radius) {
    radius_ = radius;
  }
  //@}

 private:
  point_t center_;
  orientation_t orientation_;
  value_type radius_;
};

/// @brief (Implementation) Type trait to determine if a type is a Circle3d
template <typename T>
struct is_circle3d_impl : std::false_type {};
//
template <typename Scalar, ValidPointType PointType, ValidQuaternionType QuaternionType>
struct is_circle3d_impl<Circle3D<Scalar, PointType, QuaternionType>> : std::true_type {};

/// @brief Type trait to determine if a type is a Circle3d
template <typename T>
struct is_circle3d : is_circle3d_impl<std::remove_cv_t<T>> {};
//
template <typename T>
constexpr bool is_circle3d_v = is_circle3d<T>::value;

/// @brief Concept to determine if a type is a valid Circle3D type
template <typename Circle3DType>
concept ValidCircle3DType = is_circle3d_v<Circle3DType>;

//! \name Non-member functions for ValidCircle3DType objects
//@{

/// \brief Get a deep, owning copy of a Circle3D.
template <ValidCircle3DType T>
KOKKOS_FUNCTION constexpr auto copy(const T& circle3d) {
  return circle3d.copy();
}

/// \brief Element-wise approximate equality (within a tolerance)
template <ValidCircle3DType T1, ValidCircle3DType T2>
KOKKOS_FUNCTION constexpr bool is_close(
    const T1& c1, const T2& c2,
    typename T1::value_type tol = get_comparison_tolerance<typename T1::value_type, typename T2::value_type>()) {
  return is_close(c1.radius(), c2.radius(), tol) && is_close(c1.center(), c2.center(), tol) &&
         is_close(c1.orientation(), c2.orientation(), tol);
}

/// \brief Element-wise approximate equality (within a relaxed tolerance)
template <ValidCircle3DType T1, ValidCircle3DType T2>
KOKKOS_FUNCTION constexpr bool is_approx_close(
    const T1& c1, const T2& c2,
    typename T1::value_type tol = get_relaxed_comparison_tolerance<typename T1::value_type, typename T2::value_type>()) {
  return is_close(c1, c2, tol);
}

/// \brief OStream operator
template <ValidCircle3DType Circle3DType>
std::ostream& operator<<(std::ostream& os, const Circle3DType& circle3d) {
  os << "{" << circle3d.center() << ":" << circle3d.orientation() << ":" << circle3d.radius() << "}";
  return os;
}
//@}

//! \name Point visitation
//@{

/// \brief Visit the geometric point of a Circle3D (its center).
template <ValidCircle3DType T, typename Functor>
KOKKOS_INLINE_FUNCTION void for_each_point(const T& c, Functor&& f) {
  f(c.center());
}

/// \brief Visit and mutate the geometric point of a Circle3D.
template <ValidCircle3DType T, typename Functor>
KOKKOS_INLINE_FUNCTION void for_each_point_mutable(T& c, Functor&& f) {
  f(c.center());
}

//@}

/// @}

namespace impl {

/// \brief A copy of the circle in its passive scalar type, with any derivative dropped; an ordinary
/// owning copy when the scalar is already arithmetic.
/// \internal
template <ValidCircle3DType C>
KOKKOS_FUNCTION Circle3D<passive_scalar_t<typename C::value_type>> passive_copy(const C& c) {
  using P = passive_scalar_t<typename C::value_type>;
  return Circle3D<P>(
      Point<P>(passive_value(c.center()[0]), passive_value(c.center()[1]), passive_value(c.center()[2])),
      Quaternion<P>(passive_value(c.orientation().w()), passive_value(c.orientation().x()),
                    passive_value(c.orientation().y()), passive_value(c.orientation().z())),
      passive_value(c.radius()));
}

}  // namespace impl

}  // namespace mundy

#endif  // MUNDY_GEOM_PRIMITIVES_CIRCLE3D_HPP_
