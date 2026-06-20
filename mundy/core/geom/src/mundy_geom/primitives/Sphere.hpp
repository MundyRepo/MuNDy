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

#ifndef MUNDY_GEOM_PRIMITIVES_SPHERE_HPP_
#define MUNDY_GEOM_PRIMITIVES_SPHERE_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core
#include <iostream>
#include <stdexcept>
#include <utility>

// Our libs
#include <mundy_geom/primitives/Point.hpp>  // for mundy::Point
#include <mundy_utils/requires.hpp>
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_ASSERT

namespace mundy {

/// \addtogroup MundyGeomPrimitives
/// @{

template <typename Scalar, ValidPointType PointType = Point<Scalar>>
class Sphere {
  static_assert(std::is_same_v<typename PointType::value_type, Scalar>,
                "The scalar type of the PointType must match the scalar type of the Sphere.");

 public:
  //! \name Type aliases
  //@{

  /// \brief Our scalar type
  using value_type = Scalar;

  /// \brief Our point type
  using point_t = PointType;

  /// \brief Our deep-copy (owning) type
  using deep_copy_t = Sphere<Scalar>;

  static constexpr bool is_finite = true;

  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor for owning Spheres. Default initializes the center and sets the radius to an invalid
  /// value of -1
  KOKKOS_FUNCTION
  constexpr Sphere() MUNDY_REQUIRES(HasNArgConstructor<point_t, value_type, 3>)
      : center_(value_type(), value_type(), value_type()), radius_(static_cast<value_type>(-1)) {
  }

  /// \brief Constructor to initialize the center and radius.
  /// \param[in] center The center of the Sphere.
  /// \param[in] radius The radius of the Sphere.
  KOKKOS_FUNCTION
  constexpr Sphere(const point_t& center, const value_type& radius) : center_(center), radius_(radius) {
  }

  /// \brief Constructor to initialize the center and radius.
  /// \param[in] center The center of the Sphere.
  /// \param[in] radius The radius of the Sphere.
  template <ValidPointType OtherPointType>
  KOKKOS_FUNCTION constexpr Sphere(const OtherPointType& center, const value_type& radius)
      MUNDY_REQUIRES(!std::is_same_v<OtherPointType, point_t>)
      : center_(center), radius_(radius) {
  }

  /// \brief Destructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr ~Sphere() = default;

  /// \brief Deep copy constructor
  KOKKOS_FUNCTION
  constexpr Sphere(const Sphere<value_type, point_t>& other) : center_(other.center_), radius_(other.radius_) {
  }

  /// \brief Deep copy constructor with different sphere type
  template <typename OtherSphereType>
  KOKKOS_FUNCTION constexpr Sphere(const OtherSphereType& other)
      MUNDY_REQUIRES(!std::is_same_v<OtherSphereType, Sphere<value_type, point_t>>)
      : center_(other.center_), radius_(other.radius_) {
  }

  /// \brief Deep move constructor
  KOKKOS_FUNCTION
  constexpr Sphere(Sphere<value_type, point_t>&& other)
      : center_(std::move(other.center_)), radius_(std::move(other.radius_)) {
  }

  /// \brief Deep move constructor
  template <typename OtherSphereType>
  KOKKOS_FUNCTION constexpr Sphere(OtherSphereType&& other)
      MUNDY_REQUIRES(!std::is_same_v<OtherSphereType, Sphere<value_type, point_t>>)
      : center_(std::move(other.center_)), radius_(std::move(other.radius_)) {
  }
  //@}

  //! \name Operators
  //@{

  /// \brief Copy assignment operator
  KOKKOS_FUNCTION
  constexpr Sphere<value_type, point_t>& operator=(const Sphere<value_type, point_t>& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = other.center_;
    radius_ = other.radius_;
    return *this;
  }

  /// \brief Copy assignment operator
  template <typename OtherSphereType>
  KOKKOS_FUNCTION constexpr Sphere<value_type, point_t>& operator=(const OtherSphereType& other)
      MUNDY_REQUIRES(!std::is_same_v<OtherSphereType, Sphere<value_type, point_t>>) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = other.center_;
    radius_ = other.radius_;
    return *this;
  }

  /// \brief Move assignment operator
  KOKKOS_FUNCTION
  constexpr Sphere<value_type, point_t>& operator=(Sphere<value_type, point_t>&& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = std::move(other.center_);
    radius_ = std::move(other.radius_);
    return *this;
  }

  /// \brief Move assignment operator
  template <typename OtherSphereType>
  KOKKOS_FUNCTION constexpr Sphere<value_type, point_t>& operator=(OtherSphereType&& other)
      MUNDY_REQUIRES(!std::is_same_v<OtherSphereType, Sphere<value_type, point_t>>) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = std::move(other.center_);
    radius_ = std::move(other.radius_);
    return *this;
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Accessor for the center
  KOKKOS_FUNCTION
  constexpr const point_t& center() const {
    return center_;
  }

  /// \brief Accessor for the center
  KOKKOS_FUNCTION
  constexpr point_t& center() {
    return center_;
  }

  /// \brief Accessor for the radius
  KOKKOS_FUNCTION
  constexpr const value_type& radius() const {
    return radius_;
  }

  /// \brief Accessor for the radius
  KOKKOS_FUNCTION
  constexpr value_type& radius() {
    return radius_;
  }

  /// \brief Get a deep, owning copy of the Sphere.
  KOKKOS_FUNCTION
  constexpr deep_copy_t copy() const {
    return *this;
  }
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

  /// \brief Set the radius
  /// \param[in] radius The new radius.
  KOKKOS_FUNCTION
  constexpr void set_radius(const value_type& radius) {
    radius_ = radius;
  }
  //@}

 private:
  point_t center_;
  value_type radius_;
};

/// @brief (Implementation) Type trait to determine if a type is a Sphere
template <typename T>
struct is_sphere_impl : std::false_type {};
//
template <typename Scalar, ValidPointType PointType>
struct is_sphere_impl<Sphere<Scalar, PointType>> : std::true_type {};

/// \brief Type trait to determine if a type is a Sphere
template <typename T>
struct is_sphere : is_sphere_impl<std::remove_cv_t<T>> {};
//
template <typename T>
constexpr bool is_sphere_v = is_sphere<T>::value;

/// @brief Concept to check if a type is a valid Sphere type
template <typename SphereType>
concept ValidSphereType = is_sphere_v<SphereType>;

static_assert(ValidSphereType<Sphere<float>> && ValidSphereType<const Sphere<float>> &&
                  ValidSphereType<Sphere<double>> && ValidSphereType<const Sphere<double>>,
              "Sphere should satisfy the ValidSphereType concept.");

//! \name Non-member functions for ValidSphereType objects
//@{

/// \brief Element-wise approximate equality (within a tolerance)
template <ValidSphereType T1, ValidSphereType T2>
KOKKOS_FUNCTION constexpr bool is_close(
    const T1& s1, const T2& s2,
    typename T1::value_type tol = get_comparison_tolerance<typename T1::value_type, typename T2::value_type>()) {
  return is_close(s1.radius(), s2.radius(), tol) && is_close(s1.center(), s2.center(), tol);
}

/// \brief Element-wise approximate equality (within a relaxed tolerance)
template <ValidSphereType T1, ValidSphereType T2>
KOKKOS_FUNCTION constexpr bool is_approx_close(
    const T1& s1, const T2& s2,
    typename T1::value_type tol = get_relaxed_comparison_tolerance<typename T1::value_type, typename T2::value_type>()) {
  return is_close(s1, s2, tol);
}

/// \brief Get a deep, owning copy of a Sphere.
template <ValidSphereType T>
KOKKOS_FUNCTION constexpr auto copy(const T& sphere) {
  return sphere.copy();
}

/// \brief OStream operator
template <ValidSphereType SphereType>
std::ostream& operator<<(std::ostream& os, const SphereType& sphere) {
  os << "{" << sphere.center() << ":" << sphere.radius() << "}";
  return os;
}
//@}

/// @}

//! \name Point visitation
//@{

/// \brief Visit the geometric point of a Sphere (its center).
template <ValidSphereType T, typename Functor>
KOKKOS_INLINE_FUNCTION void for_each_point(const T& s, Functor&& f) {
  f(s.center());
}

/// \brief Visit and mutate the geometric point of a Sphere.
template <ValidSphereType T, typename Functor>
KOKKOS_INLINE_FUNCTION void for_each_point_mutable(T& s, Functor&& f) {
  f(s.center());
}

//@}

}  // namespace mundy

#endif  // MUNDY_GEOM_PRIMITIVES_SPHERE_HPP_
