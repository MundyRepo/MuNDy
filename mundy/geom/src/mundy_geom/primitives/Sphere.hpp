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
#include <mundy_core/throw_assert.hpp>      // for MUNDY_THROW_ASSERT
#include <mundy_geom/primitives/Point.hpp>  // for mundy::geom::Point

namespace mundy {

namespace geom {

template <typename Scalar, ValidPointType PointType = Point<Scalar>, typename OwnershipType = math::Ownership::Owns>
class Sphere {
  static_assert(std::is_same_v<typename PointType::scalar_t, Scalar>,
                "The scalar type of the PointType must match the scalar type of the Sphere.");
  static_assert(std::is_same_v<typename PointType::ownership_t, OwnershipType>,
                "The ownership type of the PointType must match the ownership type of the Sphere.\n"
                "This is somewhat restrictive, and we may want to relax this constraint in the future.\n"
                "If you need to use a different ownership type, please let us know and we'll remove this restriction.");

 public:
  //! \name Type aliases
  //@{

  /// \brief Our scalar type
  using scalar_t = Scalar;

  /// \brief Our point type
  using point_t = PointType;

  /// \brief Our ownership type
  using ownership_t = OwnershipType;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor for owning Spheres. Default initializes the center and sets the radius to an invalid
  /// value of -1
  KOKKOS_FUNCTION
  constexpr Sphere()
    requires std::is_same_v<OwnershipType, math::Ownership::Owns>
      : center_(scalar_t(), scalar_t(), scalar_t()), radius_(static_cast<scalar_t>(-1)) {
  }

  /// \brief No default constructor for viewing Spheres.
  KOKKOS_FUNCTION
  constexpr Sphere()
    requires std::is_same_v<OwnershipType, math::Ownership::Views>
  = delete;

  /// \brief Constructor to initialize the center and radius.
  /// \param[in] center The center of the Sphere.
  /// \param[in] radius The radius of the Sphere.
  KOKKOS_FUNCTION
  constexpr Sphere(const point_t& center, const scalar_t& radius) : center_(center), radius_(radius) {
  }

  /// \brief Constructor to initialize the center and radius.
  /// \param[in] center The center of the Sphere.
  /// \param[in] radius The radius of the Sphere.
  template <typename OtherPointType>
  KOKKOS_FUNCTION constexpr Sphere(const OtherPointType& center, const scalar_t& radius)
    requires(!std::is_same_v<OtherPointType, point_t>)
      : center_(center), radius_(radius) {
  }

  /// \brief Destructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr ~Sphere() = default;

  /// \brief Deep copy constructor
  KOKKOS_FUNCTION
  constexpr Sphere(const Sphere<scalar_t, point_t, ownership_t>& other)
      : center_(other.center_), radius_(other.radius_) {
  }

  /// \brief Deep copy constructor with different sphere type
  template <typename OtherSphereType>
  KOKKOS_FUNCTION constexpr Sphere(const OtherSphereType& other)
    requires(!std::is_same_v<OtherSphereType, Sphere<scalar_t, point_t, ownership_t>>)
      : center_(other.center_), radius_(other.radius_) {
  }

  /// \brief Deep move constructor
  KOKKOS_FUNCTION
  constexpr Sphere(Sphere<scalar_t, point_t, ownership_t>&& other)
      : center_(std::move(other.center_)), radius_(std::move(other.radius_)) {
  }

  /// \brief Deep move constructor
  template <typename OtherSphereType>
  KOKKOS_FUNCTION constexpr Sphere(OtherSphereType&& other)
    requires(!std::is_same_v<OtherSphereType, Sphere<scalar_t, point_t, ownership_t>>)
      : center_(std::move(other.center_)), radius_(std::move(other.radius_)) {
  }
  //@}

  //! \name Operators
  //@{

  /// \brief Copy assignment operator
  KOKKOS_FUNCTION
  constexpr Sphere<scalar_t, point_t, ownership_t>& operator=(const Sphere<scalar_t, point_t, ownership_t>& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = other.center_;
    radius_ = other.radius_;
    return *this;
  }

  /// \brief Copy assignment operator
  template <typename OtherSphereType>
  KOKKOS_FUNCTION constexpr Sphere<scalar_t, point_t, ownership_t>& operator=(const OtherSphereType& other)
    requires(!std::is_same_v<OtherSphereType, Sphere<scalar_t, point_t, ownership_t>>)
  {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = other.center_;
    radius_ = other.radius_;
    return *this;
  }

  /// \brief Move assignment operator
  KOKKOS_FUNCTION
  constexpr Sphere<scalar_t, point_t, ownership_t>& operator=(Sphere<scalar_t, point_t, ownership_t>&& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = std::move(other.center_);
    radius_ = std::move(other.radius_);
    return *this;
  }

  /// \brief Move assignment operator
  template <typename OtherSphereType>
  KOKKOS_FUNCTION constexpr Sphere<scalar_t, point_t, ownership_t>& operator=(OtherSphereType&& other)
    requires(!std::is_same_v<OtherSphereType, Sphere<scalar_t, point_t, ownership_t>>)
  {
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
  constexpr const scalar_t& radius() const {
    return radius_;
  }

  /// \brief Accessor for the radius
  KOKKOS_FUNCTION
  constexpr scalar_t& radius() {
    return radius_;
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
  constexpr void set_center(const scalar_t& x, const scalar_t& y, const scalar_t& z) {
    center_[0] = x;
    center_[1] = y;
    center_[2] = z;
  }

  /// \brief Set the radius
  /// \param[in] radius The new radius.
  KOKKOS_FUNCTION
  constexpr void set_radius(const scalar_t& radius) {
    radius_ = radius;
  }
  //@}

 private:
  point_t center_;
  std::conditional_t<std::is_same_v<ownership_t, math::Ownership::Owns>, scalar_t, scalar_t&> radius_;
};

/// @brief (Implementation) Type trait to determine if a type is a Sphere
template <typename T>
struct is_sphere_impl : std::false_type {};
//
template <typename Scalar, ValidPointType PointType, typename OwnershipType>
struct is_sphere_impl<Sphere<Scalar, PointType, OwnershipType>> : std::true_type {};

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

/// \brief Equality operator
template <ValidSphereType SphereType1, ValidSphereType SphereType2>
KOKKOS_FUNCTION constexpr bool operator==(const SphereType1& sphere1, const SphereType2& sphere2) {
  return (sphere1.radius() == sphere2.radius()) && (sphere1.center() == sphere2.center());
}

/// \brief Inequality operator
template <ValidSphereType SphereType1, ValidSphereType SphereType2>
KOKKOS_FUNCTION constexpr bool operator!=(const SphereType1& sphere1, const SphereType2& sphere2) {
  return (sphere1.radius() != sphere2.radius()) || (sphere1.center() != sphere2.center());
}

/// \brief OStream operator
template <ValidSphereType SphereType>
std::ostream& operator<<(std::ostream& os, const SphereType& sphere) {
  os << "{" << sphere.center() << ":" << sphere.radius() << "}";
  return os;
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_PRIMITIVES_SPHERE_HPP_
