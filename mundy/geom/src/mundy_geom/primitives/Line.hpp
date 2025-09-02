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

#ifndef MUNDY_GEOM_PRIMITIVES_LINE_HPP_
#define MUNDY_GEOM_PRIMITIVES_LINE_HPP_

// External libs
#include <Kokkos_Core.hpp>

// C++ core
#include <iostream>
#include <stdexcept>
#include <utility>

// Our libs
#include <mundy_core/throw_assert.hpp>      // for MUNDY_THROW_ASSERT
#include <mundy_geom/primitives/Point.hpp>  // for mundy::geom::Point
#include <mundy_math/Vector3.hpp>           // for mundy::math::Vector3

namespace mundy {

namespace geom {

template <typename Scalar, ValidPointType PointType = Point<Scalar>,
          typename OwnershipType = math::Ownership::Owns>
class Line {
  static_assert(std::is_same_v<typename PointType::scalar_t, Scalar>,
                "The scalar_t of the PointType must match the Scalar type.");
  static_assert(std::is_same_v<typename PointType::ownership_t, OwnershipType>,
                "The ownership type of the PointType must match the OwnershipType.\n"
                "This is somewhat restrictive, and we may want to relax this constraint in the future.\n"
                "If you need to use a different ownership type, please let us know and we'll remove this restriction.");

 public:
  //! \name Type aliases
  //@{

  /// \brief Our scalar type
  using scalar_t = Scalar;

  /// \brief Our point type
  using point_t = PointType;

  /// \brief Our vector type
  using vector_t = PointType;

  /// \brief Our ownership type
  using ownership_t = OwnershipType;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor for owning Lines. Default initialize the
  KOKKOS_FUNCTION
  constexpr Line()
    requires std::is_same_v<OwnershipType, math::Ownership::Owns>
      : center_(scalar_t(), scalar_t(), scalar_t()), direction_(scalar_t(), scalar_t(), scalar_t()) {
  }

  /// \brief No default constructor for viewing Lines.
  KOKKOS_FUNCTION
  constexpr Line()
    requires std::is_same_v<OwnershipType, math::Ownership::Views>
  = delete;

  /// \brief Constructor to initialize the center and radius.
  /// \param[in] center The center of the Line.
  /// \param[in] direction The direction of the Line.
  KOKKOS_FUNCTION
  constexpr Line(const point_t& center, const vector_t& direction) : center_(center), direction_(direction) {
  }

  /// \brief Constructor to initialize the center and radius.
  /// \param[in] center The center of the Line.
  /// \param[in] direction The direction of the Line.
  template <ValidPointType OtherPointType, math::ValidVectorType OtherVectorType>
  KOKKOS_FUNCTION constexpr Line(const OtherPointType& center, const OtherVectorType& direction)
    requires(!std::is_same_v<OtherPointType, point_t> || !std::is_same_v<OtherVectorType, vector_t>)
      : center_(center), direction_(direction) {
  }

  /// \brief Destructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr ~Line() = default;

  /// \brief Deep copy constructor
  KOKKOS_FUNCTION
  constexpr Line(const Line<scalar_t, point_t, ownership_t>& other) : center_(other.center_), direction_(other.direction_) {
  }

  /// \brief Deep copy constructor
  template <typename OtherLineType>
  KOKKOS_FUNCTION constexpr Line(const OtherLineType& other)
    requires(!std::is_same_v<OtherLineType, Line<scalar_t, point_t, ownership_t>>)
      : center_(other.center_), direction_(other.direction_) {
  }

  /// \brief Deep move constructor
  KOKKOS_FUNCTION
  constexpr Line(Line<scalar_t, point_t, ownership_t>&& other)
      : center_(std::move(other.center_)), direction_(std::move(other.direction_)) {
  }

  /// \brief Deep move constructor
  template <typename OtherLineType>
  KOKKOS_FUNCTION constexpr Line(OtherLineType&& other)
    requires(!std::is_same_v<OtherLineType, Line<scalar_t, point_t, ownership_t>>)
      : center_(std::move(other.center_)), direction_(std::move(other.direction_)) {
  }
  //@}

  //! \name Operators
  //@{

  /// \brief Copy assignment operator
  KOKKOS_FUNCTION
  constexpr Line<scalar_t, point_t, ownership_t>& operator=(const Line<scalar_t, point_t, ownership_t>& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = other.center_;
    direction_ = other.direction_;
    return *this;
  }

  /// \brief Copy assignment operator
  template <typename OtherLineType>
  KOKKOS_FUNCTION constexpr Line<scalar_t, point_t, ownership_t>& operator=(const OtherLineType& other)
    requires(!std::is_same_v<OtherLineType, Line<scalar_t, point_t, ownership_t>>)
  {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = other.center_;
    direction_ = other.direction_;
    return *this;
  }

  /// \brief Move assignment operator
  KOKKOS_FUNCTION
  constexpr Line<scalar_t, point_t, ownership_t>& operator=(Line<scalar_t, point_t, ownership_t>&& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = std::move(other.center_);
    direction_ = std::move(other.direction_);
    return *this;
  }

  /// \brief Move assignment operator
  template <typename OtherLineType>
  KOKKOS_FUNCTION constexpr Line<scalar_t, point_t, ownership_t>& operator=(OtherLineType&& other)
    requires(!std::is_same_v<OtherLineType, Line<scalar_t, point_t, ownership_t>>)
  {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    center_ = std::move(other.center_);
    direction_ = std::move(other.direction_);
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

  /// \brief Accessor for the direction
  KOKKOS_FUNCTION
  constexpr const vector_t& direction() const {
    return direction_;
  }

  /// \brief Accessor for the direction
  KOKKOS_FUNCTION
  constexpr vector_t& direction() {
    return direction_;
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
  constexpr void set_center(const Scalar& x, const Scalar& y, const Scalar& z) {
    center_[0] = x;
    center_[1] = y;
    center_[2] = z;
  }

  /// \brief Set the direction
  /// \param[in] direction The new direction.
  template <math::ValidVectorType OtherVectorType>
  KOKKOS_FUNCTION constexpr void set_direction(const OtherVectorType& direction) {
    direction_ = direction;
  }

  /// \brief Set the direction
  /// \param[in] x The x-component.
  /// \param[in] y The y-component.
  /// \param[in] z The z-component.
  KOKKOS_FUNCTION
  constexpr void set_direction(const Scalar& x, const Scalar& y, const Scalar& z) {
    direction_[0] = x;
    direction_[1] = y;
    direction_[2] = z;
  }
  //@}

 private:
  point_t center_;      ///< The center of the line.
  vector_t direction_;  ///< The direction of the line.
};

/// @brief (Implementation) Type trait to determine if a type is a Line
template <typename T>
struct impl_is_line : std::false_type {};
//
template <typename Scalar, ValidPointType PointType, typename OwnershipType>
struct impl_is_line<Line<Scalar, PointType, OwnershipType>> : std::true_type {};

/// \brief Type trait to determine if a type is a Line
template<typename T>
struct is_line : impl_is_line<std::remove_cv_t<T>> {};
//
template <typename T>
inline constexpr bool is_line_v = is_line<T>::value;

/// @brief Concept to check if a type is a valid Line type
template <typename LineType>
concept ValidLineType = is_line_v<LineType>;

static_assert(ValidLineType<Line<float>> && ValidLineType<const Line<float>> && ValidLineType<Line<double>> &&
                  ValidLineType<const Line<double>>,
              "Line should satisfy the ValidLineType concept");

//! \name Non-member functions for ValidLineType objects
//@{

/// \brief Equality operator
template <ValidLineType LineType1, ValidLineType LineType2>
KOKKOS_FUNCTION constexpr bool operator==(const LineType1& line1, const LineType2& line2) {
  return (line1.center() == line2.center()) && (line1.direction() == line2.direction());
}

/// \brief Inequality operator
template <ValidLineType LineType1, ValidLineType LineType2>
KOKKOS_FUNCTION constexpr bool operator!=(const LineType1& line1, const LineType2& line2) {
  return (line1.center() != line2.center()) || (line1.direction() != line2.direction());
}

/// \brief Output stream operator
template <ValidLineType LineType>
std::ostream& operator<<(std::ostream& os, const LineType& line) {
  os << "{" << line.center() << ":" << line.direction() << "}";
  return os;
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_PRIMITIVES_LINE_HPP_
