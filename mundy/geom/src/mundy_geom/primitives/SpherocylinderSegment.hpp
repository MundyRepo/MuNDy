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

#ifndef MUNDY_GEOM_PRIMITIVES_SPHEROCYLINDERSEGMENT_HPP_
#define MUNDY_GEOM_PRIMITIVES_SPHEROCYLINDERSEGMENT_HPP_

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
          typename OwnershipType = math::Ownership::Owns>
class SpherocylinderSegment {
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

  /// \brief Our ownership type
  using ownership_t = OwnershipType;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor for owning SpherocylinderSegments. Default initialize the start and end points.
  KOKKOS_FUNCTION
  constexpr SpherocylinderSegment()
    requires std::is_same_v<OwnershipType, math::Ownership::Owns>
      : start_(scalar_t(), scalar_t(), scalar_t()),
        end_(scalar_t(), scalar_t(), scalar_t()),
        radius_(static_cast<scalar_t>(-1)) {
  }

  /// \brief No default constructor for viewing SpherocylinderSegmentss.
  KOKKOS_FUNCTION
  constexpr SpherocylinderSegment()
    requires std::is_same_v<OwnershipType, math::Ownership::Views>
  = delete;

  /// \brief Constructor to initialize the start and end points.
  /// \param[in] start The start of the SpherocylinderSegment.
  /// \param[in] end The end of the SpherocylinderSegment.
  KOKKOS_FUNCTION
  constexpr SpherocylinderSegment(const point_t& start, const point_t& end, const scalar_t& radius)
      : start_(start), end_(end), radius_(radius) {
  }

  /// \brief Constructor to initialize the start and end points.
  /// \param[in] start The start of the SpherocylinderSegment.
  /// \param[in] end The end of the SpherocylinderSegment.
  template <ValidPointType OtherPointType>
  KOKKOS_FUNCTION constexpr SpherocylinderSegment(const OtherPointType& start, const OtherPointType& end, const scalar_t& radius)
      : start_(start), end_(end), radius_(radius) {
  }

  /// \brief Destructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr ~SpherocylinderSegment() = default;

  /// \brief Deep copy constructor
  KOKKOS_FUNCTION
  constexpr SpherocylinderSegment(const SpherocylinderSegment<scalar_t, point_t, ownership_t>& other)
      : start_(other.start_), end_(other.end_), radius_(other.radius_) {
  }

  /// \brief Deep copy constructor
  template <typename OtherSpherocylinderSegmentType>
  KOKKOS_FUNCTION constexpr SpherocylinderSegment(const OtherSpherocylinderSegmentType& other)
    requires(!std::is_same_v<OtherSpherocylinderSegmentType, SpherocylinderSegment<scalar_t, point_t, ownership_t>>)
      : start_(other.start_), end_(other.end_), radius_(other.radius_) {
  }

  /// \brief Deep move constructor
  KOKKOS_FUNCTION
  constexpr SpherocylinderSegment(SpherocylinderSegment<scalar_t, point_t, ownership_t>&& other)
      : start_(std::move(other.start_)), end_(std::move(other.end_)), radius_(std::move(other.radius_)) {
  }

  /// \brief Deep move constructor
  template <typename OtherSpherocylinderSegmentType>
  KOKKOS_FUNCTION constexpr SpherocylinderSegment(OtherSpherocylinderSegmentType&& other)
    requires(!std::is_same_v<OtherSpherocylinderSegmentType, SpherocylinderSegment<scalar_t, point_t, ownership_t>>)
      : start_(std::move(other.start_)), end_(std::move(other.end_)), radius_(std::move(other.radius_)) {
  }
  //@}

  //! \name Operators
  //@{

  /// \brief Copy assignment operator
  KOKKOS_FUNCTION
  constexpr SpherocylinderSegment<scalar_t, point_t, ownership_t>& operator=(
      const SpherocylinderSegment<scalar_t, point_t, ownership_t>& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    start_ = other.start_;
    end_ = other.end_;
    radius_ = other.radius_;
    return *this;
  }

  /// \brief Copy assignment operator
  template <typename OtherSpherocylinderSegmentType>
  KOKKOS_FUNCTION constexpr SpherocylinderSegment<scalar_t, point_t, ownership_t>& operator=(
      const OtherSpherocylinderSegmentType& other)
    requires(!std::is_same_v<OtherSpherocylinderSegmentType, SpherocylinderSegment<scalar_t, point_t, ownership_t>>)
  {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    start_ = other.start_;
    end_ = other.end_;
    radius_ = other.radius_;
    return *this;
  }

  /// \brief Move assignment operator
  KOKKOS_FUNCTION
  constexpr SpherocylinderSegment<scalar_t, point_t, ownership_t>& operator=(
      SpherocylinderSegment<scalar_t, point_t, ownership_t>&& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    start_ = std::move(other.start_);
    end_ = std::move(other.end_);
    radius_ = std::move(other.radius_);
    return *this;
  }

  /// \brief Move assignment operator
  template <typename OtherSpherocylinderSegmentType>
  KOKKOS_FUNCTION constexpr SpherocylinderSegment<scalar_t, point_t, ownership_t>& operator=(
      OtherSpherocylinderSegmentType&& other)
    requires(!std::is_same_v<OtherSpherocylinderSegmentType, SpherocylinderSegment<scalar_t, point_t, ownership_t>>)
  {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    start_ = std::move(other.start_);
    end_ = std::move(other.end_);
    radius_ = std::move(other.radius_);
    return *this;
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Accessor for the start
  KOKKOS_FUNCTION
  constexpr const point_t& start() const {
    return start_;
  }

  /// \brief Accessor for the start
  KOKKOS_FUNCTION
  constexpr point_t& start() {
    return start_;
  }

  /// \brief Accessor for the end
  KOKKOS_FUNCTION
  constexpr const point_t& end() const {
    return end_;
  }

  /// \brief Accessor for the end
  KOKKOS_FUNCTION
  constexpr point_t& end() {
    return end_;
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

  /// \brief Set the start point
  /// \param[in] start The new start point.
  template <ValidPointType OtherPointType>
  KOKKOS_FUNCTION constexpr void set_start(const OtherPointType& start) {
    start_ = start;
  }

  /// \brief Set the start point
  /// \param[in] x The x-coordinate.
  /// \param[in] y The y-coordinate.
  /// \param[in] z The z-coordinate.
  KOKKOS_FUNCTION
  constexpr void set_start(const Scalar& x, const Scalar& y, const Scalar& z) {
    start_[0] = x;
    start_[1] = y;
    start_[2] = z;
  }

  /// \brief Set the end point
  /// \param[in] end The new end point.
  template <ValidPointType OtherPointType>
  KOKKOS_FUNCTION constexpr void set_end(const OtherPointType& end) {
    end_ = end;
  }

  /// \brief Set the end point
  /// \param[in] x The x-coordinate.
  /// \param[in] y The y-coordinate.
  /// \param[in] z The z-coordinate.
  KOKKOS_FUNCTION
  constexpr void set_end(const Scalar& x, const Scalar& y, const Scalar& z) {
    end_[0] = x;
    end_[1] = y;
    end_[2] = z;
  }

  /// \brief Set the radius
  /// \param[in] radius The new radius.
  KOKKOS_FUNCTION
  constexpr void set_radius(const scalar_t& radius) {
    radius_ = radius;
  }
  //@}

 private:
  point_t start_;
  point_t end_;
  std::conditional_t<std::is_same_v<OwnershipType, math::Ownership::Owns>, scalar_t, scalar_t&> radius_;
};

/// @brief (Implementation) Type trait to determine if a type is a SpherocylinderSegment
template <typename T>
struct impl_is_spherocylinder_segment : std::false_type {};
//
template <typename Scalar, ValidPointType PointType, typename OwnershipType>
struct impl_is_spherocylinder_segment<SpherocylinderSegment<Scalar, PointType, OwnershipType>> : std::true_type {};

/// \brief Type trait to determine if a type is a SpherocylinderSegment
template <typename T>
struct is_spherocylinder_segment : impl_is_spherocylinder_segment<std::remove_cv_t<T>> {};
//
template <typename T>
inline constexpr bool is_spherocylinder_segment_v = is_spherocylinder_segment<T>::value;

/// @brief Concept to check if a type is a valid SpherocylinderSegment type
template <typename SpherocylinderSegmentType>
concept ValidSpherocylinderSegmentType =
    is_spherocylinder_segment_v<SpherocylinderSegmentType>;

static_assert(ValidSpherocylinderSegmentType<SpherocylinderSegment<float>> &&
                  ValidSpherocylinderSegmentType<const SpherocylinderSegment<float>> &&
                  ValidSpherocylinderSegmentType<SpherocylinderSegment<double>> &&
                  ValidSpherocylinderSegmentType<const SpherocylinderSegment<double>>,
              "SpherocylinderSegment should satisfy the ValidSpherocylinderSegmentType concept");

//! \name Non-member functions for ValidSpherocylinderSegmentType objects
//@{

/// \brief Equality operator
template <ValidSpherocylinderSegmentType SpherocylinderSegmentType1,
          ValidSpherocylinderSegmentType SpherocylinderSegmentType2>
KOKKOS_FUNCTION constexpr bool operator==(const SpherocylinderSegmentType1& spherocylinder_segment1,
                                const SpherocylinderSegmentType2& spherocylinder_segment2) {
  return (spherocylinder_segment1.radius() == spherocylinder_segment2.radius()) &&
         (spherocylinder_segment1.start() == spherocylinder_segment2.start()) &&
         (spherocylinder_segment1.end() == spherocylinder_segment2.end());
}

/// \brief Inequality operator
template <ValidSpherocylinderSegmentType SpherocylinderSegmentType1,
          ValidSpherocylinderSegmentType SpherocylinderSegmentType2>
KOKKOS_FUNCTION constexpr bool operator!=(const SpherocylinderSegmentType1& spherocylinder_segment1,
                                const SpherocylinderSegmentType2& spherocylinder_segment2) {
  return ((spherocylinder_segment1.radius() != spherocylinder_segment2.radius())) ||
         (spherocylinder_segment1.start() != spherocylinder_segment2.start()) ||
         (spherocylinder_segment1.end() != spherocylinder_segment2.end());
}

/// \brief OStream operator
template <ValidSpherocylinderSegmentType SpherocylinderSegmentType>
std::ostream& operator<<(std::ostream& os, const SpherocylinderSegmentType& spherocylinder_segment) {
  os << "{" << spherocylinder_segment.start() << "->" << spherocylinder_segment.end() << ":"
     << spherocylinder_segment.radius() << "}";
  return os;
}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_PRIMITIVES_SPHEROCYLINDERSEGMENT_HPP_
