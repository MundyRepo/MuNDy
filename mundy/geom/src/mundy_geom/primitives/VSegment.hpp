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

#ifndef MUNDY_GEOM_PRIMITIVES_VSEGMENT_HPP_
#define MUNDY_GEOM_PRIMITIVES_VSEGMENT_HPP_

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
          typename OwnershipType = mundy::math::Ownership::Owns>
class VSegment {
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

  /// \brief Default constructor for owning VSegments. Default initialize the start, middle, and end points.
  KOKKOS_FUNCTION
  VSegment()
    requires std::is_same_v<OwnershipType, mundy::math::Ownership::Owns>
      : start_(scalar_t(), scalar_t(), scalar_t()),
        middle_(scalar_t(), scalar_t(), scalar_t()),
        end_(scalar_t(), scalar_t(), scalar_t()) {
  }

  /// \brief No default constructor for viewing VSegmentss.
  KOKKOS_FUNCTION
  VSegment()
    requires std::is_same_v<OwnershipType, mundy::math::Ownership::Views>
  = delete;

  /// \brief Constructor to initialize the start, middle, and end points.
  /// \param[in] start The start of the VSegment.
  /// \param[in] middle The middle of the VSegment.
  /// \param[in] end The end of the VSegment.
  KOKKOS_FUNCTION
  VSegment(const point_t& start, const point_t& middle, const point_t& end)
      : start_(start), middle_(middle), end_(end) {
  }

  /// \brief Constructor to initialize the start, middle, and end points.
  /// \param[in] start The start of the VSegment.
  /// \param[in] middle The middle of the VSegment.
  /// \param[in] end The end of the VSegment.
  template <ValidPointType OtherPointType>
  KOKKOS_FUNCTION VSegment(const OtherPointType& start, const OtherPointType& middle, const OtherPointType& end)
      : start_(start), middle_(middle), end_(end) {
  }

  /// \brief Destructor
  KOKKOS_DEFAULTED_FUNCTION
  ~VSegment() = default;

  /// \brief Deep copy constructor
  KOKKOS_FUNCTION
  VSegment(const VSegment<scalar_t, point_t, ownership_t>& other)
      : start_(other.start_), middle_(other.middle_), end_(other.end_) {
  }

  /// \brief Deep copy constructor
  template <typename OtherVSegmentType>
  KOKKOS_FUNCTION VSegment(const OtherVSegmentType& other)
    requires(!std::is_same_v<OtherVSegmentType, VSegment<scalar_t, point_t, ownership_t>>)
      : start_(other.start_), middle_(other.middle_), end_(other.end_) {
  }

  /// \brief Deep move constructor
  KOKKOS_FUNCTION
  VSegment(VSegment<scalar_t, point_t, ownership_t>&& other)
      : start_(std::move(other.start_)), middle_(std::move(other.middle_)), end_(std::move(other.end_)) {
  }

  /// \brief Deep move constructor
  template <typename OtherVSegmentType>
  KOKKOS_FUNCTION VSegment(OtherVSegmentType&& other)
    requires(!std::is_same_v<OtherVSegmentType, VSegment<scalar_t, point_t, ownership_t>>)
      : start_(std::move(other.start_)), middle_(std::move(other.middle_)), end_(std::move(other.end_)) {
  }
  //@}

  //! \name Operators
  //@{

  /// \brief Copy assignment operator
  KOKKOS_FUNCTION
  VSegment<scalar_t, point_t, ownership_t>& operator=(const VSegment<scalar_t, point_t, ownership_t>& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    start_ = other.start_;
    middle_ = other.middle_;
    end_ = other.end_;
    return *this;
  }

  /// \brief Copy assignment operator
  template <typename OtherVSegmentType>
  KOKKOS_FUNCTION VSegment<scalar_t, point_t, ownership_t>& operator=(const OtherVSegmentType& other)
    requires(!std::is_same_v<OtherVSegmentType, VSegment<scalar_t, point_t, ownership_t>>)
  {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    start_ = other.start_;
    middle_ = other.middle_;
    end_ = other.end_;
    return *this;
  }

  /// \brief Move assignment operator
  KOKKOS_FUNCTION
  VSegment<scalar_t, point_t, ownership_t>& operator=(VSegment<scalar_t, point_t, ownership_t>&& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    start_ = std::move(other.start_);
    middle_ = std::move(other.middle_);
    end_ = std::move(other.end_);
    return *this;
  }

  /// \brief Move assignment operator
  template <typename OtherVSegmentType>
  KOKKOS_FUNCTION VSegment<scalar_t, point_t, ownership_t>& operator=(OtherVSegmentType&& other)
    requires(!std::is_same_v<OtherVSegmentType, VSegment<scalar_t, point_t, ownership_t>>)
  {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    start_ = std::move(other.start_);
    middle_ = std::move(other.middle_);
    end_ = std::move(other.end_);
    return *this;
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Accessor for the start
  KOKKOS_FUNCTION
  const point_t& start() const {
    return start_;
  }

  /// \brief Accessor for the start
  KOKKOS_FUNCTION
  point_t& start() {
    return start_;
  }

  /// \brief Accessor for the middle
  KOKKOS_FUNCTION
  const point_t& middle() const {
    return middle_;
  }

  /// \brief Accessor for the middle
  KOKKOS_FUNCTION
  point_t& middle() {
    return middle_;
  }

  /// \brief Accessor for the end
  KOKKOS_FUNCTION
  const point_t& end() const {
    return end_;
  }

  /// \brief Accessor for the end
  KOKKOS_FUNCTION
  point_t& end() {
    return end_;
  }
  //@}

  //! \name Setters
  //@{

  /// \brief Set the start point
  /// \param[in] start The new start point.
  template <ValidPointType OtherPointType>
  KOKKOS_FUNCTION void set_start(const OtherPointType& start) {
    start_ = start;
  }

  /// \brief Set the start point
  /// \param[in] x The x-coordinate.
  /// \param[in] y The y-coordinate.
  /// \param[in] z The z-coordinate.
  KOKKOS_FUNCTION
  void set_start(const Scalar& x, const Scalar& y, const Scalar& z) {
    start_[0] = x;
    start_[1] = y;
    start_[2] = z;
  }

  /// \brief Set the middle point
  /// \param[in] middle The new middle point.
  template <ValidPointType OtherPointType>
  KOKKOS_FUNCTION void set_middle(const OtherPointType& middle) {
    middle_ = middle;
  }

  /// \brief Set the middle point
  /// \param[in] x The x-coordinate.
  /// \param[in] y The y-coordinate.
  /// \param[in] z The z-coordinate.
  KOKKOS_FUNCTION
  void set_middle(const Scalar& x, const Scalar& y, const Scalar& z) {
    middle_[0] = x;
    middle_[1] = y;
    middle_[2] = z;
  }

  /// \brief Set the end point
  /// \param[in] end The new end point.
  template <ValidPointType OtherPointType>
  KOKKOS_FUNCTION void set_end(const OtherPointType& end) {
    end_ = end;
  }

  /// \brief Set the end point
  /// \param[in] x The x-coordinate.
  /// \param[in] y The y-coordinate.
  /// \param[in] z The z-coordinate.
  KOKKOS_FUNCTION
  void set_end(const Scalar& x, const Scalar& y, const Scalar& z) {
    end_[0] = x;
    end_[1] = y;
    end_[2] = z;
  }
  //@}

 private:
  point_t start_;
  point_t middle_;
  point_t end_;
};

/// @brief Type trait to determine if a type is a VSegment
template <typename T>
struct is_v_segment : std::false_type {};
//
template <typename Scalar, ValidPointType PointType, typename OwnershipType>
struct is_v_segment<VSegment<Scalar, PointType, OwnershipType>> : std::true_type {};
//
template <typename Scalar, ValidPointType PointType, typename OwnershipType>
struct is_v_segment<const VSegment<Scalar, PointType, OwnershipType>> : std::true_type {};
//
template <typename T>
inline constexpr bool is_v_segment_v = is_v_segment<T>::value;

/// @brief Concept to check if a type is a valid VSegment type
template <typename VSegmentType>
concept ValidVSegmentType = is_v_segment_v<std::remove_cv_t<VSegmentType>> &&
                            is_point_v<decltype(std::declval<std::remove_cv_t<VSegmentType>>().start())> &&
                            is_point_v<decltype(std::declval<std::remove_cv_t<VSegmentType>>().middle())> &&
                            is_point_v<decltype(std::declval<std::remove_cv_t<VSegmentType>>().end())> &&
                            is_point_v<decltype(std::declval<const std::remove_cv_t<VSegmentType>>().start())> &&
                            is_point_v<decltype(std::declval<const std::remove_cv_t<VSegmentType>>().middle())> &&
                            is_point_v<decltype(std::declval<const std::remove_cv_t<VSegmentType>>().end())> &&
                            requires(std::remove_cv_t<VSegmentType> line) {
                              typename std::remove_cv_t<VSegmentType>::scalar_t;
                            };  // ValidVSegmentType

static_assert(ValidVSegmentType<VSegment<float>> && ValidVSegmentType<const VSegment<float>> &&
                  ValidVSegmentType<VSegment<double>> && ValidVSegmentType<const VSegment<double>>,
              "VSegment should satisfy the ValidVSegmentType concept");

//! \name Non-member functions for ValidVSegmentType objects
//@{

/// \brief Equality operator
template <ValidVSegmentType VSegmentType1, ValidVSegmentType VSegmentType2>
KOKKOS_FUNCTION bool operator==(const VSegmentType1& v_segment1, const VSegmentType2& v_segment2) {
  return (v_segment1.start() == v_segment2.start()) && (v_segment1.middle() == v_segment2.middle()) &&
         (v_segment1.end() == v_segment2.end());
}

/// \brief Inequality operator
template <ValidVSegmentType VSegmentType1, ValidVSegmentType VSegmentType2>
KOKKOS_FUNCTION bool operator!=(const VSegmentType1& v_segment1, const VSegmentType2& v_segment2) {
  return (v_segment1.start() != v_segment2.start()) || (v_segment1.middle() != v_segment2.middle()) ||
         (v_segment1.end() != v_segment2.end());
}

/// \brief OStream operator
template <ValidVSegmentType VSegmentType>
std::ostream& operator<<(std::ostream& os, const VSegmentType& v_segment) {
  os << "{" << v_segment.start() << "->" << v_segment.middle() << "->" << v_segment.end() << "}";
  return os;
}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_PRIMITIVES_VSEGMENT_HPP_
