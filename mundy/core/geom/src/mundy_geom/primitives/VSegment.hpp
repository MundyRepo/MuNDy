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
#include <mundy_geom/primitives/Point.hpp>  // for mundy::Point
#include <mundy_utils/requires.hpp>
#include <mundy_utils/throw_assert.hpp>  // for MUNDY_THROW_ASSERT

namespace mundy {

/// \addtogroup MundyGeomPrimitives
/// @{

template <typename Scalar, ValidPointType PointType = Point<Scalar>>
class VSegment {
  static_assert(std::is_same_v<typename PointType::value_type, Scalar>,
                "The value_type of the PointType must match the Scalar type.");

 public:
  //! \name Type aliases
  //@{

  /// \brief Our scalar type
  using value_type = Scalar;

  /// \brief Our point type
  using point_t = PointType;

  static constexpr bool is_finite = true;

  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor for owning VSegments. Default initialize the start, middle, and end points.
  KOKKOS_FUNCTION
  constexpr VSegment() MUNDY_REQUIRES(HasNArgConstructor<point_t, value_type, 3>)
      : start_(value_type(), value_type(), value_type()),
        middle_(value_type(), value_type(), value_type()),
        end_(value_type(), value_type(), value_type()) {
  }

  /// \brief Constructor to initialize the start, middle, and end points.
  /// \param[in] start The start of the VSegment.
  /// \param[in] middle The middle of the VSegment.
  /// \param[in] end The end of the VSegment.
  KOKKOS_FUNCTION
  constexpr VSegment(const point_t& start, const point_t& middle, const point_t& end)
      : start_(start), middle_(middle), end_(end) {
  }

  /// \brief Constructor to initialize the start, middle, and end points.
  /// \param[in] start The start of the VSegment.
  /// \param[in] middle The middle of the VSegment.
  /// \param[in] end The end of the VSegment.
  template <ValidPointType OtherPointType>
  KOKKOS_FUNCTION constexpr VSegment(const OtherPointType& start, const OtherPointType& middle,
                                     const OtherPointType& end)
      : start_(start), middle_(middle), end_(end) {
  }

  /// \brief Destructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr ~VSegment() = default;

  /// \brief Deep copy constructor
  KOKKOS_FUNCTION
  constexpr VSegment(const VSegment<value_type, point_t>& other)
      : start_(other.start_), middle_(other.middle_), end_(other.end_) {
  }

  /// \brief Deep copy constructor
  template <typename OtherVSegmentType>
  KOKKOS_FUNCTION constexpr VSegment(const OtherVSegmentType& other)
      MUNDY_REQUIRES(!std::is_same_v<OtherVSegmentType, VSegment<value_type, point_t>>)
      : start_(other.start_), middle_(other.middle_), end_(other.end_) {
  }

  /// \brief Deep move constructor
  KOKKOS_FUNCTION
  constexpr VSegment(VSegment<value_type, point_t>&& other)
      : start_(std::move(other.start_)), middle_(std::move(other.middle_)), end_(std::move(other.end_)) {
  }

  /// \brief Deep move constructor
  template <typename OtherVSegmentType>
  KOKKOS_FUNCTION constexpr VSegment(OtherVSegmentType&& other)
      MUNDY_REQUIRES(!std::is_same_v<OtherVSegmentType, VSegment<value_type, point_t>>)
      : start_(std::move(other.start_)), middle_(std::move(other.middle_)), end_(std::move(other.end_)) {
  }
  //@}

  //! \name Operators
  //@{

  /// \brief Copy assignment operator
  KOKKOS_FUNCTION
  constexpr VSegment<value_type, point_t>& operator=(const VSegment<value_type, point_t>& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    start_ = other.start_;
    middle_ = other.middle_;
    end_ = other.end_;
    return *this;
  }

  /// \brief Copy assignment operator
  template <typename OtherVSegmentType>
  KOKKOS_FUNCTION constexpr VSegment<value_type, point_t>& operator=(const OtherVSegmentType& other)
      MUNDY_REQUIRES(!std::is_same_v<OtherVSegmentType, VSegment<value_type, point_t>>) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    start_ = other.start_;
    middle_ = other.middle_;
    end_ = other.end_;
    return *this;
  }

  /// \brief Move assignment operator
  KOKKOS_FUNCTION
  constexpr VSegment<value_type, point_t>& operator=(VSegment<value_type, point_t>&& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    start_ = std::move(other.start_);
    middle_ = std::move(other.middle_);
    end_ = std::move(other.end_);
    return *this;
  }

  /// \brief Move assignment operator
  template <typename OtherVSegmentType>
  KOKKOS_FUNCTION constexpr VSegment<value_type, point_t>& operator=(OtherVSegmentType&& other)
      MUNDY_REQUIRES(!std::is_same_v<OtherVSegmentType, VSegment<value_type, point_t>>) {
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
  constexpr const point_t& start() const {
    return start_;
  }

  /// \brief Accessor for the start
  KOKKOS_FUNCTION
  constexpr point_t& start() {
    return start_;
  }

  /// \brief Accessor for the middle
  KOKKOS_FUNCTION
  constexpr const point_t& middle() const {
    return middle_;
  }

  /// \brief Accessor for the middle
  KOKKOS_FUNCTION
  constexpr point_t& middle() {
    return middle_;
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

  /// \brief Set the middle point
  /// \param[in] middle The new middle point.
  template <ValidPointType OtherPointType>
  KOKKOS_FUNCTION constexpr void set_middle(const OtherPointType& middle) {
    middle_ = middle;
  }

  /// \brief Set the middle point
  /// \param[in] x The x-coordinate.
  /// \param[in] y The y-coordinate.
  /// \param[in] z The z-coordinate.
  KOKKOS_FUNCTION
  constexpr void set_middle(const Scalar& x, const Scalar& y, const Scalar& z) {
    middle_[0] = x;
    middle_[1] = y;
    middle_[2] = z;
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
  //@}

 private:
  point_t start_;
  point_t middle_;
  point_t end_;
};

/// @brief (Implementation) Type trait to determine if a type is a VSegment
template <typename T>
struct impl_is_v_segment : std::false_type {};
//
template <typename Scalar, ValidPointType PointType>
struct impl_is_v_segment<VSegment<Scalar, PointType>> : std::true_type {};

/// \brief Type trait to determine if a type is a VSegment
template <typename T>
struct is_v_segment : impl_is_v_segment<std::remove_cv_t<T>> {};
//
template <typename T>
inline constexpr bool is_v_segment_v = is_v_segment<T>::value;

/// @brief Concept to check if a type is a valid VSegment type
template <typename VSegmentType>
concept ValidVSegmentType = is_v_segment_v<VSegmentType>;

static_assert(ValidVSegmentType<VSegment<float>> && ValidVSegmentType<const VSegment<float>> &&
                  ValidVSegmentType<VSegment<double>> && ValidVSegmentType<const VSegment<double>>,
              "VSegment should satisfy the ValidVSegmentType concept");

//! \name Non-member functions for ValidVSegmentType objects
//@{

/// \brief Element-wise approximate equality (within a tolerance)
template <ValidVSegmentType T1, ValidVSegmentType T2>
KOKKOS_FUNCTION constexpr bool is_close(
    const T1& vs1, const T2& vs2,
    typename T1::value_type tol = get_comparison_tolerance<typename T1::value_type, typename T2::value_type>()) {
  return is_close(vs1.start(), vs2.start(), tol) && is_close(vs1.middle(), vs2.middle(), tol) &&
         is_close(vs1.end(), vs2.end(), tol);
}

/// \brief Element-wise approximate equality (within a relaxed tolerance)
template <ValidVSegmentType T1, ValidVSegmentType T2>
KOKKOS_FUNCTION constexpr bool is_approx_close(
    const T1& vs1, const T2& vs2,
    typename T1::value_type tol = get_relaxed_comparison_tolerance<typename T1::value_type, typename T2::value_type>()) {
  return is_close(vs1, vs2, tol);
}

/// \brief OStream operator
template <ValidVSegmentType VSegmentType>
std::ostream& operator<<(std::ostream& os, const VSegmentType& v_segment) {
  os << "{" << v_segment.start() << "->" << v_segment.middle() << "->" << v_segment.end() << "}";
  return os;
}

/// @}

//! \name Point visitation
//@{

/// \brief Visit each geometric point of a VSegment (start, middle, end).
template <ValidVSegmentType T, typename Functor>
KOKKOS_INLINE_FUNCTION void for_each_point(const T& vs, Functor&& f) {
  f(vs.start());
  f(vs.middle());
  f(vs.end());
}

/// \brief Visit and mutate each geometric point of a VSegment.
template <ValidVSegmentType T, typename Functor>
KOKKOS_INLINE_FUNCTION void for_each_point_mutable(T& vs, Functor&& f) {
  f(vs.start());
  f(vs.middle());
  f(vs.end());
}

//@}

}  // namespace mundy

#endif  // MUNDY_GEOM_PRIMITIVES_VSEGMENT_HPP_
