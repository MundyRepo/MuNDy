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

#ifndef MUNDY_GEOM_PRIMITIVES_LINESEGMENT_HPP_
#define MUNDY_GEOM_PRIMITIVES_LINESEGMENT_HPP_

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

template <typename Scalar, ValidPointType StartPointType = Point<Scalar>, ValidPointType EndPointType = Point<Scalar>>
class LineSegment {
  static_assert(std::is_same_v<typename StartPointType::value_type, Scalar> &&
                    std::is_same_v<typename EndPointType::value_type, Scalar>,
                "The value_type of the StartPointType and EndPointType must match the Scalar type.");

 public:
  //! \name Type aliases
  //@{

  /// \brief Our scalar type
  using value_type = Scalar;

  /// \brief Our point type for the start point
  using start_point_t = StartPointType;

  /// \brief Our point type for the end point
  using end_point_t = EndPointType;

  static constexpr bool is_finite = true;

  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor for owning LineSegments. Default initialize the start and end points.
  KOKKOS_FUNCTION
  constexpr LineSegment()
      MUNDY_REQUIRES(HasNArgConstructor<start_point_t, value_type, 3>&& HasNArgConstructor<end_point_t, value_type, 3>)
      : start_(value_type(), value_type(), value_type()), end_(value_type(), value_type(), value_type()) {
  }

  /// \brief Constructor to initialize the start and end points.
  /// \param[in] start The start of the LineSegment.
  /// \param[in] end The end of the LineSegment.
  KOKKOS_FUNCTION
  constexpr LineSegment(const start_point_t& start, const end_point_t& end) : start_(start), end_(end) {
  }

  /// \brief Constructor to initialize the start and end points.
  /// \param[in] start The start of the LineSegment.
  /// \param[in] end The end of the LineSegment.
  template <ValidPointType OtherStartPointType, ValidPointType OtherEndPointType>
  KOKKOS_FUNCTION constexpr LineSegment(const OtherStartPointType& start, const OtherEndPointType& end)
      MUNDY_REQUIRES(!std::is_same_v<OtherStartPointType, start_point_t> ||
                     !std::is_same_v<OtherEndPointType, end_point_t>)
      : start_(start), end_(end) {
  }

  /// \brief Destructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr ~LineSegment() = default;

  /// \brief Deep copy constructor
  KOKKOS_FUNCTION
  constexpr LineSegment(const LineSegment<value_type, start_point_t, end_point_t>& other)
      : start_(other.start_), end_(other.end_) {
  }

  /// \brief Deep copy constructor
  template <typename OtherLineSegmentType>
  KOKKOS_FUNCTION constexpr LineSegment(const OtherLineSegmentType& other)
      MUNDY_REQUIRES(!std::is_same_v<OtherLineSegmentType, LineSegment<value_type, start_point_t, end_point_t>>)
      : start_(other.start_), end_(other.end_) {
  }

  /// \brief Deep move constructor
  KOKKOS_FUNCTION
  constexpr LineSegment(LineSegment<value_type, start_point_t, end_point_t>&& other)
      : start_(std::move(other.start_)), end_(std::move(other.end_)) {
  }

  /// \brief Deep move constructor
  template <typename OtherLineSegmentType>
  KOKKOS_FUNCTION constexpr LineSegment(OtherLineSegmentType&& other)
      MUNDY_REQUIRES(!std::is_same_v<OtherLineSegmentType, LineSegment<value_type, start_point_t, end_point_t>>)
      : start_(std::move(other.start_)), end_(std::move(other.end_)) {
  }
  //@}

  //! \name Operators
  //@{

  /// \brief Copy assignment operator
  KOKKOS_FUNCTION
  constexpr LineSegment<value_type, start_point_t, end_point_t>& operator=(
      const LineSegment<value_type, start_point_t, end_point_t>& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    start_ = other.start_;
    end_ = other.end_;
    return *this;
  }

  /// \brief Copy assignment operator
  template <typename OtherLineSegmentType>
  KOKKOS_FUNCTION constexpr LineSegment<value_type, start_point_t, end_point_t>& operator=(
      const OtherLineSegmentType& other)
      MUNDY_REQUIRES(!std::is_same_v<OtherLineSegmentType, LineSegment<value_type, start_point_t, end_point_t>>) {
    start_ = other.start_;
    end_ = other.end_;
    return *this;
  }

  /// \brief Move assignment operator
  KOKKOS_FUNCTION
  constexpr LineSegment<value_type, start_point_t, end_point_t>& operator=(
      LineSegment<value_type, start_point_t, end_point_t>&& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    start_ = std::move(other.start_);
    end_ = std::move(other.end_);
    return *this;
  }

  /// \brief Move assignment operator
  template <typename OtherLineSegmentType>
  KOKKOS_FUNCTION constexpr LineSegment<value_type, start_point_t, end_point_t>& operator=(OtherLineSegmentType&& other)
      MUNDY_REQUIRES(!std::is_same_v<OtherLineSegmentType, LineSegment<value_type, start_point_t, end_point_t>>) {
    start_ = std::move(other.start_);
    end_ = std::move(other.end_);
    return *this;
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Accessor for the start
  KOKKOS_FUNCTION
  constexpr const start_point_t& start() const {
    return start_;
  }

  /// \brief Accessor for the start
  KOKKOS_FUNCTION
  constexpr start_point_t& start() {
    return start_;
  }

  /// \brief Accessor for the end
  KOKKOS_FUNCTION
  constexpr const end_point_t& end() const {
    return end_;
  }

  /// \brief Accessor for the end
  KOKKOS_FUNCTION
  constexpr end_point_t& end() {
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
  //! \name Friends <3
  //@{

  // We must be friends with other LineSegment types to access their private members
  template <typename, ValidPointType, ValidPointType>
  friend class LineSegment;
  //@}

  start_point_t start_;
  end_point_t end_;
};

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
/// \brief Deduction guide for LineSegment
template <ValidPointType StartPointType, ValidPointType EndPointType>
LineSegment(StartPointType, EndPointType)
    -> LineSegment<typename StartPointType::value_type, StartPointType, EndPointType>;
#endif

/// @brief (Implementation) Type trait to determine if a type is a LineSegment
template <typename T>
struct imple_is_line_segment : std::false_type {};
//
template <typename Scalar, ValidPointType StartPointType, ValidPointType EndPointType>
struct imple_is_line_segment<LineSegment<Scalar, StartPointType, EndPointType>> : std::true_type {};

/// @brief Type trait to determine if a type is a LineSegment
template <typename T>
struct is_line_segment : imple_is_line_segment<std::remove_cv_t<T>> {};
//
template <typename T>
inline constexpr bool is_line_segment_v = is_line_segment<T>::value;

/// @brief Concept to check if a type is a valid LineSegment type
template <typename LineSegmentType>
concept ValidLineSegmentType = is_line_segment_v<LineSegmentType>;

static_assert(ValidLineSegmentType<LineSegment<float>> && ValidLineSegmentType<const LineSegment<float>> &&
                  ValidLineSegmentType<LineSegment<double>> && ValidLineSegmentType<const LineSegment<double>>,
              "LineSegment should satisfy the ValidLineSegmentType concept");
static_assert(ValidLineSegmentType<LineSegment<double, Point<double>, APoint<double, double*>>> &&
                  ValidLineSegmentType<LineSegment<double, APoint<double, double*>, Point<double>>>,
              "LineSegment should support different start and end point types");

//! \name Non-member functions for ValidLineSegmentType objects
//@{

/// \brief Element-wise approximate equality (within a tolerance)
template <ValidLineSegmentType T1, ValidLineSegmentType T2>
KOKKOS_FUNCTION constexpr bool is_close(
    const T1& ls1, const T2& ls2,
    typename T1::value_type tol = get_comparison_tolerance<typename T1::value_type, typename T2::value_type>()) {
  return is_close(ls1.start(), ls2.start(), tol) && is_close(ls1.end(), ls2.end(), tol);
}

/// \brief Element-wise approximate equality (within a relaxed tolerance)
template <ValidLineSegmentType T1, ValidLineSegmentType T2>
KOKKOS_FUNCTION constexpr bool is_approx_close(
    const T1& ls1, const T2& ls2,
    typename T1::value_type tol = get_relaxed_comparison_tolerance<typename T1::value_type, typename T2::value_type>()) {
  return is_close(ls1, ls2, tol);
}

/// \brief OStream operator
template <ValidLineSegmentType LineSegmentType>
std::ostream& operator<<(std::ostream& os, const LineSegmentType& line_segment) {
  os << "{" << line_segment.start() << "->" << line_segment.end() << "}";
  return os;
}

/// @}

//! \name Point visitation
//@{

/// \brief Visit each geometric point of a LineSegment (start, end).
template <ValidLineSegmentType T, typename Functor>
KOKKOS_INLINE_FUNCTION void for_each_point(const T& ls, Functor&& f) {
  f(ls.start());
  f(ls.end());
}

/// \brief Visit and mutate each geometric point of a LineSegment.
template <ValidLineSegmentType T, typename Functor>
KOKKOS_INLINE_FUNCTION void for_each_point_mutable(T& ls, Functor&& f) {
  f(ls.start());
  f(ls.end());
}

//@}

}  // namespace mundy

#endif  // MUNDY_GEOM_PRIMITIVES_LINESEGMENT_HPP_
