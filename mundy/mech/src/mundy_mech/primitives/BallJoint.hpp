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

#ifndef MUNDY_MECH_PRIMITIVES_BALLJOINT_HPP_
#define MUNDY_MECH_PRIMITIVES_BALLJOINT_HPP_

// External
#include <Kokkos_Core.hpp>

// C++ core
#include <iostream>
#include <stdexcept>
#include <utility>

// Mundy
#include <mundy_geom/primitives/LineSegment.hpp>  // for mundy::geom::LineSegment

namespace mundy {

namespace mech {

/// \brief A ball-and-socket joint between two points
///
/// The BallJoint is a simple mechanical joint that constrains two points to have zero separation.
/// It does not have any physical properties beyond those of its underlying geometry. Although,
/// depending on the use case, it will be augmented to include a finite spring constant (if imposed as a soft
/// constraint) or three Lagrange multipliers (if imposed as a hard constraint).
template <typename Scalar, mundy::geom::ValidLineSegmentType LineSegmentType = mundy::geom::LineSegment<Scalar>,
          typename OwnershipType = mundy::math::Ownership::Owns>
class BallJoint {
  static_assert(std::is_same_v<typename LineSegmentType::scalar_t, Scalar>,
                "The scalar_t of the LineSegmentType must match the Scalar type.");
  static_assert(std::is_same_v<typename LineSegmentType::ownership_t, OwnershipType>,
                "The ownership type of the LineSegmentType must match the OwnershipType.\n"
                "This is somewhat restrictive, and we may want to relax this constraint in the future.\n"
                "If you need to use a different ownership type, please let us know and we'll remove this restriction.");

 public:
  //! \name Type aliases
  //@{

  /// \brief Our scalar type
  using scalar_t = Scalar;

  /// \brief Our line segment type
  using line_segment_t = LineSegmentType;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor for owning BallJoints. Default initialize the line segment.
  KOKKOS_FUNCTION
  BallJoint()
    requires std::is_same_v<OwnershipType, mundy::math::Ownership::Owns>
      : line_segment_() {
  }

  /// \brief No default constructor for viewing BallJointss.
  KOKKOS_FUNCTION
  BallJoint()
    requires std::is_same_v<OwnershipType, mundy::math::Ownership::Views>
  = delete;

  /// \brief Constructor to initialize the underlying line segment.
  KOKKOS_FUNCTION
  BallJoint(const line_segment_t& line_segment) : line_segment_(line_segment) {
  }

  /// \brief Constructor to initialize the underlying line segment.
  template <mundy::geom::ValidLineSegmentType OtherLineSegmentType>
  KOKKOS_FUNCTION BallJoint(const OtherLineSegmentType& line_segment)
    requires(!std::is_same_v<OtherLineSegmentType, line_segment_t>)
      : line_segment_(line_segment) {
  }

  /// \brief Destructor
  KOKKOS_DEFAULTED_FUNCTION
  ~BallJoint() = default;

  /// \brief Deep copy constructor
  KOKKOS_FUNCTION
  BallJoint(const BallJoint<scalar_t, line_segment_t, ownership_t>& other) : line_segment_(other.line_segment_) {
  }

  /// \brief Deep copy constructor
  template <typename OtherBallJointType>
  KOKKOS_FUNCTION BallJoint(const OtherBallJointType& other)
    requires(!std::is_same_v<OtherBallJointType, BallJoint<scalar_t, line_segment_t, ownership_t>>)
      : line_segment_(other.line_segment_) {
  }

  /// \brief Deep move constructor
  KOKKOS_FUNCTION
  BallJoint(BallJoint<scalar_t, line_segment_t, ownership_t>&& other) : line_segment_(std::move(other.line_segment_)) {
  }

  /// \brief Deep move constructor
  template <typename OtherBallJointType>
  KOKKOS_FUNCTION BallJoint(OtherBallJointType&& other)
    requires(!std::is_same_v<OtherBallJointType, BallJoint<scalar_t, line_segment_t, ownership_t>>)
      : line_segment_(std::move(other.line_segment_)) {
  }
  //@}

  //! \name Operators
  //@{

  /// \brief Copy assignment operator
  KOKKOS_FUNCTION
  BallJoint<scalar_t, line_segment_t, ownership_t>& operator=(
      const BallJoint<scalar_t, line_segment_t, ownership_t>& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    line_segment_ = other.line_segment_;
    return *this;
  }

  /// \brief Copy assignment operator
  template <typename OtherBallJointType>
  KOKKOS_FUNCTION BallJoint<scalar_t, line_segment_t, ownership_t>& operator=(const OtherBallJointType& other)
    requires(!std::is_same_v<OtherBallJointType, BallJoint<scalar_t, line_segment_t, ownership_t>>)
  {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    line_segment_ = other.line_segment_;
    return *this;
  }

  /// \brief Move assignment operator
  KOKKOS_FUNCTION
  BallJoint<scalar_t, line_segment_t, ownership_t>& operator=(
      BallJoint<scalar_t, line_segment_t, ownership_t>&& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    line_segment_ = std::move(other.line_segment_);
    return *this;
  }

  /// \brief Move assignment operator
  template <typename OtherBallJointType>
  KOKKOS_FUNCTION BallJoint<scalar_t, line_segment_t, ownership_t>& operator=(OtherBallJointType&& other)
    requires(!std::is_same_v<OtherBallJointType, BallJoint<scalar_t, line_segment_t, ownership_t>>)
  {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    line_segment_ = std::move(other.line_segment_);
    return *this;
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Accessor for the line segment
  KOKKOS_FUNCTION
  const line_segment_t& line_segment() const {
    return line_segment_;
  }

  /// \brief Accessor for the line segment
  KOKKOS_FUNCTION
  line_segment_t& line_segment() {
    return line_segment_;
  }
  //@}

  //! \name Setters
  //@{

  /// \brief Set the line segment
  /// \param[in] line_segment The new line segment.
  template <mundy::geom::ValidLineSegmentType OtherLineSegmentType>
  KOKKOS_FUNCTION void set_line_segment(const OtherLineSegmentType& line_segment) {
    line_segment_ = line_segment;
  }
  //@}

 private:
  line_segment_t line_segment_;
};  // class BallJoint

/// @brief Type trait to determine if a type is a BallJoint
template <typename T>
struct is_ball_joint : std::false_type {};
//
template <typename Scalar, mundy::geom::ValidLineSegmentType LineSegmentType, typename OwnershipType>
struct is_ball_joint<BallJoint<Scalar, LineSegmentType, OwnershipType>> : std::true_type {};
//
template <typename Scalar, mundy::geom::ValidLineSegmentType LineSegmentType, typename OwnershipType>
struct is_ball_joint<const BallJoint<Scalar, LineSegmentType, OwnershipType>> : std::true_type {};
//
template <typename T>
inline constexpr bool is_ball_joint_v = is_ball_joint<T>::value;

/// @brief Concept to check if a type is a valid BallJoint type
template <typename BallJointType>
concept ValidBallJointType = mundy::geom::ValidLineSegmentType<BallJointType>;

static_assert(ValidBallJointType<BallJoint<float>> && ValidBallJointType<const BallJoint<float>> &&
                  ValidBallJointType<BallJoint<double>> && ValidBallJointType<const BallJoint<double>>,
              "BallJoint should satisfy the ValidBallJointType concept");

}  // namespace mech

}  // namespace mundy

#endif  // MUNDY_MECH_PRIMITIVES_BALLJOINT_HPP_
