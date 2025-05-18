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

#ifndef MUNDY_MECH_PRIMITIVES_TORSIONALSPRING_HPP_
#define MUNDY_MECH_PRIMITIVES_TORSIONALSPRING_HPP_

// External
#include <Kokkos_Core.hpp>

// C++ core
#include <iostream>
#include <stdexcept>
#include <utility>

// Mundy
#include <mundy_geom/primitives/VSegment.hpp>  // for mundy::geom::VSegment

namespace mundy {

namespace mech {

/// \brief A hookean spring between two points with a rest angle and spring constant
template <typename Scalar, mundy::geom::ValidVSegmentType VSegmentType = mundy::geom::VSegment<Scalar>,
          typename OwnershipType = mundy::math::Ownership::Owns>
class TorsionalSpring {
  static_assert(std::is_same_v<typename VSegmentType::scalar_t, Scalar>,
                "The scalar_t of the VSegmentType must match the Scalar type.");
  static_assert(std::is_same_v<typename VSegmentType::ownership_t, OwnershipType>,
                "The ownership type of the VSegmentType must match the OwnershipType.\n"
                "This is somewhat restrictive, and we may want to relax this constraint in the future.\n"
                "If you need to use a different ownership type, please let us know and we'll remove this restriction.");

 public:
  //! \name Type aliases
  //@{

  /// \brief Our scalar type
  using scalar_t = Scalar;

  /// \brief Our line segment type
  using v_segment_t = VSegmentType;
  //@}

  //! \name Constructors and destructor
  //@{

  /// \brief Default constructor for owning TorsionalSprings. Default initialize the line segment and set the spring
  /// constant and rest angle to -1.
  KOKKOS_FUNCTION
  TorsionalSpring()
    requires std::is_same_v<OwnershipType, mundy::math::Ownership::Owns>
      : v_segment_(), rest_angle_(static_cast<scalar_t>(-1)), spring_constant_(static_cast<scalar_t>(-1)) {
  }

  /// \brief No default constructor for viewing TorsionalSpringss.
  KOKKOS_FUNCTION
  TorsionalSpring()
    requires std::is_same_v<OwnershipType, mundy::math::Ownership::Views>
  = delete;

  /// \brief Constructor to initialize the line segment, rest angle, and spring constant.
  KOKKOS_FUNCTION
  TorsionalSpring(const v_segment_t& v_segment, const scalar_t& rest_angle, const scalar_t& spring_constant)
      : v_segment_(v_segment), rest_angle_(rest_angle), spring_constant_(spring_constant) {
  }

  /// \brief Constructor to initialize the start and end points.
  /// \param[in] start The start of the TorsionalSpring.
  /// \param[in] end The end of the TorsionalSpring.
  template <mundy::geom::ValidVSegmentType OtherVSegmentType>
  KOKKOS_FUNCTION TorsionalSpring(const OtherVSegmentType& v_segment, const scalar_t& rest_angle,
                                  const scalar_t& spring_constant)
    requires(!std::is_same_v<OtherVSegmentType, v_segment_t>)
      : v_segment_(v_segment), rest_angle_(rest_angle), spring_constant_(spring_constant) {
  }

  /// \brief Destructor
  KOKKOS_DEFAULTED_FUNCTION
  ~TorsionalSpring() = default;

  /// \brief Deep copy constructor
  KOKKOS_FUNCTION
  TorsionalSpring(const TorsionalSpring<scalar_t, v_segment_t, ownership_t>& other)
      : v_segment_(other.v_segment_), rest_angle_(other.rest_angle_), spring_constant_(other.spring_constant_) {
  }

  /// \brief Deep copy constructor
  template <typename OtherTorsionalSpringType>
  KOKKOS_FUNCTION TorsionalSpring(const OtherTorsionalSpringType& other)
    requires(!std::is_same_v<OtherTorsionalSpringType, TorsionalSpring<scalar_t, v_segment_t, ownership_t>>)
      : v_segment_(other.v_segment_), rest_angle_(other.rest_angle_), spring_constant_(other.spring_constant_) {
  }

  /// \brief Deep move constructor
  KOKKOS_FUNCTION
  TorsionalSpring(TorsionalSpring<scalar_t, v_segment_t, ownership_t>&& other)
      : v_segment_(std::move(other.v_segment_)),
        rest_angle_(std::move(other.rest_angle_)),
        spring_constant_(std::move(other.spring_constant_)) {
  }

  /// \brief Deep move constructor
  template <typename OtherTorsionalSpringType>
  KOKKOS_FUNCTION TorsionalSpring(OtherTorsionalSpringType&& other)
    requires(!std::is_same_v<OtherTorsionalSpringType, TorsionalSpring<scalar_t, v_segment_t, ownership_t>>)
      : v_segment_(std::move(other.v_segment_)),
        rest_angle_(std::move(other.rest_angle_)),
        spring_constant_(std::move(other.spring_constant_)) {
  }
  //@}

  //! \name Operators
  //@{

  /// \brief Copy assignment operator
  KOKKOS_FUNCTION
  TorsionalSpring<scalar_t, v_segment_t, ownership_t>& operator=(
      const TorsionalSpring<scalar_t, v_segment_t, ownership_t>& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    v_segment_ = other.v_segment_;
    rest_angle_ = other.rest_angle_;
    spring_constant_ = other.spring_constant_;
    return *this;
  }

  /// \brief Copy assignment operator
  template <typename OtherTorsionalSpringType>
  KOKKOS_FUNCTION TorsionalSpring<scalar_t, v_segment_t, ownership_t>& operator=(const OtherTorsionalSpringType& other)
    requires(!std::is_same_v<OtherTorsionalSpringType, TorsionalSpring<scalar_t, v_segment_t, ownership_t>>)
  {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    v_segment_ = other.v_segment_;
    rest_angle_ = other.rest_angle_;
    spring_constant_ = other.spring_constant_;
    return *this;
  }

  /// \brief Move assignment operator
  KOKKOS_FUNCTION
  TorsionalSpring<scalar_t, v_segment_t, ownership_t>& operator=(
      TorsionalSpring<scalar_t, v_segment_t, ownership_t>&& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    v_segment_ = std::move(other.v_segment_);
    rest_angle_ = std::move(other.rest_angle_);
    spring_constant_ = std::move(other.spring_constant_);
    return *this;
  }

  /// \brief Move assignment operator
  template <typename OtherTorsionalSpringType>
  KOKKOS_FUNCTION TorsionalSpring<scalar_t, v_segment_t, ownership_t>& operator=(OtherTorsionalSpringType&& other)
    requires(!std::is_same_v<OtherTorsionalSpringType, TorsionalSpring<scalar_t, v_segment_t, ownership_t>>)
  {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    v_segment_ = std::move(other.v_segment_);
    rest_angle_ = std::move(other.rest_angle_);
    spring_constant_ = std::move(other.spring_constant_);
    return *this;
  }
  //@}

  //! \name Accessors
  //@{

  /// \brief Accessor for the line segment
  KOKKOS_FUNCTION
  const v_segment_t& v_segment() const {
    return v_segment_;
  }

  /// \brief Accessor for the line segment
  KOKKOS_FUNCTION
  v_segment_t& v_segment() {
    return v_segment_;
  }

  /// \brief Accessor for the rest angle
  KOKKOS_FUNCTION
  const scalar_t& rest_angle() const {
    return rest_angle_;
  }

  /// \brief Accessor for the rest angle
  KOKKOS_FUNCTION
  scalar_t& rest_angle() {
    return rest_angle_;
  }

  /// \brief Accessor for the spring constant
  KOKKOS_FUNCTION
  const scalar_t& spring_constant() const {
    return spring_constant_;
  }

  /// \brief Accessor for the spring constant
  KOKKOS_FUNCTION
  scalar_t& spring_constant() {
    return spring_constant_;
  }
  //@}

  //! \name Setters
  //@{

  /// \brief Set the line segment
  /// \param[in] v_segment The new line segment.
  template <mundy::geom::ValidVSegmentType OtherVSegmentType>
  KOKKOS_FUNCTION void set_v_segment(const OtherVSegmentType& v_segment) {
    v_segment_ = v_segment;
  }

  /// \brief Set the rest angle
  /// \param[in] rest_angle The new rest angle.
  KOKKOS_FUNCTION
  void set_rest_angle(const scalar_t& rest_angle) {
    rest_angle_ = rest_angle;
  }

  /// \brief Set the spring constant
  /// \param[in] spring_constant The new spring constant.
  KOKKOS_FUNCTION
  void set_spring_constant(const scalar_t& spring_constant) {
    spring_constant_ = spring_constant;
  }
  //@}

 private:
  v_segment_t v_segment_;
  std::conditional_t<std::is_same_v<ownership_t, mundy::math::Ownership::Owns>, scalar_t, scalar_t&> rest_angle_;
  std::conditional_t<std::is_same_v<ownership_t, mundy::math::Ownership::Owns>, scalar_t, scalar_t&> spring_constant_;
};  // class TorsionalSpring

/// @brief Type trait to determine if a type is a TorsionalSpring
template <typename T>
struct is_torsional_spring : std::false_type {};
//
template <typename Scalar, mundy::geom::ValidVSegmentType VSegmentType, typename OwnershipType>
struct is_torsional_spring<TorsionalSpring<Scalar, VSegmentType, OwnershipType>> : std::true_type {};
//
template <typename Scalar, mundy::geom::ValidVSegmentType VSegmentType, typename OwnershipType>
struct is_torsional_spring<const TorsionalSpring<Scalar, VSegmentType, OwnershipType>> : std::true_type {};
//
template <typename T>
inline constexpr bool is_torsional_spring_v = is_torsional_spring<T>::value;

/// @brief Concept to check if a type is a valid TorsionalSpring type
template <typename TorsionalSpringType>
concept ValidTorsionalSpringType = mundy::geom::ValidTorsionalSpringType<TorsionalSpringType>;

static_assert(ValidTorsionalSpringType<TorsionalSpring<float>> &&
                  ValidTorsionalSpringType<const TorsionalSpring<float>> &&
                  ValidTorsionalSpringType<TorsionalSpring<double>> &&
                  ValidTorsionalSpringType<const TorsionalSpring<double>>,
              "TorsionalSpring should satisfy the ValidTorsionalSpringType concept");

}  // namespace mech

}  // namespace mundy

#endif  // MUNDY_MECH_PRIMITIVES_TORSIONALSPRING_HPP_
