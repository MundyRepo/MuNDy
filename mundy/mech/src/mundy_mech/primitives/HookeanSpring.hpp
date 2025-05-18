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

#ifndef MUNDY_MECH_PRIMITIVES_HOOKEANSPRING_HPP_
#define MUNDY_MECH_PRIMITIVES_HOOKEANSPRING_HPP_

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

/// \brief A hookean spring between two points with a rest length and spring constant
template <typename Scalar, mundy::geom::ValidLineSegmentType LineSegmentType = mundy::geom::LineSegment<Scalar>,
          typename OwnershipType = mundy::math::Ownership::Owns>
class HookeanSpring {
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

  /// \brief Default constructor for owning HookeanSprings. Default initialize the line segment and set the spring
  /// constant and rest length to -1.
  KOKKOS_FUNCTION
  HookeanSpring()
    requires std::is_same_v<OwnershipType, mundy::math::Ownership::Owns>
      : line_segment_(), rest_length_(static_cast<scalar_t>(-1)), spring_constant_(static_cast<scalar_t>(-1)) {
  }

  /// \brief No default constructor for viewing HookeanSpringss.
  KOKKOS_FUNCTION
  HookeanSpring()
    requires std::is_same_v<OwnershipType, mundy::math::Ownership::Views>
  = delete;

  /// \brief Constructor to initialize the line segment, rest length, and spring constant.
  KOKKOS_FUNCTION
  HookeanSpring(const line_segment_t& line_segment, const scalar_t& rest_length, const scalar_t& spring_constant)
      : line_segment_(line_segment), rest_length_(rest_length), spring_constant_(spring_constant) {
  }

  /// \brief Constructor to initialize the start and end points.
  /// \param[in] start The start of the HookeanSpring.
  /// \param[in] end The end of the HookeanSpring.
  template <mundy::geom::ValidLineSegmentType OtherLineSegmentType>
  KOKKOS_FUNCTION HookeanSpring(const OtherLineSegmentType& line_segment, const scalar_t& rest_length,
                                const scalar_t& spring_constant)
    requires(!std::is_same_v<OtherLineSegmentType, line_segment_t>)
      : line_segment_(line_segment), rest_length_(rest_length), spring_constant_(spring_constant) {
  }

  /// \brief Destructor
  KOKKOS_DEFAULTED_FUNCTION
  ~HookeanSpring() = default;

  /// \brief Deep copy constructor
  KOKKOS_FUNCTION
  HookeanSpring(const HookeanSpring<scalar_t, line_segment_t, ownership_t>& other)
      : line_segment_(other.line_segment_), rest_length_(other.rest_length_), spring_constant_(other.spring_constant_) {
  }

  /// \brief Deep copy constructor
  template <typename OtherHookeanSpringType>
  KOKKOS_FUNCTION HookeanSpring(const OtherHookeanSpringType& other)
    requires(!std::is_same_v<OtherHookeanSpringType, HookeanSpring<scalar_t, line_segment_t, ownership_t>>)
      : line_segment_(other.line_segment_), rest_length_(other.rest_length_), spring_constant_(other.spring_constant_) {
  }

  /// \brief Deep move constructor
  KOKKOS_FUNCTION
  HookeanSpring(HookeanSpring<scalar_t, line_segment_t, ownership_t>&& other)
      : line_segment_(std::move(other.line_segment_)),
        rest_length_(std::move(other.rest_length_)),
        spring_constant_(std::move(other.spring_constant_)) {
  }

  /// \brief Deep move constructor
  template <typename OtherHookeanSpringType>
  KOKKOS_FUNCTION HookeanSpring(OtherHookeanSpringType&& other)
    requires(!std::is_same_v<OtherHookeanSpringType, HookeanSpring<scalar_t, line_segment_t, ownership_t>>)
      : line_segment_(std::move(other.line_segment_)),
        rest_length_(std::move(other.rest_length_)),
        spring_constant_(std::move(other.spring_constant_)) {
  }
  //@}

  //! \name Operators
  //@{

  /// \brief Copy assignment operator
  KOKKOS_FUNCTION
  HookeanSpring<scalar_t, line_segment_t, ownership_t>& operator=(
      const HookeanSpring<scalar_t, line_segment_t, ownership_t>& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    line_segment_ = other.line_segment_;
    rest_length_ = other.rest_length_;
    spring_constant_ = other.spring_constant_;
    return *this;
  }

  /// \brief Copy assignment operator
  template <typename OtherHookeanSpringType>
  KOKKOS_FUNCTION HookeanSpring<scalar_t, line_segment_t, ownership_t>& operator=(const OtherHookeanSpringType& other)
    requires(!std::is_same_v<OtherHookeanSpringType, HookeanSpring<scalar_t, line_segment_t, ownership_t>>)
  {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    line_segment_ = other.line_segment_;
    rest_length_ = other.rest_length_;
    spring_constant_ = other.spring_constant_;
    return *this;
  }

  /// \brief Move assignment operator
  KOKKOS_FUNCTION
  HookeanSpring<scalar_t, line_segment_t, ownership_t>& operator=(
      HookeanSpring<scalar_t, line_segment_t, ownership_t>&& other) {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    line_segment_ = std::move(other.line_segment_);
    rest_length_ = std::move(other.rest_length_);
    spring_constant_ = std::move(other.spring_constant_);
    return *this;
  }

  /// \brief Move assignment operator
  template <typename OtherHookeanSpringType>
  KOKKOS_FUNCTION HookeanSpring<scalar_t, line_segment_t, ownership_t>& operator=(OtherHookeanSpringType&& other)
    requires(!std::is_same_v<OtherHookeanSpringType, HookeanSpring<scalar_t, line_segment_t, ownership_t>>)
  {
    MUNDY_THROW_ASSERT(this != &other, std::invalid_argument, "Cannot assign to self");
    line_segment_ = std::move(other.line_segment_);
    rest_length_ = std::move(other.rest_length_);
    spring_constant_ = std::move(other.spring_constant_);
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

  /// \brief Accessor for the rest length
  KOKKOS_FUNCTION
  const scalar_t& rest_length() const {
    return rest_length_;
  }

  /// \brief Accessor for the rest length
  KOKKOS_FUNCTION
  scalar_t& rest_length() {
    return rest_length_;
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
  /// \param[in] line_segment The new line segment.
  template <mundy::geom::ValidLineSegmentType OtherLineSegmentType>
  KOKKOS_FUNCTION void set_line_segment(const OtherLineSegmentType& line_segment) {
    line_segment_ = line_segment;
  }

  /// \brief Set the rest length
  /// \param[in] rest_length The new rest length.
  KOKKOS_FUNCTION
  void set_rest_length(const scalar_t& rest_length) {
    rest_length_ = rest_length;
  }

  /// \brief Set the spring constant
  /// \param[in] spring_constant The new spring constant.
  KOKKOS_FUNCTION
  void set_spring_constant(const scalar_t& spring_constant) {
    spring_constant_ = spring_constant;
  }
  //@}

 private:
  line_segment_t line_segment_;
  std::conditional_t<std::is_same_v<ownership_t, mundy::math::Ownership::Owns>, scalar_t, scalar_t&> rest_length_;
  std::conditional_t<std::is_same_v<ownership_t, mundy::math::Ownership::Owns>, scalar_t, scalar_t&> spring_constant_;
};  // class HookeanSpring

/// @brief Type trait to determine if a type is a HookeanSpring
template <typename T>
struct is_hookean_spring : std::false_type {};
//
template <typename Scalar, mundy::geom::ValidLineSegmentType LineSegmentType, typename OwnershipType>
struct is_hookean_spring<HookeanSpring<Scalar, LineSegmentType, OwnershipType>> : std::true_type {};
//
template <typename Scalar, mundy::geom::ValidLineSegmentType LineSegmentType, typename OwnershipType>
struct is_hookean_spring<const HookeanSpring<Scalar, LineSegmentType, OwnershipType>> : std::true_type {};
//
template <typename T>
inline constexpr bool is_hookean_spring_v = is_hookean_spring<T>::value;

/// @brief Concept to check if a type is a valid HookeanSpring type
template <typename HookeanSpringType>
concept ValidHookeanSpringType = mundy::geom::ValidHookeanSpringType<HookeanSpringType>;

static_assert(ValidHookeanSpringType<HookeanSpring<float>> && ValidHookeanSpringType<const HookeanSpring<float>> &&
                  ValidHookeanSpringType<HookeanSpring<double>> && ValidHookeanSpringType<const HookeanSpring<double>>,
              "HookeanSpring should satisfy the ValidHookeanSpringType concept");

}  // namespace mech

}  // namespace mundy

#endif  // MUNDY_MECH_PRIMITIVES_HOOKEANSPRING_HPP_
