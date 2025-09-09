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

#ifndef MUNDY_GEOM_DISTANCE_METRICS_HPP_
#define MUNDY_GEOM_DISTANCE_METRICS_HPP_

/// \brief Distance metrics
///
/// These structures are used for calculating the distance between two points in a given geometric space, including free
/// space.

// External libs
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_geom/primitives.hpp>      // for mundy::geom::Point, ...
#include <mundy_math/Matrix3.hpp>         // for mundy::math::Matrix3
#include <mundy_math/Quaternion.hpp>      // for mundy::math::Quaternion
#include <mundy_math/Vector3.hpp>         // for mundy::math::Vector3
#include <mundy_math/Tolerance.hpp>       // for mundy::math::get_zero_tolerance

namespace mundy {

namespace geom {

class FreeSpaceMetric {
 public:
  /// \brief Distance vector between two points in free space (from point1 to point2)
  template <typename Scalar>
  KOKKOS_INLINE_FUNCTION constexpr Point<Scalar> operator()(const Point<Scalar>& point1,
                                                            const Point<Scalar>& point2) const {
    return point2 - point1;
  }
};

template <typename Scalar>
class PeriodicSpaceMetric {
 public:
  /// \brief Default constructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr PeriodicSpaceMetric() = default;

  /// \brief Constructor with unit cell matrix
  KOKKOS_INLINE_FUNCTION
  explicit constexpr PeriodicSpaceMetric(const mundy::math::Matrix3<Scalar>& h)
      : h_(h), h_inv_(mundy::math::inverse(h)) {
  }

  /// \brief Set the unit cell matrix
  KOKKOS_INLINE_FUNCTION
  void constexpr set_unit_cell_matrix(const mundy::math::Matrix3<Scalar>& h) {
    h_ = h;
    h_inv_ = mundy::math::inverse(h_);
  }

  /// \brief Distance vector between two points in periodic space (from point1 to point2)
  KOKKOS_INLINE_FUNCTION
  constexpr Point<Scalar> operator()(const Point<Scalar>& point1, const Point<Scalar>& point2) const {
    // Convert the difference to fractional coordinates
    auto ds_frac = h_inv_ * (point2 - point1);

    // Minimum-image convention: wrap the fractional coordinates into the unit cell
    ds_frac = apply([](Scalar x) { return x - Kokkos::round(x); }, ds_frac);

    // Map the fractional coordinates back to real space
    return h_ * ds_frac;
  }

 private:
  mundy::math::Matrix3<Scalar> h_;      ///< Unit cell matrix
  mundy::math::Matrix3<Scalar> h_inv_;  ///< Inverse of the unit cell matrix
};  // PeriodicSpaceMetric

template <typename Scalar>
class PeriodicScaledSpaceMetric {
 public:
  /// \brief Default constructor
  KOKKOS_DEFAULTED_FUNCTION
  constexpr PeriodicScaledSpaceMetric() = default;

  /// \brief Constructor with unit cell matrix
  KOKKOS_INLINE_FUNCTION
  explicit constexpr PeriodicScaledSpaceMetric(const mundy::math::Vector3<Scalar>& cell_size)
      : scale_(cell_size), scale_inv_(Scalar(1.0) / scale_[0], Scalar(1.0) / scale_[1], Scalar(1.0) / scale_[2]) {
  }

  /// \brief Set the cell size
  KOKKOS_INLINE_FUNCTION
  void set_cell_size(const mundy::math::Vector3<Scalar>& cell_size) {
    scale_ = cell_size;
    scale_inv_.set(Scalar(1.0) / scale_[0], Scalar(1.0) / scale_[1], Scalar(1.0) / scale_[2]);
  }

  /// \brief Distance vector between two points in periodic space (from point1 to point2)
  KOKKOS_INLINE_FUNCTION
  constexpr Point<Scalar> operator()(const Point<Scalar>& point1, const Point<Scalar>& point2) const {
    // Convert the difference to fractional coordinates
    auto ds_frac = mundy::math::elementwise_multiply(scale_inv_, point2 - point1);

    // Minimum-image convention: wrap the fractional coordinates into the unit cell
    ds_frac = apply([](Scalar x) { return x - Kokkos::round(x); }, ds_frac);

    // Map the fractional coordinates back to real space
    return mundy::math::elementwise_multiply(scale_, ds_frac);
  }

 private:
  mundy::math::Vector3<Scalar> scale_;      ///< Unit cell scaling factors
  mundy::math::Vector3<Scalar> scale_inv_;  ///< Inverse of the scaling factors
};  // PeriodicScaledSpaceMetric

//! \name Non-member constructors
//@{

/// \brief Create a periodic space metric from a unit cell size
template <typename Scalar>
KOKKOS_INLINE_FUNCTION constexpr PeriodicSpaceMetric<Scalar> periodic_metric_from_unit_cell(
    const mundy::math::Vector3<Scalar>& cell_size) {
  auto h = mundy::math::Matrix3<Scalar>::identity();
  h(0, 0) = cell_size[0];
  h(1, 1) = cell_size[1];
  h(2, 2) = cell_size[2];
  return PeriodicSpaceMetric<Scalar>{std::move(h)};
}

/// \brief Create a periodic scaled space metric from a unit cell size
template <typename Scalar>
KOKKOS_INLINE_FUNCTION constexpr PeriodicScaledSpaceMetric<Scalar> periodic_scaled_metric_from_unit_cell(
    const mundy::math::Vector3<Scalar>& cell_size) {
  return PeriodicScaledSpaceMetric<Scalar>{cell_size};
}
//@}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_DISTANCE_METRICS_HPP_
