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
#include <mundy_geom/distance/Types.hpp>    // for mundy::geom::SharedNormalSigned, Euclidean
#include <mundy_geom/primitives/Point.hpp>  // for mundy::geom::Point
#include <mundy_math/Matrix3.hpp>           // for mundy::math::Matrix3
#include <mundy_math/Vector3.hpp>           // for mundy::math::Vector3

namespace mundy {

namespace geom {

namespace impl {

// XXX Move the following function somewhere else, this just to call something like Kokkos::floor but on a floating
// point number. This is also templated by both a Scalar integer and Scalar floating point type, for use later when
// needing this to run on a GPU, and having to consider 32 vs. 64-bit calculations.
template <typename ScalarInt, typename Scalar>
KOKKOS_INLINE_FUNCTION constexpr Scalar pbc_floor(Scalar x) {
  return static_cast<Scalar>(x < Scalar(0.0) ? static_cast<ScalarInt>(x - Scalar(0.5))
                                             : static_cast<ScalarInt>(x + Scalar(0.5)));
}

}  // namespace impl

template <typename Scalar>
class FreeSpaceMetric {
 public:
  /// \brief Distance vector between two points in free space (from point1 to point2)
  KOKKOS_INLINE_FUNCTION
  constexpr Point<Scalar> operator()(const Point<Scalar>& point1, const Point<Scalar>& point2) const {
    return point2 - point1;
  }
};

template <typename Scalar>
class PeriodicSpaceMetric {
 public:
  /// \brief Default constructor
  constexpr PeriodicSpaceMetric() = default;

  /// \brief Constructor with unit cell matrix
  explicit constexpr PeriodicSpaceMetric(const mundy::math::Matrix3<Scalar>& h)
      : h_(h), h_inv_(mundy::math::inverse(h)) {
  }

  /// \brief Set the unit cell matrix
  void constexpr set_unit_cell_matrix(const mundy::math::Matrix3<Scalar>& h) {
    h_ = h;
    h_inv_ = mundy::math::inverse(h_);
  }

  /// \brief Distance vector between two points in periodic space (from point1 to point2)
  KOKKOS_INLINE_FUNCTION
  constexpr Point<Scalar> operator()(const Point<Scalar>& point1, const Point<Scalar>& point2) const {
    // Convert to fractional coordinates
    mundy::math::Vector3<Scalar> point1_scaled = h_inv_ * point1;
    mundy::math::Vector3<Scalar> point2_scaled = h_inv_ * point2;

    // Wrap the scaled coordinates back into the unit cell
    auto wrap_scaled_coordinates = [](double xyz) -> double {
      // Guard against numerical errors that may cause the scaled coordinates to be exactly 1.0
      // via multiplying by a boolean mask
      bool is_safe = Kokkos::fabs(xyz - 1.0) >= mundy::math::get_zero_tolerance<double>();
      return is_safe * (xyz - impl::pbc_floor<int64_t, double>(xyz));
    };
    point1_scaled = apply(wrap_scaled_coordinates, point1_scaled);
    point2_scaled = apply(wrap_scaled_coordinates, point2_scaled);

    // Now we can do the separation vector from the scaled coordinates and put it back in real space at the end.
    mundy::math::Vector3<Scalar> ds = point2_scaled - point1_scaled;
    ds = apply([](double xyz) -> double { return xyz - impl::pbc_floor<int64_t, double>(xyz); }, ds);

    return h_ * ds;
  }

 private:
  mundy::math::Matrix3<Scalar> h_;      ///< Unit cell matrix
  mundy::math::Matrix3<Scalar> h_inv_;  ///< Inverse of the unit cell matrix
};  // PeriodicSpaceMetric

template <typename Scalar>
class PeriodicScaledSpaceMetric {
 public:
  /// \brief Default constructor
  constexpr PeriodicScaledSpaceMetric() = default;

  /// \brief Constructor with unit cell matrix
  explicit constexpr PeriodicScaledSpaceMetric(const mundy::math::Vector3<Scalar>& cell_size)
      : scale_(cell_size),
        scale_inv_(Scalar(1.0) / cell_size[0], Scalar(1.0) / cell_size[1], Scalar(1.0) / cell_size[2]) {
  }

  /// \brief Set the cell size
  void set_cell_size(const mundy::math::Vector3<Scalar>& cell_size) {
    scale_ = cell_size;
    scale_inv_.set(Scalar(1.0) / cell_size[0], Scalar(1.0) / cell_size[1], Scalar(1.0) / cell_size[2]);
  }

  /// \brief Distance vector between two points in periodic space (from point1 to point2)
  KOKKOS_INLINE_FUNCTION
  constexpr Point<Scalar> operator()(const Point<Scalar>& point1, const Point<Scalar>& point2) const {
    // Convert to fractional coordinates
    mundy::math::Vector3<Scalar> point1_scaled = mundy::math::elementwise_multiply(scale_inv_, point1);
    mundy::math::Vector3<Scalar> point2_scaled = mundy::math::elementwise_multiply(scale_inv_, point2);

    // Wrap the scaled coordinates back into the unit cell
    auto wrap_scaled_coordinates = [](double xyz) -> double {
      // Guard against numerical errors that may cause the scaled coordinates to be exactly 1.0
      // via multiplying by a boolean mask
      bool is_safe = Kokkos::fabs(xyz - 1.0) >= mundy::math::get_zero_tolerance<double>();
      return is_safe * (xyz - impl::pbc_floor<int64_t, double>(xyz));
    };
    point1_scaled = apply(wrap_scaled_coordinates, point1_scaled);
    point2_scaled = apply(wrap_scaled_coordinates, point2_scaled);

    // Now we can do the separation vector from the scaled coordinates and put it back in real space at the end.
    mundy::math::Vector3<Scalar> ds = point2_scaled - point1_scaled;
    ds = apply([](double xyz) -> double { return xyz - impl::pbc_floor<int64_t, double>(xyz); }, ds);

    return mundy::math::elementwise_multiply(scale_, ds);
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
