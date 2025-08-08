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

// External libs
#include <Kokkos_Core.hpp>

// Mundy
#include <mundy_geom/distance/Types.hpp>    // for mundy::geom::SharedNormalSigned, Euclidean
#include <mundy_geom/primitives/Point.hpp>  // for mundy::geom::Point
#include <mundy_math/Matrix3.hpp>           // for mundy::math::Matrix3
#include <mundy_math/Vector3.hpp>           // for mundy::math::Vector3

namespace mundy {

namespace geom {

/// \brief Distance metrics
///
/// These structures are used for calculating the distance between two points in a given geometric space, including free
/// space.
///

// XXX Move the following function somewhere else, this just to call something like Kokkos::floor but on a floating
// point number.
template <typename ScalarInt, typename Scalar>
KOKKOS_INLINE_FUNCTION Scalar pbc_floor(Scalar x) {
  return static_cast<Scalar>(x < Scalar(0.0) ? static_cast<ScalarInt>(x - Scalar(0.5))
                                             : static_cast<ScalarInt>(x + Scalar(0.5)));
}

/// \brief Compute the free space distance between two points
/// \param[in] point1 The first point
/// \param[in] point2 The second point
/// \param[out] sep The separation vector (from point1 to point2)
template <typename Scalar>
struct FreeSpaceMetric {
  KOKKOS_INLINE_FUNCTION
  void operator()(Point<Scalar>& sep, const Point<Scalar>& point1, const Point<Scalar>& point2) {
    sep = point2 - point1;
  }
};

/// \brief Compute the periodic space distance between two points
/// \param[in] point1 The first point
/// \param[in] point2 The second point
/// \param[out] sep The separation vector (from point1 to point2)
template <typename Scalar>
struct PeriodicSpaceMetric {
  KOKKOS_INLINE_FUNCTION
  void operator()(Point<Scalar>& sep, const Point<Scalar>& point1, const Point<Scalar>& point2) {
    // Convert to fractional coordinates
    mundy::math::Vector3<Scalar> point1_scaled = point1 * h_inv;
    mundy::math::Vector3<Scalar> point2_scaled = point2 * h_inv;

    // Wrap the scaled coordinates back into the unit cell
    for (size_t i = 0; i < 3; ++i) {
      point1_scaled[i] -= pbc_floor<int64_t, double>(point1_scaled[i]);
      point2_scaled[i] -= pbc_floor<int64_t, double>(point2_scaled[i]);

      // Guard against numerical errors that may cause the scaled coordinates to be exactly 1.0
      if (Kokkos::fabs(point1_scaled[i] - 1.0) < mundy::math::get_zero_tolerance<Scalar>()) {
        point1_scaled[i] = 0.0;
      }
      if (Kokkos::fabs(point2_scaled[i] - 1.0) < mundy::math::get_zero_tolerance<Scalar>()) {
        // Guard against numerical errors that may cause the scaled coordinates to be exactly 1.0
        point2_scaled[i] = 0.0;
      }
    }

    // Now we can do the separation vector from the scaled coordinates and put it back in real space at the end.
    mundy::math::Vector3<Scalar> ds = point2_scaled - point1_scaled;
    for (size_t i = 0; i < 3; ++i) {
      ds[i] -= pbc_floor<int64_t, double>(ds[i]);
    }

    sep = ds * h;
  }

  mundy::math::Matrix3<Scalar> h;      ///< Unit cell matrix
  mundy::math::Matrix3<Scalar> h_inv;  ///< Inverse of the unit cell matrix
};

template <typename Scalar>
KOKKOS_INLINE_FUNCTION constexpr void unit_cell_box(PeriodicSpaceMetric<Scalar>& metric,  //
                                                    const mundy::math::Vector3<Scalar>& cell_size) {
  metric.h = mundy::math::Matrix3<Scalar>::identity();
  metric.h(0, 0) = cell_size[0];
  metric.h(1, 1) = cell_size[1];
  metric.h(2, 2) = cell_size[2];
  metric.h_inv = mundy::math::inverse(metric.h);
}

}  // namespace geom

}  // namespace mundy

#endif  // MUNDY_GEOM_DISTANCE_METRICS_HPP_
