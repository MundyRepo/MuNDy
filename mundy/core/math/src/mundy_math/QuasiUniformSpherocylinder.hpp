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

#ifndef MUNDY_MATH_QUASIUNIFORMSPHEROCYLINDER_HPP_
#define MUNDY_MATH_QUASIUNIFORMSPHEROCYLINDER_HPP_

/// \file QuasiUniformSpherocylinder.hpp
/// \brief Surface quadrature with quasi-uniform nodes on a spherocylinder aligned with the z axis.
///
/// The cylinder uses midpoint rings with alternating half-longitude offsets. Each cap uses a Fibonacci spiral,
/// with midpoints equally spaced in |cos(theta)|. Weights are constant within each patch: 2 pi r L / (NTheta NZ)
/// on the cylinder and 2 pi r^2 / NCap on each hemisphere. Constants integrate to the surface area, up to rounding;
/// the rule favors spatial coverage rather than Gaussian polynomial exactness.
///
/// Points are flattened (x, y, z) triples, ordered cylinder rings from negative to positive z, north cap, south cap.
/// There are NTheta NZ + 2 NCap points. Radius is positive and L is the nonnegative cylindrical length, excluding the
/// caps. A sphere has L = 0 and may use NZ = 0; retained cylinder rings then have zero weight.
///
/// The fixed-count class computes its reference points at compile time and maps them to the requested geometry.
/// Run-time overloads build the same nodes in std::vectors or Kokkos views, using explicit patch counts or an
/// approximate target count. The latter keeps at least eight longitudes and eight nodes per cap, so small targets
/// are rounded up. Its counts depend on aspect ratio, not overall scale.

// External
#include <Kokkos_Core.hpp>  // for Kokkos::Array, KOKKOS_INLINE_FUNCTION

// C++ core
#include <type_traits>  // for std::is_same_v
#include <vector>       // for std::vector

// Mundy
#include <mundy_math/impl/QuasiUniformSpherocylinderImpl.hpp>  // for reference points, patch weights, count selection
#include <mundy_utils/requires.hpp>                            // for MUNDY_REQUIRES

namespace mundy {

/// \brief A fixed-count surface rule; points, weights, and integration support constexpr and device evaluation.
///
/// Geometry must be finite, with radius > 0 and cylinder_length >= 0. Positive length requires NZ > 0.
/// Invalid geometry throws on the host and aborts on the device, following Mundy's requirement checks.
/// Large rules should use the run-time builders to avoid a large compile-time table and per-thread array storage.
template <class Scalar, unsigned NTheta, unsigned NZ, unsigned NCap>
class QuasiUniformSpherocylinder {
  static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>,
                "QuasiUniformSpherocylinder: Scalar must be float or double.");
  static_assert(impl::valid_spherocylinder_counts(NTheta, NZ, NCap),
                "QuasiUniformSpherocylinder: invalid or overflowing patch counts.");

 public:
  using value_type = Scalar;
  static constexpr unsigned num_longitudes = NTheta;
  static constexpr unsigned num_cylinder_rings = NZ;
  static constexpr unsigned num_cap_points = NCap;
  static constexpr unsigned num_points = NTheta * NZ + 2 * NCap;

  /// \brief Physical (x, y, z) triples, in cylinder, north-cap, south-cap order.
  KOKKOS_INLINE_FUNCTION static constexpr Kokkos::Array<Scalar, 3 * num_points> points(Scalar radius,
                                                                                       Scalar cylinder_length) {
    impl::validate_spherocylinder_geometry(NZ, radius, cylinder_length);
    constexpr auto reference = reference_points();
    Kokkos::Array<Scalar, 3 * num_points> points{};
    for (unsigned i = 0; i < num_points; ++i) {
      const auto point = impl::spherocylinder_physical_point(i < NTheta * NZ, radius, cylinder_length,  //
                                                             reference[3 * i],                          //
                                                             reference[3 * i + 1],                      //
                                                             reference[3 * i + 2]);
      for (unsigned d = 0; d < 3; ++d) {
        points[3 * i + d] = point[d];
      }
    }
    return points;
  }

  /// \brief Surface-area weights, in the order of points(). No point table is needed to compute the weights.
  KOKKOS_INLINE_FUNCTION static constexpr Kokkos::Array<Scalar, num_points> weights(Scalar radius,
                                                                                    Scalar cylinder_length) {
    impl::validate_spherocylinder_geometry(NZ, radius, cylinder_length);
    Kokkos::Array<Scalar, num_points> weights{};
    const Scalar cylinder = cylinder_weight(radius, cylinder_length);
    const Scalar cap = impl::spherocylinder_patch_weight(NCap, false, radius, cylinder_length);
    for (unsigned i = 0; i < num_points; ++i) {
      weights[i] = i < NTheta * NZ ? cylinder : cap;
    }
    return weights;
  }

  /// \brief sum_i w_i f(x_i, y_i, z_i), approximating the physical surface integral.
  template <class Function>
  KOKKOS_INLINE_FUNCTION static constexpr auto integrate(Scalar radius, Scalar cylinder_length, const Function& f) {
    impl::validate_spherocylinder_geometry(NZ, radius, cylinder_length);
    constexpr auto reference = reference_points();
    const Scalar cylinder = cylinder_weight(radius, cylinder_length);
    const Scalar cap = impl::spherocylinder_patch_weight(NCap, false, radius, cylinder_length);
    auto point =
        impl::spherocylinder_physical_point(NZ > 0, radius, cylinder_length, reference[0], reference[1], reference[2]);
    auto sum = (NZ > 0 ? cylinder : cap) * f(point[0], point[1], point[2]);
    for (unsigned i = 1; i < num_points; ++i) {
      point = impl::spherocylinder_physical_point(i < NTheta * NZ, radius, cylinder_length,  //
                                                  reference[3 * i],                          //
                                                  reference[3 * i + 1],                      //
                                                  reference[3 * i + 2]);
      sum += (i < NTheta * NZ ? cylinder : cap) * f(point[0], point[1], point[2]);
    }
    return sum;
  }

 private:
  KOKKOS_INLINE_FUNCTION static constexpr auto reference_points() {
    constexpr auto points = impl::make_quasi_uniform_spherocylinder_points<Scalar, NTheta, NZ, NCap>();
    return points;
  }

  KOKKOS_INLINE_FUNCTION static constexpr Scalar cylinder_weight(Scalar radius, Scalar length) {
    if constexpr (NZ == 0) {
      return Scalar(0);
    } else {
      return impl::spherocylinder_patch_weight(NTheta * NZ, true, radius, length);
    }
  }
};

/// \brief Build a rule with explicit patch counts into std::vectors, resizing points and weights as needed.
template <class Scalar>
void quasi_uniform_spherocylinder_rule(unsigned n_theta, unsigned n_z, unsigned n_cap, Scalar radius,
                                       Scalar cylinder_length, std::vector<Scalar>& points,
                                       std::vector<Scalar>& weights) {
  static_assert(std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>,
                "quasi_uniform_spherocylinder_rule: Scalar must be float or double.");
  impl::validate_spherocylinder_counts(n_theta, n_z, n_cap);
  impl::validate_spherocylinder_geometry(n_z, radius, cylinder_length);
  const unsigned num_points = n_theta * n_z + 2 * n_cap;
  points.resize(3 * num_points);
  weights.resize(num_points);
  for (unsigned i = 0; i < num_points; ++i) {
    impl::write_quasi_uniform_spherocylinder_point(n_theta, n_z, n_cap, i, radius, cylinder_length, points, weights);
  }
}

/// \brief Choose approximately equally spaced patch resolutions from a target count, then build the vector rule.
template <class Scalar>
void quasi_uniform_spherocylinder_rule(unsigned target, Scalar radius, Scalar cylinder_length,
                                       std::vector<Scalar>& points, std::vector<Scalar>& weights) {
  const auto counts = impl::quasi_uniform_spherocylinder_counts(target, radius, cylinder_length);
  quasi_uniform_spherocylinder_rule(counts.num_longitudes, counts.num_cylinder_rings, counts.num_cap_points, radius,
                                    cylinder_length, points, weights);
}

/// \brief Build into writable rank-1 Kokkos views, one point per thread on the points' execution space.
///
/// Called from the host. Resizes both views and returns after the rule is complete. Weights must be accessible
/// from the points' execution space; fixed-count rules can instead be evaluated directly inside a device kernel.
template <class PointsView, class WeightsView>
MUNDY_REQUIRES(Kokkos::is_view_v<PointsView>&& Kokkos::is_view_v<WeightsView>)
void quasi_uniform_spherocylinder_rule(unsigned n_theta, unsigned n_z, unsigned n_cap,
                                       typename PointsView::non_const_value_type radius,
                                       typename PointsView::non_const_value_type cylinder_length, PointsView& points,
                                       WeightsView& weights) {
  using scalar_t = typename PointsView::non_const_value_type;
  using execution_space = typename PointsView::execution_space;
  static_assert(PointsView::rank() == 1 && WeightsView::rank() == 1,
                "quasi_uniform_spherocylinder_rule: views must be rank 1.");
  static_assert(std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, double>,
                "quasi_uniform_spherocylinder_rule: Scalar must be float or double.");
  static_assert(std::is_same_v<scalar_t, typename PointsView::value_type> &&
                    std::is_same_v<scalar_t, typename WeightsView::value_type>,
                "quasi_uniform_spherocylinder_rule: views must be writable and have the same scalar type.");
  static_assert(Kokkos::SpaceAccessibility<execution_space, typename WeightsView::memory_space>::accessible,
                "quasi_uniform_spherocylinder_rule: weights must be accessible from the points' execution space.");
  impl::validate_spherocylinder_counts(n_theta, n_z, n_cap);
  impl::validate_spherocylinder_geometry(n_z, radius, cylinder_length);
  const unsigned num_points = n_theta * n_z + 2 * n_cap;
  Kokkos::resize(points, 3 * num_points);
  Kokkos::resize(weights, num_points);
  const execution_space space;
  Kokkos::parallel_for(
      "mundy::quasi_uniform_spherocylinder_rule", Kokkos::RangePolicy<execution_space>(space, 0, num_points),
      KOKKOS_LAMBDA(const unsigned i) {
        impl::write_quasi_uniform_spherocylinder_point(n_theta, n_z, n_cap, i, radius, cylinder_length, points,
                                                       weights);
      });
  space.fence("mundy::quasi_uniform_spherocylinder_rule");
}

/// \brief Choose patch resolutions from a target count, then build the rule in Kokkos views.
template <class PointsView, class WeightsView>
MUNDY_REQUIRES(Kokkos::is_view_v<PointsView>&& Kokkos::is_view_v<WeightsView>)
void quasi_uniform_spherocylinder_rule(unsigned target, typename PointsView::non_const_value_type radius,
                                       typename PointsView::non_const_value_type cylinder_length, PointsView& points,
                                       WeightsView& weights) {
  const auto counts = impl::quasi_uniform_spherocylinder_counts(target, radius, cylinder_length);
  quasi_uniform_spherocylinder_rule(counts.num_longitudes, counts.num_cylinder_rings, counts.num_cap_points, radius,
                                    cylinder_length, points, weights);
}

}  // namespace mundy

#endif  // MUNDY_MATH_QUASIUNIFORMSPHEROCYLINDER_HPP_
